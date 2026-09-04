//! 影片去煙霧：把整支影片拆成一格一格，逐格丟進照片那套去煙演算法
//! （[`dehaze::remove_smoke`]），再重新編碼成一支新影片。
//!
//! 演算法完全沒有另做一份——影片這邊只負責「拆格、餵進去、收回來、接起來」，
//! 所以同一組滑桿在照片與影片上是同一種效果。
//!
//! 整條路是兩個 ffmpeg 行程夾著本程式：
//!
//! ```text
//! ffmpeg 解碼 ──rawvideo(rgb24)──▶ 本程式（N 條執行緒平行去煙）──rawvideo──▶ ffmpeg 編碼
//! ```
//!
//! 中間不落地：一格 1080p 的原始像素就要 6MB，一分鐘的片子攤成檔案是 11GB。
//! ffmpeg-sidecar 的事件通道是 `sync_channel(0)`（完全不緩衝），我們慢下來
//! 解碼那端就自己停住，記憶體因此只吃「同時在算的那一批」。
//!
//! 逐格獨立處理會不會閃爍？參數是**整支影片共用一組**（不像照片模組那樣
//! 每張自動量），煙霧層雖然是逐格估的，但煙本身是低頻、移動又連續，估出來
//! 的那一層跟著平順地變，實際看不出一格一格跳。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use image::RgbImage;

use crate::dehaze::{self, SmokeParams};
use crate::edit::{self, RegionMasks};
use crate::{ActiveGrade, RegionGrade};

/// 一支影片的基本資料（開檔時量一次，之後預覽與輸出都照它走）
#[derive(Clone, Debug)]
pub struct VideoInfo {
    /// 解碼出來的實際寬高。**不是**檔頭上寫的那一組——手機直拍的影片檔頭是
    /// 橫的、另外附一個旋轉矩陣，ffmpeg 解碼時已經幫我們轉正了，
    /// 所以這裡取的是「真的拿到手的那一格」的尺寸
    pub w: u32,
    pub h: u32,
    /// 影格率
    pub fps: f32,
    /// 片長（秒）
    pub secs: f64,
    /// 音訊的編碼名稱；None＝這支影片沒有聲音
    pub audio: Option<String>,
}

impl VideoInfo {
    /// 總格數（進度條的分母；VFR 的片子只是估計值）
    pub fn frames(&self) -> u64 {
        ((self.secs * self.fps as f64).round() as i64).max(1) as u64
    }

    /// 長邊（決定預覽要不要縮）
    pub fn long(&self) -> u32 {
        self.w.max(self.h)
    }
}

/// 背景執行緒回報給畫面的事
pub enum MovieMsg {
    /// 預覽底圖取好了；附上當時的時間點，來回拖時間軸時用來丟棄過期的結果
    Grabbed(f64, Result<RgbImage, String>),
    /// 預覽算完；附上當時的參數與時間點，用來判斷是不是已經過期
    Preview(SmokeParams, f64, RgbImage),
    /// 分區調色算完；附上當時的調色與時間點（一樣是拿來判斷有沒有過期），
    /// 外加那一格的三區權重——天際線只跟去煙結果有關，算一次就能一直沿用
    Graded(ActiveGrade, f64, RgbImage, Arc<RegionMasks>),
    /// 輸出進度：已經處理完幾格
    Progress(u64),
    /// 批次輸出：開始跑第 i 支（0 起算），附上它的檔名與總格數
    /// （進度條要照現在這一支的長度算，不是預覽那支）
    Started(usize, String, u64),
    /// 批次裡的某一支失敗了。**不中斷**其他支，最後一起顯示
    Failed(String),
    /// 輸出結束（成功時附上最後一支成品的路徑）
    Done(Result<PathBuf, String>),
}

/// 一次 ffmpeg 取格的結果
struct Grab {
    w: u32,
    h: u32,
    data: Vec<u8>,
    fps: f32,
    secs: f64,
    audio: Option<String>,
}

/// 從影片取一格回來（順便把檔案的基本資料一起帶出來）。
///
/// `seek` 是要取哪個時間點（None＝第一格），`max_long` 給了就把那一格等比
/// 縮到長邊不超過它。取格用的是 ffmpeg 自己的解碼，所以旋轉、色彩空間、
/// 各種容器格式都由它處理，我們拿到的一律是轉正過的 RGB
fn grab(path: &Path, seek: Option<f64>, max_long: Option<u32>) -> Result<Grab, String> {
    let mut cmd = FfmpegCommand::new();
    // -ss 放在 -i 前面是「關鍵格快轉」：長片跳到中段是瞬間的事，
    // 放在後面則要從頭解碼過去
    if let Some(t) = seek {
        cmd.args(["-ss", &format!("{:.3}", t.max(0.0))]);
    }
    cmd.input(path.to_string_lossy());
    if let Some(m) = max_long {
        // 用正方形的框配 decrease：不管影片是橫是直、有沒有旋轉矩陣，
        // 長邊都會落在 m。自己算寬高再寫死反而會在旋轉的片子上算錯
        cmd.args([
            "-vf",
            &format!("scale={m}:{m}:force_original_aspect_ratio=decrease"),
        ]);
    }
    cmd.args(["-frames:v", "1"]).rawvideo();

    let mut child = cmd.spawn().map_err(|e| format!("FFmpeg 啟動失敗：{e}"))?;
    let mut frame: Option<(u32, u32, Vec<u8>)> = None;
    let mut fps = 0.0f32;
    let mut secs = 0.0f64;
    let mut audio: Option<String> = None;
    let mut errs: Vec<String> = Vec::new();
    let iter = child
        .iter()
        .map_err(|e| format!("FFmpeg 輸出讀取失敗：{e}"))?;
    for ev in iter {
        match ev {
            FfmpegEvent::OutputFrame(f) => {
                if frame.is_none() {
                    frame = Some((f.width, f.height, f.data));
                }
            }
            // 只看輸入那一邊：輸出是我們自己指定的 rawvideo，問它沒有意義
            FfmpegEvent::ParsedInputStream(s) => {
                if let Some(v) = s.video_data() {
                    if fps <= 0.0 {
                        fps = v.fps;
                    }
                } else if s.is_audio() && audio.is_none() {
                    audio = Some(s.format.clone());
                }
            }
            FfmpegEvent::ParsedDuration(d) => {
                if secs <= 0.0 {
                    secs = d.duration;
                }
            }
            FfmpegEvent::Log(LogLevel::Error | LogLevel::Fatal, m) => errs.push(m),
            _ => {}
        }
    }
    let _ = child.wait();
    let Some((w, h, data)) = frame else {
        return Err(if errs.is_empty() {
            "讀不到這支影片的畫面（格式可能不支援）".into()
        } else {
            errs.join("\n")
        });
    };
    Ok(Grab {
        w,
        h,
        data,
        fps,
        secs,
        audio,
    })
}

/// 把 ffmpeg 吐回來的一格原始像素包成影像
fn to_image(w: u32, h: u32, data: Vec<u8>) -> Result<RgbImage, String> {
    RgbImage::from_raw(w, h, data).ok_or_else(|| "影格資料長度不對".to_string())
}

/// 開檔時量一次：寬高、影格率、片長、有沒有聲音
pub fn probe(path: &Path) -> Result<VideoInfo, String> {
    let g = grab(path, None, None)?;
    Ok(VideoInfo {
        w: g.w,
        h: g.h,
        // 有些容器的影格率讀不出來（如某些串流錄下來的檔）：當成 30，
        // 只影響進度分母與輸出的時間軸，畫面內容不受影響
        fps: if g.fps > 0.0 { g.fps } else { 30.0 },
        secs: g.secs.max(0.0),
        audio: g.audio,
    })
}

/// 取某個時間點的一格當預覽底圖（縮到長邊不超過 `max_long`）
pub fn preview_frame(path: &Path, secs: f64, max_long: Option<u32>) -> Result<RgbImage, String> {
    let g = grab(path, Some(secs), max_long)?;
    to_image(g.w, g.h, g.data)
}

/// 音訊要重編還是直接搬過去。
///
/// 直接搬（copy）不掉品質也不花時間，但來源的編碼要塞得進 MP4；
/// 塞不進的（如 WAV 裡的 PCM）就重編成 AAC。這件事在開跑**之前**就決定好——
/// 跑到最後才失敗，等於把使用者剛才等的幾十分鐘丟掉
fn audio_args(audio: Option<&str>) -> Vec<&'static str> {
    match audio {
        None => vec!["-an"],
        Some(codec) => {
            let c = codec.to_ascii_lowercase();
            if ["aac", "mp3", "ac3", "eac3", "alac", "mp2"].contains(&c.as_str()) {
                vec!["-c:a", "copy"]
            } else {
                vec!["-c:a", "aac", "-b:a", "192k"]
            }
        }
    }
}

/// 收編碼那一端的行程：畫面從 stdin 餵進去，log 由另一條執行緒收著
/// （不收的話 stderr 的管線塞滿，ffmpeg 就整個停住）
struct Encoder {
    stdin: Option<std::process::ChildStdin>,
    join: thread::JoinHandle<(bool, Vec<String>)>,
}

impl Encoder {
    /// 開一個從 stdin 吃 rawvideo、把聲音從原始檔搬過來的編碼行程
    fn start(
        src: &Path,
        dst: &Path,
        w: u32,
        h: u32,
        fps: f32,
        audio: Option<&str>,
        codec: &[&str],
    ) -> Result<Self, String> {
        let mut cmd = FfmpegCommand::new();
        cmd.arg("-y")
            // 第 0 個輸入：我們算好的畫面
            .args([
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{w}x{h}"),
                "-r",
                &format!("{fps}"),
                "-i",
                "pipe:0",
            ])
            // 第 1 個輸入：原始檔，只為了把聲音搬過來
            .input(src.to_string_lossy())
            .args(["-map", "0:v:0"]);
        if audio.is_some() {
            // 加問號＝找不到就算了，不要整個失敗
            cmd.args(["-map", "1:a:0?"]);
        }
        cmd.args([
            // yuv420p 只吃偶數邊長；來源是奇數（少見但存在）就切掉最後一列/行
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ])
        .args(codec)
        .args(audio_args(audio));
        if audio.is_some() {
            // 聲音比畫面長（或反過來）時以短的那個為準，結尾才不會拖一段黑畫面
            cmd.arg("-shortest");
        }
        cmd.output(dst.to_string_lossy());

        let mut child = cmd.spawn().map_err(|e| format!("FFmpeg 啟動失敗：{e}"))?;
        let stdin = child
            .take_stdin()
            .ok_or_else(|| "拿不到 FFmpeg 的輸入管線".to_string())?;
        let join = thread::spawn(move || {
            let mut errs: Vec<String> = Vec::new();
            if let Ok(iter) = child.iter() {
                for ev in iter {
                    if let FfmpegEvent::Log(LogLevel::Error | LogLevel::Fatal, m) = ev {
                        errs.push(m);
                    }
                }
            }
            let ok = child.wait().map(|s| s.success()).unwrap_or(false);
            (ok, errs)
        });
        Ok(Self {
            stdin: Some(stdin),
            join,
        })
    }

    /// 把一格寫進去。管線斷掉代表編碼那端已經死了，錯誤內容要等收工才拿得到
    fn write(&mut self, data: &[u8]) -> Result<(), ()> {
        match self.stdin.as_mut() {
            Some(s) => s.write_all(data).map_err(|_| ()),
            None => Err(()),
        }
    }

    /// 關掉輸入管線、等編碼收尾，回傳（成功與否, 錯誤訊息）
    fn finish(mut self) -> (bool, Vec<String>) {
        // 先關 stdin，ffmpeg 才知道沒有下一格了、可以把檔案封起來
        self.stdin.take();
        self.join.join().unwrap_or((false, Vec::new()))
    }
}

/// 一批影格平行去煙（勾了分區調色就順手一起調完）。
/// 回傳的順序與傳進來的一致（影片的格順序不能亂）
fn dehaze_batch(
    frames: Vec<Vec<u8>>,
    w: u32,
    h: u32,
    params: &SmokeParams,
    grade: &RegionGrade,
) -> Result<Vec<Vec<u8>>, String> {
    // 一格一條執行緒：批次大小本來就是照核心數決定的（見 crate::movie_workers），
    // 這裡不必再自己排班
    let graded = grade.is_active();
    let out: Vec<Option<Vec<u8>>> = thread::scope(|s| {
        let handles: Vec<_> = frames
            .into_iter()
            .map(|d| {
                s.spawn(move || {
                    let img = RgbImage::from_raw(w, h, d)?;
                    let mut out = dehaze::remove_smoke(&img, params);
                    // 調色套在去煙結果上，與預覽同一個順序。
                    // 三區的界線逐格自己判：畫面在動，天際線與煙火的位置
                    // 本來就一格一個樣（判準本身是大尺度的，不會一格一個跳）
                    if graded {
                        let masks = RegionMasks::new(&out);
                        edit::apply_region_grade(&mut out, grade, &masks);
                    }
                    Some(out.into_raw())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or(None))
            .collect()
    });
    out.into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "影格處理失敗".to_string())
}

/// 把整支影片去煙後輸出成新檔。
///
/// `codec` 是編碼器參數（由 [`crate::detect_h264_encoder`] 決定，與照片轉影片
/// 用的是同一組）；`grade` 是分區調色（去煙之後逐格套上去，見
/// [`crate::edit::apply_region_grade`]）；`workers` 是同時處理幾格；
/// `cancel` 被設起就中止並把寫到一半的檔案清掉；
/// `progress` 每處理完一批回報一次累計格數
#[allow(clippy::too_many_arguments)]
pub fn export(
    src: &Path,
    dst: &Path,
    info: &VideoInfo,
    params: &SmokeParams,
    grade: &RegionGrade,
    scale: Option<(u32, u32)>,
    codec: &[&str],
    workers: usize,
    cancel: &AtomicBool,
    progress: &dyn Fn(u64),
) -> Result<(), String> {
    let mut dec = FfmpegCommand::new();
    dec.input(src.to_string_lossy());
    // 縮小放在**解碼這一端**，也就是去煙**之前**：輸出要 1080p 的話就沒必要
    // 先花四倍的力氣把 4K 那一格去完煙再丟掉四分之三的像素。
    // 選小一號的尺寸因此不只是檔案變小，速度也直接跟著快
    if let Some((w, h)) = scale {
        dec.args(["-vf", &format!("scale={w}:{h}")]);
    }
    dec.rawvideo();
    let mut dec = dec.spawn().map_err(|e| format!("FFmpeg 啟動失敗：{e}"))?;
    let mut iter = dec
        .iter()
        .map_err(|e| format!("FFmpeg 輸出讀取失敗：{e}"))?;

    let batch_size = workers.max(1);
    let mut enc: Option<Encoder> = None;
    // 真正解碼出來的尺寸：第一格到手才知道（旋轉過的影片，檔頭寫的是轉之前的）
    let mut dims: Option<(u32, u32)> = None;
    let mut dec_errs: Vec<String> = Vec::new();
    let mut done: u64 = 0;
    // 中止或失敗時要把半成品刪掉，這裡記著要不要刪
    let mut failed: Option<String> = None;
    let mut aborted = false;

    'outer: loop {
        if cancel.load(Ordering::Relaxed) {
            aborted = true;
            break;
        }
        // 先收一批。收的過程中解碼那端是被我們的節奏擋著的
        // （sidecar 的事件通道不緩衝），所以不會愈積愈多
        let mut batch: Vec<Vec<u8>> = Vec::with_capacity(batch_size);
        let mut eof = false;
        while batch.len() < batch_size {
            match iter.next() {
                Some(FfmpegEvent::OutputFrame(f)) => {
                    // 編碼那端要知道尺寸才開得起來，而尺寸要等真的拿到一格
                    // 才作準（旋轉過的影片檔頭寫的是轉之前的）
                    if enc.is_none() {
                        dims = Some((f.width, f.height));
                        match Encoder::start(
                            src,
                            dst,
                            f.width,
                            f.height,
                            info.fps,
                            info.audio.as_deref(),
                            codec,
                        ) {
                            Ok(e) => enc = Some(e),
                            Err(e) => {
                                failed = Some(e);
                                break 'outer;
                            }
                        }
                    }
                    batch.push(f.data);
                }
                Some(FfmpegEvent::Log(LogLevel::Error | LogLevel::Fatal, m)) => dec_errs.push(m),
                Some(_) => {}
                None => {
                    eof = true;
                    break;
                }
            }
        }
        if batch.is_empty() {
            break;
        }
        let n = batch.len() as u64;
        let (fw, fh) = dims.unwrap_or((info.w, info.h));
        let outs = match dehaze_batch(batch, fw, fh, params, grade) {
            Ok(o) => o,
            Err(e) => {
                failed = Some(e);
                break;
            }
        };
        let Some(e) = enc.as_mut() else { break };
        for o in &outs {
            if e.write(o).is_err() {
                failed = Some("影片編碼中斷".into());
                break 'outer;
            }
        }
        done += n;
        progress(done);
        if eof {
            break;
        }
    }

    // 收尾：先讓編碼把檔案封起來，再確認解碼那端有沒有中途出事
    let enc_result = enc.map(|e| e.finish());
    if aborted || failed.is_some() {
        let _ = dec.kill();
    }
    let _ = dec.wait();

    if aborted {
        let _ = std::fs::remove_file(dst);
        return Err(String::new());
    }
    if let Some(e) = failed {
        let _ = std::fs::remove_file(dst);
        return Err(e);
    }
    match enc_result {
        Some((true, _)) => Ok(()),
        Some((false, errs)) => {
            let _ = std::fs::remove_file(dst);
            Err(if errs.is_empty() {
                "影片編碼失敗".into()
            } else {
                errs.join("\n")
            })
        }
        None => {
            let _ = std::fs::remove_file(dst);
            Err(if dec_errs.is_empty() {
                "讀不到這支影片的畫面".into()
            } else {
                dec_errs.join("\n")
            })
        }
    }
}

/// 把好幾支**已經處理好**的影片接成一支。
///
/// 用 ffmpeg 的 concat demuxer 且**不重新編碼**（`-c copy`）：這些片段都是
/// 本程式剛編出來的，編碼器、像素格式與（選了固定尺寸時的）解析度都一樣，
/// 直接接起來最快、也不會再掉一次畫質。
///
/// 規格真的對不上（例如尺寸選「與來源相同」而各段大小不同）時 ffmpeg 會失敗，
/// 這裡照實回報，呼叫端再退回「分開輸出」（見 `App::movie_export`）
pub fn concat(parts: &[PathBuf], dst: &Path) -> Result<(), String> {
    if parts.is_empty() {
        return Err("沒有可以合併的片段".into());
    }
    // 清單檔放在第一段旁邊（那是本程式自己的暫存資料夾）
    let list = parts[0].with_file_name("concat.txt");
    let mut text = String::new();
    for p in parts {
        // concat demuxer 的路徑用單引號包起來，路徑裡的單引號要跳脫；
        // 反斜線在這個格式裡也是跳脫字元，一律換成斜線最省事
        let s = p.to_string_lossy().replace('\\', "/").replace('\'', "'\\''");
        text.push_str(&format!("file '{s}'\n"));
    }
    std::fs::write(&list, text).map_err(|e| format!("寫不出合併清單：{e}"))?;

    let mut cmd = FfmpegCommand::new();
    cmd.args(["-y", "-f", "concat", "-safe", "0"])
        .input(list.to_string_lossy())
        .args(["-c", "copy"])
        .output(dst.to_string_lossy());
    let mut child = cmd.spawn().map_err(|e| format!("FFmpeg 啟動失敗：{e}"))?;
    let mut errs: Vec<String> = Vec::new();
    if let Ok(iter) = child.iter() {
        for ev in iter {
            if let FfmpegEvent::Log(LogLevel::Error | LogLevel::Fatal, m) = ev {
                errs.push(m);
            }
        }
    }
    let ok = child.wait().map(|s| s.success()).unwrap_or(false);
    let _ = std::fs::remove_file(&list);
    if ok && dst.exists() {
        Ok(())
    } else {
        let _ = std::fs::remove_file(dst);
        Err(if errs.is_empty() {
            "合併失敗".into()
        } else {
            errs.join("\n")
        })
    }
}
