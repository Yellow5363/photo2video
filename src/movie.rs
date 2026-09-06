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
use std::thread;

use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use image::RgbImage;

use crate::dehaze::{self, SmokeParams};
use crate::edit;
use crate::{ActiveGrade, MaskKey};

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
    /// 遮罩檢視算完（專業模式：紅色蓋住的地方不會被去煙／調色）；
    /// 附上它是照哪一份、哪一組形狀、哪個時間點算的，用法與 Preview 相同
    MaskView(MaskKey, RgbImage),
    /// 調色算完；附上當時作用的那幾區與時間點（一樣是拿來判斷有沒有過期）
    Graded(Vec<ActiveGrade>, f64, RgbImage),
    /// 試播的一小段：已經處理完幾格、總共幾格
    ClipProgress(usize, usize),
    /// 試播的一小段準備好了（或失敗；空字串＝使用者按了取消）。
    /// 附上它是照哪一組設定產生的，設定改了才知道要標「已變更」
    ClipReady(crate::ClipKey, Result<ClipFrames, String>),
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
fn grab(
    path: &Path,
    seek: Option<f64>,
    max_long: Option<u32>,
    crop: Option<&str>,
) -> Result<Grab, String> {
    let mut cmd = FfmpegCommand::new();
    // -ss 放在 -i 前面是「關鍵格快轉」：長片跳到中段是瞬間的事，
    // 放在後面則要從頭解碼過去
    if let Some(t) = seek {
        cmd.args(["-ss", &format!("{:.3}", t.max(0.0))]);
    }
    cmd.input(path.to_string_lossy());
    // 用正方形的框配 decrease：不管影片是橫是直、有沒有旋轉矩陣，
    // 長邊都會落在 m。自己算寬高再寫死反而會在旋轉的片子上算錯
    let scale = max_long.map(|m| format!("scale={m}:{m}:force_original_aspect_ratio=decrease"));
    if let Some(vf) = vf_chain(crop, scale) {
        cmd.args(["-vf", &vf]);
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

/// 把裁切與縮放接成一條 `-vf`：**先裁再縮**，取格、試播與輸出都走同一個順序，
/// 預覽看到的才是成品那一塊。兩個都沒有就回 None（不必掛濾鏡）
fn vf_chain(crop: Option<&str>, scale: Option<String>) -> Option<String> {
    let parts: Vec<String> = crop
        .map(str::to_string)
        .into_iter()
        .chain(scale)
        .collect();
    (!parts.is_empty()).then(|| parts.join(","))
}

/// 把 ffmpeg 吐回來的一格原始像素包成影像
fn to_image(w: u32, h: u32, data: Vec<u8>) -> Result<RgbImage, String> {
    RgbImage::from_raw(w, h, data).ok_or_else(|| "影格資料長度不對".to_string())
}

/// 開檔時量一次：寬高、影格率、片長、有沒有聲音
pub fn probe(path: &Path) -> Result<VideoInfo, String> {
    let g = grab(path, None, None, None)?;
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

/// 取某個時間點的一格當預覽底圖（先照 `crop` 裁、再縮到長邊不超過 `max_long`；
/// `crop` 是 ffmpeg 的 crop 濾鏡字串，見 `Crop::filter`）
pub fn preview_frame(
    path: &Path,
    secs: f64,
    max_long: Option<u32>,
    crop: Option<&str>,
) -> Result<RgbImage, String> {
    let g = grab(path, Some(secs), max_long, crop)?;
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

/// 調色的遮色片鋪成 `w`×`h` 的權重圖（1＝要調、0＝不動）。
/// 沒畫形狀就回 None＝整張都調，呼叫端不必為此多混一趟
pub fn grade_weights(g: &ActiveGrade, w: usize, h: usize) -> Option<Vec<f32>> {
    if g.shapes.is_empty() {
        return None;
    }
    let mut wt = dehaze::shape_weights(&g.shapes, g.feather, g.density, w, h);
    // 反選：畫到的地方不調、其餘都調
    if g.invert {
        for v in &mut wt {
            *v = 1.0 - *v;
        }
    }
    Some(wt)
}

/// 把調色套到一格上：有權重圖就只調權重蓋到的地方，沒有就整張調。
/// 預覽與輸出都走這一條，看到的才與成品一致
pub fn apply_active_grade(img: &mut RgbImage, g: &ActiveGrade, weights: Option<&[f32]>) {
    match weights {
        Some(wt) => edit::apply_grade_masked(img, &g.grade, wt),
        None => edit::apply_grade(img, &g.grade),
    }
}

/// 輸出時的一段：從 `start` 秒起（到下一段開始為止）用這一組去煙參數與調色。
/// 分段是為了遮色片（鏡頭會動），其餘設定其實每段都一樣，但整份帶著最單純
#[derive(Clone, PartialEq)]
pub struct ExportSeg {
    pub start: f64,
    pub params: SmokeParams,
    /// 這一段真的會作用的調色遮色區，照 1、2、3 的順序**依序**疊上去
    pub grades: Vec<ActiveGrade>,
}

/// 試播用的一小段：原始與處理後各一份，逐格放在記憶體裡循環播
pub struct ClipFrames {
    /// 實際每秒幾格（可能比原片低，見 [`prepare_clip`]）
    pub fps: f32,
    pub base: Vec<RgbImage>,
    pub after: Vec<RgbImage>,
}

/// 從 `at` 秒起抓 `secs` 秒、每秒 `fps` 格、長邊不超過 `max_long`、最多 `max_frames` 格。
/// 回傳（寬, 高, 每一格的原始像素）
fn grab_clip(
    path: &Path,
    at: f64,
    secs: f64,
    fps: f32,
    max_long: u32,
    max_frames: usize,
    crop: Option<&str>,
) -> Result<(u32, u32, Vec<Vec<u8>>), String> {
    let mut cmd = FfmpegCommand::new();
    // -ss 放在 -i 前面是關鍵格快轉（見 grab）
    cmd.args(["-ss", &format!("{:.3}", at.max(0.0))]);
    cmd.input(path.to_string_lossy());
    cmd.args(["-t", &format!("{:.3}", secs.max(0.1))]);
    // 先降影格率、再裁、再縮，三件事都在 ffmpeg 裡做，餵回來的就是要的那幾格
    let mut vf = format!("fps={fps:.3}");
    if let Some(c) = crop {
        vf.push(',');
        vf.push_str(c);
    }
    vf.push_str(&format!(
        ",scale={max_long}:{max_long}:force_original_aspect_ratio=decrease"
    ));
    cmd.args(["-vf", &vf]);
    cmd.args(["-frames:v", &max_frames.to_string()]).rawvideo();
    let mut child = cmd.spawn().map_err(|e| format!("FFmpeg 啟動失敗：{e}"))?;
    let mut dims: Option<(u32, u32)> = None;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut errs: Vec<String> = Vec::new();
    let iter = child
        .iter()
        .map_err(|e| format!("FFmpeg 輸出讀取失敗：{e}"))?;
    for ev in iter {
        match ev {
            FfmpegEvent::OutputFrame(f) => {
                let d = *dims.get_or_insert((f.width, f.height));
                if (f.width, f.height) == d && frames.len() < max_frames {
                    frames.push(f.data);
                }
            }
            FfmpegEvent::Log(LogLevel::Error | LogLevel::Fatal, m) => errs.push(m),
            _ => {}
        }
    }
    let _ = child.wait();
    match dims {
        Some((w, h)) if !frames.is_empty() => Ok((w, h, frames)),
        _ => Err(if errs.is_empty() {
            "這個時間點抓不到畫面（可能已經是片尾）".into()
        } else {
            errs.join("\n")
        }),
    }
}

/// 試播：抓一小段、用現在的設定逐格算好，給畫面循環播。
///
/// 不必等整支輸出就看得到「動起來」的樣子。畫面縮到 `max_long`、影格率
/// 壓到 `fps`，一段幾秒鐘的片子幾秒就算完；強度與天空範圍照 `out_long`
/// 折算（與預覽同一套，見 [`dehaze::preview_strength`]），看到的程度才與成品一致。
/// `segs` 照時間挑（與輸出同一個規則）；`progress(已處理, 總格數)`；
/// `cancel` 被設起就中止並回傳空字串的錯誤
#[allow(clippy::too_many_arguments)]
pub fn prepare_clip(
    src: &Path,
    at: f64,
    secs: f64,
    fps: f32,
    max_long: u32,
    max_frames: usize,
    segs: &[ExportSeg],
    crop: Option<&str>,
    out_long: u32,
    workers: usize,
    cancel: &AtomicBool,
    progress: &dyn Fn(usize, usize),
) -> Result<ClipFrames, String> {
    if segs.is_empty() {
        return Err("沒有可以套用的設定".into());
    }
    let (w, h, raws) = grab_clip(src, at, secs, fps, max_long, max_frames, crop)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(String::new());
    }
    let base: Vec<RgbImage> = raws
        .into_iter()
        .map(|d| RgbImage::from_raw(w, h, d))
        .collect::<Option<_>>()
        .ok_or_else(|| "影格資料長度不對".to_string())?;
    let total = base.len();
    // 每一段各準備一份：強度折算過的參數，以及這個尺寸的調色權重圖
    // （小圖一張只有一 MB 上下，全部先鋪好最省事）
    let long = w.max(h);
    let per_seg: Vec<(SmokeParams, Vec<Option<Vec<f32>>>)> = segs
        .iter()
        .map(|s| {
            let mut p = s.params.clone();
            p.strength = dehaze::preview_strength(p.strength, out_long, long);
            p.preview_of = Some(out_long);
            let wt = s
                .grades
                .iter()
                .map(|g| grade_weights(g, w as usize, h as usize))
                .collect();
            (p, wt)
        })
        .collect();
    let fps_f = fps.max(1e-3) as f64;
    let mut after: Vec<RgbImage> = Vec::with_capacity(total);
    let mut done = 0usize;
    for chunk in base.chunks(workers.max(1)) {
        if cancel.load(Ordering::Relaxed) {
            return Err(String::new());
        }
        let outs: Vec<Option<RgbImage>> = thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(j, img)| {
                    let idx = done + j;
                    let per_seg = &per_seg;
                    s.spawn(move || {
                        let k = seg_at(segs, at + idx as f64 / fps_f);
                        let (p, wt) = &per_seg[k];
                        let mut out = dehaze::remove_smoke(img, p);
                        // 三區依序疊上去，與輸出、預覽同一個順序
                        for (z, g) in segs[k].grades.iter().enumerate() {
                            apply_active_grade(&mut out, g, wt[z].as_deref());
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().ok()).collect()
        });
        for o in outs {
            after.push(o.ok_or_else(|| "影格處理失敗".to_string())?);
        }
        done += chunk.len();
        progress(done, total);
    }
    Ok(ClipFrames { fps, base, after })
}

/// 第 `t` 秒落在哪一段（最後一段 `start` ≤ t 的；第一段從 0 起，一定找得到）
fn seg_at(segs: &[ExportSeg], t: f64) -> usize {
    segs.iter()
        .rposition(|s| s.start <= t + 1e-6)
        .unwrap_or(0)
}

/// 一批影格平行去煙（有調色就順手一起調完）。
/// 回傳的順序與傳進來的一致（影片的格順序不能亂）。
///
/// `first` 是這一批第一格的序號、`fps` 拿來把序號換算成秒，逐格挑它落在哪一段
/// （一批可能跨兩段）；`weights` 是每一段、每一個調色遮色區的權重圖
/// （見 [`grade_weights`]，外層照段排、內層照區排，None＝那一區整張調），
/// 整支影片的每一格尺寸相同，所以呼叫端每段只鋪一次、每一批重複用
fn dehaze_batch(
    frames: Vec<Vec<u8>>,
    w: u32,
    h: u32,
    first: u64,
    fps: f32,
    segs: &[ExportSeg],
    weights: &[Vec<Option<Vec<f32>>>],
) -> Result<Vec<Vec<u8>>, String> {
    let fps = fps.max(1e-3) as f64;
    // 一格一條執行緒：批次大小本來就是照核心數決定的（見 crate::movie_workers），
    // 這裡不必再自己排班
    let out: Vec<Option<Vec<u8>>> = thread::scope(|s| {
        let handles: Vec<_> = frames
            .into_iter()
            .enumerate()
            .map(|(j, d)| {
                s.spawn(move || {
                    let img = RgbImage::from_raw(w, h, d)?;
                    let k = seg_at(segs, (first + j as u64) as f64 / fps);
                    let seg = &segs[k];
                    let mut out = dehaze::remove_smoke(&img, &seg.params);
                    // 調色套在去煙結果上，三區依序疊，與預覽同一個順序
                    for (z, g) in seg.grades.iter().enumerate() {
                        let wt = weights.get(k).and_then(|w| w.get(z)).and_then(|w| w.as_deref());
                        apply_active_grade(&mut out, g, wt);
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
/// `segs` 是照時間切好的段（至少一段、照 `start` 排好），每一格照它落在哪一段
/// 套那一段的去煙參數與調色（調色在去煙之後套，見 [`apply_active_grade`]）；
/// `codec` 是編碼器參數（由 [`crate::detect_h264_encoder`] 決定，與照片轉影片
/// 用的是同一組）；`workers` 是同時處理幾格；
/// `cancel` 被設起就中止並把寫到一半的檔案清掉；
/// `progress` 每處理完一批回報一次累計格數
#[allow(clippy::too_many_arguments)]
pub fn export(
    src: &Path,
    dst: &Path,
    info: &VideoInfo,
    segs: &[ExportSeg],
    crop: Option<&str>,
    scale: Option<(u32, u32)>,
    codec: &[&str],
    workers: usize,
    cancel: &AtomicBool,
    progress: &dyn Fn(u64),
) -> Result<(), String> {
    if segs.is_empty() {
        return Err("沒有可以套用的設定".into());
    }
    let mut dec = FfmpegCommand::new();
    dec.input(src.to_string_lossy());
    // 裁切與縮小都放在**解碼這一端**，也就是去煙**之前**：輸出要 1080p 的話就沒
    // 必要先花四倍的力氣把 4K 那一格去完煙再丟掉四分之三的像素，裁掉的那幾塊
    // 同理。選小一號的尺寸或裁掉一圈因此不只是檔案變小，速度也直接跟著快
    if let Some(vf) = vf_chain(crop, scale.map(|(w, h)| format!("scale={w}:{h}"))) {
        dec.args(["-vf", &vf]);
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
    // 每一段、每一個調色遮色區的權重圖：每一格尺寸相同，一段鋪一次就好；
    // 跑到哪一段才鋪哪一段，已經跑過的段丟掉（4K 一張要 33MB，
    // 三區乘上段數不能全留著）
    let mut grade_w: Vec<Vec<Option<Vec<f32>>>> = vec![Vec::new(); segs.len()];
    let mut grade_w_ready: Vec<bool> = vec![false; segs.len()];

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
        // 這一批會用到哪幾段（一批可能跨兩段）：權重圖先鋪好，之前的段丟掉
        let fps = info.fps.max(1e-3) as f64;
        let k0 = seg_at(segs, done as f64 / fps);
        let k1 = seg_at(segs, (done + n - 1) as f64 / fps);
        for k in k0..=k1 {
            if !grade_w_ready[k] {
                grade_w[k] = segs[k]
                    .grades
                    .iter()
                    .map(|g| grade_weights(g, fw as usize, fh as usize))
                    .collect();
                grade_w_ready[k] = true;
            }
        }
        for w in grade_w.iter_mut().take(k0) {
            w.clear();
        }
        let outs = match dehaze_batch(batch, fw, fh, done, info.fps, segs, &grade_w) {
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
