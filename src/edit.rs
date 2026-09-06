//! 去煙霧工具的後製：調色與文字。
//!
//! 主畫面的調色是接成 ffmpeg 濾鏡串、文字是交給 drawtext 燒進影片的；
//! 去煙霧工具整條路徑都在程式內跑（預覽與存檔都不經過 ffmpeg），
//! 這裡就是那一份 CPU 版本。滑桿沿用主畫面的 [`Adjustments`]：名稱、
//! 範圍與方向完全一致，同一個數字在兩邊要是同一種效果——每條濾鏡的公式
//! 都是拿灰階梯餵進 ffmpeg 量出來的，不是照記憶寫的（測試裡留著量到的
//! 對照值）。單條在 ±3 階內；整串疊完會累積到十階上下，因為 ffmpeg 是
//! 一條濾鏡一次 8 位元進出、中間還走一趟有限範圍的 YUV。
//!
//! 順序刻意與 `Adjustments::filter_chain` 相同（色溫→色調→曝光→
//! 對比/亮度/飽和→曲線→鮮豔度→清晰度），每一階段之間都夾回 0~1。
//!
//! 唯一沒照抄的是色調，理由見那一段的註解。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ab_glyph::{Font as _, FontVec, GlyphId, PxScale, ScaleFont as _};
use image::RgbImage;

use crate::{Adjustments, Crop, SubtitleStyle};

/// 一段疊在照片上的文字。位置是中心點在畫面上的比例（0~1），
/// 大小以 1080p 高度為基準（與主畫面的文字同一個尺規），
/// 旋轉單位為度（順時針）
#[derive(Clone, PartialEq)]
pub struct TextItem {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub size: i32,
    pub rot: f32,
}

impl Default for TextItem {
    fn default() -> Self {
        Self {
            text: String::new(),
            x: 0.5,
            y: 0.85,
            size: 48,
            rot: 0.0,
        }
    }
}

impl TextItem {
    /// 這一段有沒有真的要畫東西（只有空白的段落不必浪費一次點陣化）
    pub fn visible(&self) -> bool {
        !self.text.trim().is_empty()
    }
}

/// 去煙之後的加工：調色與文字。
///
/// 與去煙參數分開存放是刻意的——去煙要算上近一秒，調色與文字卻是即時的；
/// 分開才能在拖動調色滑桿時只重跑這一段（見 `App::spawn_smoke_finish`）
#[derive(Clone, PartialEq, Default)]
pub struct Finish {
    pub grade: Adjustments,
    pub texts: Vec<TextItem>,
    /// 疊在照片上的圖片（logo、標題圖…）。與文字一樣整批共用，
    /// 疊在調色之後、文字之下
    pub images: Vec<ImageItem>,
    /// 手動清除的筆跡。畫在哪一點是那一張照片自己的事，不像調色與文字
    /// 可以整批共用——所以來源是逐張存的（見 `SmokeTool::wipes`），
    /// 這裡只是順手一起交給算圖與存檔那一段
    pub wipes: Vec<Wipe>,
}

// ---------- 調色 ----------

/// BT.601 亮度係數（與 ffmpeg 的 eq 濾鏡在 yuv420p 上用的同一組）
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

/// BT.709 亮度係數（ffmpeg 的 vibrance 用這一組）
const LUMA709: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn luma(c: [f32; 3]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// 把照片轉正（先 90° 的整圈、再拉直的細角度）；沒轉就原樣回傳（不複製）。
///
/// 整圈用 `image` 的 rotate90/180/270——那是純搬像素，一點都不會糊；
/// 細角度才自己做雙線性取樣，畫布撐成外接框、四角補黑（與 ffmpeg 的
/// `rotate=…:c=black` 一致，兩條路徑出來的構圖才會相同）。
///
/// 旋轉排在**裁切之前**：裁切框的相對座標就是對著轉完的畫布算的
pub fn apply_rotate(img: RgbImage, crop: Crop) -> RgbImage {
    if !crop.has_rotation() {
        return img;
    }
    let c = crop.clamped();
    let img = match c.quarter {
        1 => image::imageops::rotate90(&img),
        2 => image::imageops::rotate180(&img),
        3 => image::imageops::rotate270(&img),
        _ => img,
    };
    if c.angle.abs() <= 1e-3 {
        return img;
    }
    let (sw, sh) = (img.width() as f32, img.height() as f32);
    // 外接框：與 Crop::canvas 同一條公式，預覽與成品的畫布才會一致
    let a = c.angle.to_radians();
    let (sa, ca) = (a.sin(), a.cos());
    let (ow, oh) = (
        (sw * ca.abs() + sh * sa.abs()).round().max(1.0) as u32,
        (sw * sa.abs() + sh * ca.abs()).round().max(1.0) as u32,
    );
    let mut out = RgbImage::new(ow, oh);
    let (ocx, ocy) = (ow as f32 / 2.0, oh as f32 / 2.0);
    let (scx, scy) = (sw / 2.0, sh / 2.0);
    for y in 0..oh {
        for x in 0..ow {
            // 反轉回原圖座標再取樣：正著推會在輸出上留下沒被寫到的洞
            let (dx, dy) = (x as f32 + 0.5 - ocx, y as f32 + 0.5 - ocy);
            let sx = dx * ca + dy * sa + scx - 0.5;
            let sy = -dx * sa + dy * ca + scy - 0.5;
            if let Some(px) = sample_bilinear(&img, sx, sy) {
                out.put_pixel(x, y, image::Rgb(px));
            }
        }
    }
    out
}

/// 雙線性取樣；落在影像外就回 None（呼叫端維持那一點的黑）
fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> Option<[u8; 3]> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    if x < -0.5 || y < -0.5 || x > w as f32 - 0.5 || y > h as f32 - 0.5 {
        return None;
    }
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |ix: i32, iy: i32| -> [f32; 3] {
        let ix = ix.clamp(0, w - 1) as u32;
        let iy = iy.clamp(0, h - 1) as u32;
        let p = img.get_pixel(ix, iy).0;
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };
    let (p00, p10, p01, p11) = (
        at(x0, y0),
        at(x0 + 1, y0),
        at(x0, y0 + 1),
        at(x0 + 1, y0 + 1),
    );
    let mut out = [0u8; 3];
    for i in 0..3 {
        let top = p00[i] + (p10[i] - p00[i]) * fx;
        let bot = p01[i] + (p11[i] - p01[i]) * fx;
        out[i] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
    }
    Some(out)
}

/// 依裁切框切出要留的那一塊；沒裁到東西就原樣回傳（不複製）。
/// `img` 必須是**已經轉正**的那張（見 [`apply_rotate`]）——裁切框的
/// 相對座標是對著旋轉後的畫布算的
pub fn apply_crop(img: RgbImage, crop: Crop) -> RgbImage {
    let c = crop.clamped();
    if c.x0 <= 5e-4 && c.y0 <= 5e-4 && c.x1 >= 1.0 - 5e-4 && c.y1 >= 1.0 - 5e-4 {
        return img;
    }
    let (x, y, w, h) = c.pixels(img.width(), img.height());
    image::imageops::crop_imm(&img, x, y, w, h).to_image()
}

/// 把調色就地套到影像上。清晰度的半徑依影像長邊縮放，
/// 預覽（縮圖）與存檔（原尺寸）才會是同一種局部對比
pub fn apply_grade(img: &mut RgbImage, adj: &Adjustments) {
    let adj = adj.clamped();
    // 只設了裁切時十二條滑桿仍是原位，這一整趟運算就免了（裁切另外做）
    if adj.grade_is_neutral() {
        return;
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let mut px: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    // 色溫：由色溫換算出一組 RGB 比例後直接相乘，與 ffmpeg 的
    // colortemperature 相同（量過它的輸出：就是這條比例乘上去，沒有混合、
    // 也沒有把亮度拉回原值）。
    //
    // 這條比例在 6500K 並不是正好 1（藍約 0.98），所以滑桿從 0 移到 ±1 時
    // 畫面會小小跳一下暖色。除以 6500K 那一組就能消掉，但主畫面沒有除，
    // 疊完整串之後藍色會差到十幾階——同一個數字在兩邊得是同一件事，
    // 這裡照著主畫面來
    if adj.temp != 0 {
        // 6500K 為中性；滑桿 +100 → 約 3000K（暖）、−100 → 約 10000K（冷）
        let kelvin = (6500.0 - adj.temp as f32 * 35.0).clamp(1000.0, 40000.0);
        let m = kelvin_rgb(kelvin);
        for p in &mut px {
            *p = [
                clamp01(p[0] * m[0]),
                clamp01(p[1] * m[1]),
                clamp01(p[2] * m[2]),
            ];
        }
    }

    // 色調：綠—洋紅軸。與主畫面的 colorbalance=gm 同號（gm 為正偏綠），
    // 只在中間調施力，黑與白兩端不動；紅藍各補一半讓亮度大致不變。
    //
    // 這一條是全篇唯一沒有照抄 ffmpeg 的：量過主畫面用的 colorbalance=gm，
    // 它只動得到 24~112 這一段（在 64 附近最強），比 112 亮的地方完全不理，
    // 照抄等於把那個怪癖搬過來
    if adj.tint != 0 {
        let gm = -adj.tint as f32 / 100.0 * 0.3;
        for p in &mut px {
            let mid = 1.0 - (2.0 * luma(*p) - 1.0).abs();
            let d = gm * mid * 0.5;
            *p = [
                clamp01(p[0] - d),
                clamp01(p[1] + d * 2.0),
                clamp01(p[2] - d),
            ];
        }
    }

    // 曝光度：與 ffmpeg 的 exposure 一樣，直接對編碼值乘上 2^EV
    if adj.exposure != 0 {
        let gain = (adj.exposure as f32 / 100.0 * 3.0).exp2();
        for p in &mut px {
            *p = [
                clamp01(p[0] * gain),
                clamp01(p[1] * gain),
                clamp01(p[2] * gain),
            ];
        }
    }

    // 對比、亮度、飽和度：比照 ffmpeg 的 eq——對比與亮度只動亮度、
    // 飽和度只縮放色度，所以先把像素拆成亮度與色度再各自處理。
    // 亮度多乘 255/219：eq 是在有限範圍（16~235）的 Y 上加這個位移，
    // 換回全範圍就是這個倍率，不補的話同一個數字會弱掉一成多
    if adj.contrast != 0 || adj.brightness != 0 || adj.saturation != 0 {
        let c = 1.0 + adj.contrast as f32 * 0.008;
        let b = adj.brightness as f32 * 0.004 * (255.0 / 219.0);
        let s = (1.0 + adj.saturation as f32 * 0.01).max(0.0);
        for p in &mut px {
            let y = luma(*p);
            let ny = c * (y - 0.5) + 0.5 + b;
            *p = [
                clamp01(ny + (p[0] - y) * s),
                clamp01(ny + (p[1] - y) * s),
                clamp01(ny + (p[2] - y) * s),
            ];
        }
    }

    // 陰影、白色、黑色：與主畫面同一組控制點，同樣用自然三次樣條內插
    if adj.shadows != 0 || adj.whites != 0 || adj.blacks != 0 {
        let sp = Spline::new(&curve_points(&adj));
        // 直接每個像素解樣條太浪費；1024 段查表再線性內插看不出差別
        const N: usize = 1024;
        let lut: Vec<f32> = (0..=N)
            .map(|i| clamp01(sp.eval(i as f32 / N as f32)))
            .collect();
        let curve = |v: f32| {
            let t = clamp01(v) * N as f32;
            let i = (t as usize).min(N - 1);
            lut[i] + (lut[i + 1] - lut[i]) * (t - i as f32)
        };
        for p in &mut px {
            *p = [curve(p[0]), curve(p[1]), curve(p[2])];
        }
    }

    // 去朦朧：薄霧就是一層加在畫面上的白幕——黑點被墊高、對比與彩度一起被壓平。
    // 正值把那層幕減掉再把範圍拉回滿格，負值反過來加一層上去；主畫面用
    // ffmpeg 的 colorlevels 做同一個仿射變換（量過它的輸出，逐階相同）。
    // 三個通道一起做，通道之間的差距會跟著被放大，彩度自己就回來了
    if adj.dehaze != 0 {
        let k = adj.dehaze.abs() as f32 / 100.0 * crate::DEHAZE_MAX_VEIL as f32;
        if adj.dehaze > 0 {
            let g = 1.0 / (1.0 - k);
            for p in &mut px {
                *p = [
                    clamp01((p[0] - k) * g),
                    clamp01((p[1] - k) * g),
                    clamp01((p[2] - k) * g),
                ];
            }
        } else {
            for p in &mut px {
                *p = [
                    clamp01(k + p[0] * (1.0 - k)),
                    clamp01(k + p[1] * (1.0 - k)),
                    clamp01(k + p[2] * (1.0 - k)),
                ];
            }
        }
    }

    // 鮮豔度：與飽和度的差別在於施力隨像素本身的彩度變化。ffmpeg 的
    // vibrance 是往「原本就鮮豔的地方加更多、減更少」那一邊做的（量過它的
    // 輸出：倍率 = 1 + 強度 × (1 + 彩度 × 強度的正負)，亮度用 BT.709），
    // 這裡照抄，同一個數字在兩邊才是同一種效果
    if adj.vibrance != 0 {
        let k = adj.vibrance as f32 / 100.0 * 2.0;
        for p in &mut px {
            let mx = p[0].max(p[1]).max(p[2]);
            let mn = p[0].min(p[1]).min(p[2]);
            let g = (1.0 + k * (1.0 + (mx - mn) * k.signum())).max(0.0);
            let y = p[0] * LUMA709[0] + p[1] * LUMA709[1] + p[2] * LUMA709[2];
            *p = [
                clamp01(y + (p[0] - y) * g),
                clamp01(y + (p[1] - y) * g),
                clamp01(y + (p[2] - y) * g),
            ];
        }
    }

    // 清晰度：大半徑低強度的 unsharp ≈ 局部對比（負值則柔化）。
    // 主畫面在 1080p 用 13px 的核，這裡照長邊等比換算成半徑，
    // 一張 6000px 的原圖與它 1600px 的預覽才會呈現同一種效果
    if adj.clarity != 0 {
        let amount = adj.clarity as f32 * 0.015;
        let r = ((w.max(h) as f32 / 300.0).round() as usize).clamp(1, 60);
        let y: Vec<f32> = px.iter().map(|p| luma(*p)).collect();
        let blur = box_blur(&y, w, h, r);
        for (p, (y0, yb)) in px.iter_mut().zip(y.iter().zip(blur.iter())) {
            let d = (y0 - yb) * amount;
            *p = [clamp01(p[0] + d), clamp01(p[1] + d), clamp01(p[2] + d)];
        }
    }

    for (dst, src) in img.pixels_mut().zip(px.iter()) {
        for c in 0..3 {
            dst[c] = (src[c] * 255.0 + 0.5) as u8;
        }
    }
}

/// 色溫（K）換算成 RGB 比例（Tanner Helland 的近似式，
/// 也是 ffmpeg colortemperature 用的那一條）
fn kelvin_rgb(k: f32) -> [f32; 3] {
    let t = k / 100.0;
    let (r, g) = if t <= 66.0 {
        (1.0, (0.390_081_58 * t.ln() - 0.631_841_45).clamp(0.0, 1.0))
    } else {
        let u = (t - 60.0).max(1e-6);
        (
            (1.292_936_2 * u.powf(-0.133_204_76)).clamp(0.0, 1.0),
            (1.129_890_9 * u.powf(-0.075_514_85)).clamp(0.0, 1.0),
        )
    };
    let b = if t >= 66.0 {
        1.0
    } else if t <= 19.0 {
        0.0
    } else {
        (0.543_206_8 * (t - 10.0).ln() - 1.196_254_1).clamp(0.0, 1.0)
    };
    [r, g, b]
}

/// 陰影／白色／黑色的曲線控制點（與 `Adjustments::filter_chain` 完全相同）
fn curve_points(adj: &Adjustments) -> Vec<(f32, f32)> {
    let mut pts: Vec<(f32, f32)> = Vec::new();
    if adj.blacks < 0 {
        // 壓黑：把輸入黑點往右移
        pts.push((0.0, 0.0));
        pts.push((-adj.blacks as f32 / 100.0 * 0.12, 0.0));
    } else {
        // 提黑：抬高輸出黑點
        pts.push((0.0, adj.blacks as f32 / 100.0 * 0.15));
    }
    if adj.shadows != 0 {
        pts.push((
            0.25,
            (0.25 + adj.shadows as f32 / 100.0 * 0.15).clamp(0.0, 1.0),
        ));
    }
    if adj.whites > 0 {
        // 提白：把輸入白點往左移
        pts.push((1.0 - adj.whites as f32 / 100.0 * 0.15, 1.0));
        pts.push((1.0, 1.0));
    } else {
        pts.push((1.0, 1.0 + adj.whites as f32 / 100.0 * 0.15));
    }
    pts
}

/// 自然三次樣條（ffmpeg 的 curves 濾鏡也是這一種內插）
struct Spline {
    xs: Vec<f32>,
    ys: Vec<f32>,
    /// 各控制點的二階導數
    y2: Vec<f32>,
}

impl Spline {
    fn new(pts: &[(f32, f32)]) -> Self {
        // x 相同的點會讓解算除以 0。控制點是算出來的，理論上不會重疊，
        // 但夾在邊界的極端值仍可能撞在一起，先濾掉比較保險
        let mut xs: Vec<f32> = Vec::with_capacity(pts.len());
        let mut ys: Vec<f32> = Vec::with_capacity(pts.len());
        for &(x, y) in pts {
            if xs.last().is_none_or(|l| x - l > 1e-5) {
                xs.push(x);
                ys.push(y);
            }
        }
        let n = xs.len();
        let mut y2 = vec![0.0f32; n];
        if n >= 3 {
            let mut u = vec![0.0f32; n];
            for i in 1..n - 1 {
                let sig = (xs[i] - xs[i - 1]) / (xs[i + 1] - xs[i - 1]);
                let p = sig * y2[i - 1] + 2.0;
                y2[i] = (sig - 1.0) / p;
                let d = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
                    - (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]);
                u[i] = (6.0 * d / (xs[i + 1] - xs[i - 1]) - sig * u[i - 1]) / p;
            }
            for k in (0..n - 1).rev() {
                y2[k] = y2[k] * y2[k + 1] + u[k];
            }
        }
        Self { xs, ys, y2 }
    }

    fn eval(&self, x: f32) -> f32 {
        let n = self.xs.len();
        match n {
            0 => x,
            1 => self.ys[0],
            _ => {
                // 控制點很少（最多四個），線性搜尋就夠
                let mut hi = 1;
                while hi < n - 1 && self.xs[hi] < x {
                    hi += 1;
                }
                let lo = hi - 1;
                let h = self.xs[hi] - self.xs[lo];
                let a = (self.xs[hi] - x) / h;
                let b = (x - self.xs[lo]) / h;
                a * self.ys[lo]
                    + b * self.ys[hi]
                    + ((a * a * a - a) * self.y2[lo] + (b * b * b - b) * self.y2[hi]) * h * h / 6.0
            }
        }
    }
}

/// 方框模糊（可分離；超出邊界的取邊界像素）
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let n = (2 * r + 1) as f32;
    let mut tmp = vec![0.0f32; src.len()];
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        let mut sum = row[0] * r as f32;
        for i in 0..=r {
            sum += row[i.min(w - 1)];
        }
        for x in 0..w {
            tmp[y * w + x] = sum / n;
            sum += row[(x + r + 1).min(w - 1)] - row[x.saturating_sub(r)];
        }
    }
    let mut out = vec![0.0f32; src.len()];
    for x in 0..w {
        let at = |y: usize| tmp[y.min(h - 1) * w + x];
        let mut sum = at(0) * r as f32;
        for i in 0..=r {
            sum += at(i);
        }
        for y in 0..h {
            out[y * w + x] = sum / n;
            sum += at(y + r + 1) - at(y.saturating_sub(r));
        }
    }
    out
}

// ---------- 套遮色片的調色 ----------

/// 把調色只套在 `weights` 蓋到的地方（每個像素一個 0~1，逐列排列；
/// 1＝全套、0＝一個像素都不動），影片去煙霧的「調色套用遮色片」走這一條。
///
/// 作法是「整張套一次調色，再照權重混回去」：調色的公式因此與其他模組共用
/// 同一份（[`apply_grade`]），同一個數字在哪裡都是同一種效果，差別只在混進去
/// 多少。權重圖長度對不上時退回整張調——那是呼叫端的錯，但寧可調過頭也
/// 不要默默什麼都沒做
pub fn apply_grade_masked(img: &mut RgbImage, adj: &Adjustments, weights: &[f32]) {
    let adj = adj.clamped();
    if adj.grade_is_neutral() {
        return;
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    if weights.len() != w * h {
        apply_grade(img, &adj);
        return;
    }
    let mut layer = img.clone();
    apply_grade(&mut layer, &adj);
    for (i, (d, l)) in img.pixels_mut().zip(layer.pixels()).enumerate() {
        let k = weights[i].clamp(0.0, 1.0);
        // 這一點幾乎沒份（半階都不到），混了也是原樣
        if k <= 1.0 / 512.0 {
            continue;
        }
        if k >= 1.0 - 1.0 / 512.0 {
            *d = *l;
            continue;
        }
        for c in 0..3 {
            d[c] = (d[c] as f32 + k * (l[c] as f32 - d[c] as f32))
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
}

// ---------- 手動清除 ----------

/// 手動清除的一筆筆跡：沿著 `pts` 連成的折線刷過去，刷到的地方換成四周
/// 補起來的夜空——去煙之後零星沒清乾淨的煙塊，用滑桿再拉重只會傷到煙火，
/// 直接把那一塊抹掉最省事。
///
/// 座標是相對座標（0~1）、半徑是佔影像長邊的比例，預覽縮圖與原尺寸才會
/// 清掉同一塊（與遮色片筆刷 `dehaze::Brush` 同一套規矩）
#[derive(Clone, PartialEq, Debug)]
pub struct Wipe {
    /// 筆跡經過的點
    pub pts: Vec<[f32; 2]>,
    /// 筆刷半徑，佔影像長邊的比例
    pub radius: f32,
    /// 保留煙火紋路：只把煙那一層扣掉，塗到的煙火線條原樣留著。
    /// 關著就是整塊換成補起來的夜空（線條也一起不見）
    pub keep_detail: bool,
    /// 羽化 0~1：邊緣過渡佔半徑的比例。0＝硬邊（會看得到一圈圓），
    /// 1＝從圓心就開始淡出。比照 Lightroom 筆刷的「羽化」
    pub feather: f32,
    /// 流暢度 0~1：圓章一次上多少。1＝一下就滿（同一筆重複經過不會更濃），
    /// 小於 1 則要在同一處多刷幾次才會堆到滿，適合慢慢加一點點
    pub flow: f32,
    /// 濃度 0~1：這一筆最多能清到什麼程度。1＝完全換成補起來的夜空，
    /// 0.5＝只做一半（原本的煙留一半），拿來輕輕壓淡而不是整塊清掉
    pub density: f32,
}

impl Default for Wipe {
    /// 一筆什麼都還沒設定的筆跡：全流量、全濃度，羽化 0.45（實心核心留
    /// 55%，與還沒有這幾個參數之前的固定值一致）
    fn default() -> Self {
        Self {
            pts: Vec::new(),
            radius: 0.0,
            keep_detail: true,
            feather: 0.45,
            flow: 1.0,
            density: 1.0,
        }
    }
}

impl Wipe {
    /// 點一下就放開也算一筆（單點＝一個圓）；半徑小到看不出來的不算。
    /// 濃度 0 也不算——那一筆什麼都不會發生，留著只是白算一輪
    pub fn is_usable(&self) -> bool {
        !self.pts.is_empty() && self.radius > 0.001 && self.density > 0.001
    }
}

/// 內插用的格點數上限（長邊）。補起來的是平滑的夜空，算在小格子上再
/// 放大回去就夠像，格子開多了只是慢
const WIPE_GRID: usize = 64;

/// 0~255 尺度上的亮度（挑「附近最暗的一格」用，不必先正規化）
fn luma_255(c: [f32; 3]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}

/// 「保留煙火紋路」量到的煙有多亮就不算煙了（鄰域亮度，sRGB 0~255）。
///
/// 被煙火照亮的濃煙再亮也就到 160 上下（去煙那邊在實照上量到的最大值是
/// 169，見 `dehaze::CORE_KEEP_AREA` 的註解）；煙火的芯則是一整片 190 以上。
/// 芯比取樣窗大，量出來的「煙」會是芯自己，照扣就把中心壓暗了——
/// 亮過這條線就一點都不扣
const WIPE_SMOKE_MAX: (f32, f32) = (170.0, 210.0);

/// 高出底下那層煙多少就算「這是煙火本身，一點都別扣」
/// （高出量 ÷ 煙的亮度）。與去煙那邊補回軌跡用的是同一組門檻
/// （見 `dehaze::RESTORE_REL`）：煙自己的紋理起伏相對於它的亮度很小，
/// 煙火則是又細又亮的一條。扣掉一層煙本來連煙火也該跟著暗一點，
/// 但煙火才是照片的主角，寧可讓它維持原樣
const WIPE_TRAIL_REL: (f32, f32) = (0.10, 0.30);

/// 算上面那個比值時分母補的那一截亮度（sRGB 0~255）。暗處純比值會失控
/// ——夜空只有雜訊，除下去照樣是個大數字（見 `dehaze::SKY_CONTRAST_FLOOR`）
const WIPE_CONTRAST_FLOOR: f32 = 40.0;

/// 標準的 smoothstep（0~1 之間平滑過渡；`e0 >= e1` 時退化成硬切）
fn smoothstep(e0: f32, e1: f32, v: f32) -> f32 {
    if e1 <= e0 {
        return if v >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 把手動清除就地套到影像上。一筆一筆依序做，後一筆看得到前一筆的結果
/// （兩筆疊在一起時才不會各自拿還沒清掉的煙去補）
pub fn apply_wipe(img: &mut RgbImage, wipes: &[Wipe]) {
    for w in wipes.iter().filter(|w| w.is_usable()) {
        wipe_one(img, w);
    }
}

/// 清掉一筆。做法是「把塗到的地方當成不知道原本長怎樣，由四周補回來」：
///
/// 1. 沿著筆跡蓋一連串軟邊圓章，得到覆蓋率（核心 1.0，往外羽化到 0）
/// 2. 沒被蓋到的像素才算數，降取樣成小格子當作取樣點
/// 3. 蓋到的格子用距離加權（近的說了算）從那些取樣點內插出顏色
/// 4. 放大回原尺寸，依覆蓋率混回原圖——邊緣是漸進的，接縫才看不出來
///
/// 勾了「保留煙火紋路」就不是整塊換掉，而是**只扣掉煙那一層**：
/// 煙是大範圍平順的、煙火線條是高出局部平均的那一點，把差額原樣加回去
/// （見下面的 `detail`）。塗到煙火上也不會把線條一起抹掉。
fn wipe_one(img: &mut RgbImage, wipe: &Wipe) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    if iw < 2 || ih < 2 {
        return;
    }
    let long = iw.max(ih) as f32;
    let r = (wipe.radius * long).max(1.0);
    // 筆跡外再留一圈當取樣範圍，內插才有依據可拿
    let pad = (r * 0.9).max(4.0);
    let pts: Vec<[f32; 2]> = wipe
        .pts
        .iter()
        .map(|p| [p[0] * iw as f32, p[1] * ih as f32])
        .collect();
    let (mut lx, mut ly, mut hx, mut hy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &pts {
        lx = lx.min(p[0]);
        hx = hx.max(p[0]);
        ly = ly.min(p[1]);
        hy = hy.max(p[1]);
    }
    let x0 = ((lx - r - pad).floor() as i32).clamp(0, iw - 1);
    let y0 = ((ly - r - pad).floor() as i32).clamp(0, ih - 1);
    let x1 = ((hx + r + pad).ceil() as i32).clamp(0, iw - 1);
    let y1 = ((hy + r + pad).ceil() as i32).clamp(0, ih - 1);
    let (bw, bh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);

    // 1) 筆跡覆蓋率。圓章間距取半徑的三成，蓋得夠密才不會刷出一節一節的邊
    let mut cov = vec![0f32; bw * bh];
    // 羽化決定實心核心留多大：0＝整個圓都實心（硬邊）、1＝從圓心就開始淡出
    let inner = r * (1.0 - wipe.feather.clamp(0.0, 1.0));
    let flow = wipe.flow.clamp(0.0, 1.0);
    let step = (r * 0.3).max(0.75);
    let mut prev: Option<[f32; 2]> = None;
    for p in &pts {
        let now = [p[0] - x0 as f32, p[1] - y0 as f32];
        match prev {
            None => stamp_disc(&mut cov, bw, bh, now, r, inner, flow),
            Some(q) => {
                let (dx, dy) = (now[0] - q[0], now[1] - q[1]);
                let n = ((dx * dx + dy * dy).sqrt() / step).ceil().max(1.0) as usize;
                for i in 1..=n {
                    let t = i as f32 / n as f32;
                    stamp_disc(&mut cov, bw, bh, [q[0] + dx * t, q[1] + dy * t], r, inner, flow);
                }
            }
        }
        prev = Some(now);
    }

    // 2) 降取樣：一格記下「沒被蓋到的像素」的平均色與張數
    let cell = bw.max(bh).div_ceil(WIPE_GRID).max(1);
    let (gw, gh) = (bw.div_ceil(cell), bh.div_ceil(cell));
    let mut sum = vec![[0f32; 3]; gw * gh];
    let mut known = vec![0f32; gw * gh];
    let mut total = vec![0f32; gw * gh];
    for y in 0..bh {
        for x in 0..bw {
            let gi = (y / cell) * gw + x / cell;
            total[gi] += 1.0;
            if cov[y * bw + x] <= 0.002 {
                let px = img.get_pixel((x0 + x as i32) as u32, (y0 + y as i32) as u32).0;
                for c in 0..3 {
                    sum[gi][c] += px[c] as f32;
                }
                known[gi] += 1.0;
            }
        }
    }
    // 一格裡大半沒被蓋到，才算「知道這裡本來是什麼顏色」；
    // 邊緣那些半蓋到的格子拿來當取樣點會把煙的顏色又補回去
    let mut val = vec![[0f32; 3]; gw * gh];
    let mut solid = vec![false; gw * gh];
    for i in 0..gw * gh {
        if known[i] > total[i] * 0.5 {
            solid[i] = true;
            for c in 0..3 {
                val[i][c] = sum[i][c] / known[i];
            }
        }
    }
    let anchors: Vec<usize> = (0..gw * gh).filter(|&i| solid[i]).collect();
    if anchors.is_empty() {
        // 整塊都被蓋住，沒有東西可以拿來補
        return;
    }

    // 3) 取樣點先往暗的一邊靠：要清的本來就是煙，洞口一圈多半也還是煙，
    // 照原樣內插只會把煙又補回去（一團糊掉的亮斑比原本更難看）。
    // 每個取樣點改用附近幾格裡最暗的那一格——那才是這附近乾淨夜空的樣子
    let dark = {
        const R: i32 = 3;
        let mut out = val.clone();
        for gy in 0..gh as i32 {
            for gx in 0..gw as i32 {
                let i = (gy * gw as i32 + gx) as usize;
                if !solid[i] {
                    continue;
                }
                let mut best = val[i];
                let mut best_y = luma_255(best);
                for ny in (gy - R).max(0)..=(gy + R).min(gh as i32 - 1) {
                    for nx in (gx - R).max(0)..=(gx + R).min(gw as i32 - 1) {
                        let j = (ny * gw as i32 + nx) as usize;
                        if !solid[j] {
                            continue;
                        }
                        let y = luma_255(val[j]);
                        if y < best_y {
                            best_y = y;
                            best = val[j];
                        }
                    }
                }
                out[i] = best;
            }
        }
        out
    };

    // 4) 距離加權內插。權重取 1/d⁴：貼著洞口的取樣點壓倒性地重，
    // 遠處的煙火亮線幾乎不會被拉進來，洞口顏色又能一路平順地接上
    for gy in 0..gh {
        for gx in 0..gw {
            let i = gy * gw + gx;
            if solid[i] {
                continue;
            }
            let (mut acc, mut wsum) = ([0f32; 3], 0f32);
            for &j in &anchors {
                let dx = (j % gw) as f32 - gx as f32;
                let dy = (j / gw) as f32 - gy as f32;
                let d2 = dx * dx + dy * dy;
                let w = 1.0 / (d2 * d2 + 1e-3);
                wsum += w;
                for c in 0..3 {
                    acc[c] += dark[j][c] * w;
                }
            }
            for c in 0..3 {
                val[i][c] = acc[c] / wsum;
            }
        }
    }

    // 5) 要保留煙火紋路的話，先量出「這一點上的煙有多厚」。
    //
    // 煙是大範圍平順的一層，煙火線條又細又亮。直接取局部平均會把線條自己的
    // 光算進去（線條越亮、量到的煙越厚），扣完線條就跟著暗掉；改成只平均
    // 「不比局部平均亮」的那些像素——線條被排除在外，量到的就是它底下的煙。
    //
    // 半徑照影像長邊等比縮放（與清晰度同一個尺規），預覽與原尺寸才是
    // 同一種取捨；縮圖上的線條也是等比例變細的
    let veil = wipe.keep_detail.then(|| {
        let r_blur = ((iw.max(ih) as f32 / 300.0).round() as usize).clamp(1, 60);
        let n = bw * bh;
        let px = |x: usize, y: usize| img.get_pixel((x0 + x as i32) as u32, (y0 + y as i32) as u32).0;
        let mut y_plane = vec![0f32; n];
        for y in 0..bh {
            for x in 0..bw {
                let p = px(x, y);
                y_plane[y * bw + x] =
                    p[0] as f32 * LUMA[0] + p[1] as f32 * LUMA[1] + p[2] as f32 * LUMA[2];
            }
        }
        // 亮度不高於局部平均的像素才算「這裡的煙」（+1 是給雜訊留的餘裕）
        let mean = box_blur(&y_plane, bw, bh, r_blur);
        let mask: Vec<f32> = (0..n)
            .map(|i| if y_plane[i] <= mean[i] + 1.0 { 1.0 } else { 0.0 })
            .collect();
        let den = box_blur(&mask, bw, bh, r_blur);
        let mut out = vec![[0f32; 3]; n];
        let mut plane = vec![0f32; n];
        for c in 0..3 {
            for y in 0..bh {
                for x in 0..bw {
                    plane[y * bw + x] = px(x, y)[c] as f32 * mask[y * bw + x];
                }
            }
            let num = box_blur(&plane, bw, bh, r_blur);
            for i in 0..n {
                out[i][c] = num[i] / den[i].max(0.02);
            }
        }
        // 上面那一關擋得住又細又亮的線條（它們高於局部平均），但煙火最亮的
        // 那一團比取樣窗還大，整窗都是它自己——量到的「煙」就是那團煙火，
        // 照扣下去正好把中心壓暗（實照上看到的正是這個症狀）。
        //
        // 所以再量第二種：把 bbox 切成小格，每一格取**最暗的那個像素**。
        // 煙火是一條一條的，縫隙裡透出來的才是它底下的煙；煙面平順的地方
        // 最暗值也就在平均值下面一點點。格子邊長取線條的粗細量級
        // （長邊的 0.25%，與去煙那邊量軌跡用的同一個尺規）
        let mcell = ((iw.max(ih) as f32 * 0.0025).round() as usize).clamp(2, 64);
        let (mw, mh) = (bw.div_ceil(mcell), bh.div_ceil(mcell));
        let mut cmin = vec![[0f32; 3]; mw * mh];
        let mut cluma = vec![f32::MAX; mw * mh];
        for y in 0..bh {
            for x in 0..bw {
                let i = (y / mcell) * mw + x / mcell;
                let l = y_plane[y * bw + x];
                if l < cluma[i] {
                    cluma[i] = l;
                    let p = px(x, y);
                    cmin[i] = [p[0] as f32, p[1] as f32, p[2] as f32];
                }
            }
        }
        // 一格一個最暗值難免跳動，先在小格圖上做一次 3×3 平均壓平
        let mut sm = cmin.clone();
        for gy in 0..mh {
            for gx in 0..mw {
                let (mut acc, mut cnt) = ([0f32; 3], 0f32);
                for ny in gy.saturating_sub(1)..=(gy + 1).min(mh - 1) {
                    for nx in gx.saturating_sub(1)..=(gx + 1).min(mw - 1) {
                        for c in 0..3 {
                            acc[c] += cmin[ny * mw + nx][c];
                        }
                        cnt += 1.0;
                    }
                }
                for c in 0..3 {
                    sm[gy * mw + gx][c] = acc[c] / cnt;
                }
            }
        }
        // 兩種估法取比較保守（暗）的那一個：平順的煙面用修剪平均最準，
        // 煙火團裡則是「縫隙」說了算，一點都不會多扣
        for y in 0..bh {
            for x in 0..bw {
                let fx = (x as f32 + 0.5) / mcell as f32 - 0.5;
                let fy = (y as f32 + 0.5) / mcell as f32 - 0.5;
                let (gx0, gy0) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - gx0, fy - gy0);
                let at = |gx: f32, gy: f32| -> [f32; 3] {
                    let gx = (gx.max(0.0) as usize).min(mw - 1);
                    let gy = (gy.max(0.0) as usize).min(mh - 1);
                    sm[gy * mw + gx]
                };
                let (c00, c10) = (at(gx0, gy0), at(gx0 + 1.0, gy0));
                let (c01, c11) = (at(gx0, gy0 + 1.0), at(gx0 + 1.0, gy0 + 1.0));
                let mut low = [0f32; 3];
                for c in 0..3 {
                    let top = c00[c] + (c10[c] - c00[c]) * tx;
                    let bot = c01[c] + (c11[c] - c01[c]) * tx;
                    low[c] = top + (bot - top) * ty;
                }
                let v = &mut out[y * bw + x];
                if luma_255(low) < luma_255(*v) {
                    *v = low;
                }
                // 最後一道：量到的煙亮過煙本身的上限就不是煙，是亮芯，
                // 一點都不扣（見 [`WIPE_SMOKE_MAX`]）
                let over = smoothstep(WIPE_SMOKE_MAX.0, WIPE_SMOKE_MAX.1, luma_255(*v));
                if over > 0.0 {
                    for c in 0..3 {
                        v[c] *= 1.0 - over;
                    }
                }
            }
        }
        out
    });

    // 6) 放大回原尺寸並依覆蓋率混回去（格心對到格子中央，雙線性取樣）。
    // 補起來的顏色只用來「壓下去」，不會比原本更亮——清除是把煙拿掉，
    // 不該在夜空上多畫出一塊比原本還亮的斑
    // 濃度封頂：這一筆最多能清到什麼程度。1 是整塊換成補起來的夜空，
    // 小一點就只壓淡一部分（原本的煙按比例留著）
    let density = wipe.density.clamp(0.0, 1.0);
    for y in 0..bh {
        for x in 0..bw {
            let a = cov[y * bw + x] * density;
            if a <= 0.002 {
                continue;
            }
            let fx = (x as f32 + 0.5) / cell as f32 - 0.5;
            let fy = (y as f32 + 0.5) / cell as f32 - 0.5;
            let (gx0, gy0) = (fx.floor(), fy.floor());
            let (tx, ty) = (fx - gx0, fy - gy0);
            let at = |gx: f32, gy: f32| -> [f32; 3] {
                let gx = (gx.max(0.0) as usize).min(gw - 1);
                let gy = (gy.max(0.0) as usize).min(gh - 1);
                val[gy * gw + gx]
            };
            let (c00, c10) = (at(gx0, gy0), at(gx0 + 1.0, gy0));
            let (c01, c11) = (at(gx0, gy0 + 1.0), at(gx0 + 1.0, gy0 + 1.0));
            let keep = veil.as_ref().map(|v| v[y * bw + x]);
            let px = img.get_pixel_mut((x0 + x as i32) as u32, (y0 + y as i32) as u32);
            // 這一點比底下那層煙高出多少：高得多就是煙火本身，那一份煙不扣，
            // 亮度維持原樣（見 [`WIPE_TRAIL_REL`]）
            let hold = keep.map_or(0.0, |v| {
                let vy = luma_255(v);
                let oy = luma_255([px[0] as f32, px[1] as f32, px[2] as f32]);
                let rel = (oy - vy).max(0.0) / (vy + WIPE_CONTRAST_FLOOR);
                smoothstep(WIPE_TRAIL_REL.0, WIPE_TRAIL_REL.1, rel)
            });
            for c in 0..3 {
                let top = c00[c] + (c10[c] - c00[c]) * tx;
                let bot = c01[c] + (c11[c] - c01[c]) * tx;
                let o = px[c] as f32;
                let bg = top + (bot - top) * ty;
                // 保留紋路：只把「煙比乾淨夜空多出來的那一層」扣掉，
                // 高出煙的部分（煙火線條）原封不動地留在新的底色上。
                // 純粹是煙的地方 o ≈ 煙的厚度，扣完就正好落在補起來的夜空上
                let target = match keep {
                    Some(v) => o - (v[c] - bg).max(0.0) * (1.0 - hold),
                    None => bg,
                };
                let target = target.min(o);
                px[c] = (o + (target - o) * a).clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

/// 在覆蓋率圖上蓋一個軟邊圓章。`at` 是圓心在這塊 bbox 裡的座標。
///
/// `flow` 是 1 時取較大值疊上去，同一處重複刷不會越刷越濃（原本的手感）；
/// 小於 1 則每章只上一部分、往滿的方向堆，同一處多刷幾次才會到底
fn stamp_disc(cov: &mut [f32], bw: usize, bh: usize, at: [f32; 2], r: f32, inner: f32, flow: f32) {
    let lo_x = (at[0] - r).floor().max(0.0) as usize;
    let lo_y = (at[1] - r).floor().max(0.0) as usize;
    let hi_x = ((at[0] + r).ceil().max(0.0) as usize).min(bw.saturating_sub(1));
    let hi_y = ((at[1] + r).ceil().max(0.0) as usize).min(bh.saturating_sub(1));
    if lo_x > hi_x || lo_y > hi_y {
        return;
    }
    for y in lo_y..=hi_y {
        let dy = y as f32 + 0.5 - at[1];
        for x in lo_x..=hi_x {
            let dx = x as f32 + 0.5 - at[0];
            let d = (dx * dx + dy * dy).sqrt();
            let a = if d <= inner {
                1.0
            } else if d >= r {
                continue;
            } else {
                // smoothstep：邊緣硬切會在畫面上留下一圈看得出來的圓
                let t = (r - d) / (r - inner).max(1e-3);
                t * t * (3.0 - 2.0 * t)
            };
            let slot = &mut cov[y * bw + x];
            if flow >= 1.0 {
                if a > *slot {
                    *slot = a;
                }
            } else {
                *slot += a * flow * (1.0 - *slot);
            }
        }
    }
}

// ---------- 疊圖片 ----------

/// 疊在照片上的一張圖片（logo、標題圖、浮水印…）。
///
/// 位置是中心點在畫面上的比例（0~1）、大小是**寬度佔照片寬度的比例**、
/// 旋轉單位為度（順時針）——全是相對值，所以預覽上擺好的樣子，
/// 存檔用原尺寸畫出來也是同一個位置與大小
#[derive(Clone, PartialEq)]
pub struct ImageItem {
    /// 來源檔（剪貼簿貼上的會先落地成暫存 PNG）
    pub path: PathBuf,
    pub x: f32,
    pub y: f32,
    /// 寬度佔照片寬度的比例
    pub scale: f32,
    pub rot: f32,
    /// 不透明度 0~1
    pub opacity: f32,
}

impl Default for ImageItem {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            x: 0.5,
            y: 0.5,
            scale: 0.3,
            rot: 0.0,
            opacity: 1.0,
        }
    }
}

impl ImageItem {
    /// 小到看不見、或還沒指定檔案的就不必畫
    pub fn visible(&self) -> bool {
        !self.path.as_os_str().is_empty() && self.scale > 0.001 && self.opacity > 0.002
    }
}

/// 疊圖片的解碼快取。預覽每動一次調色滑桿就要重畫一次，
/// 每次都重讀一張 PNG 太浪費；同一個檔案只解一次
fn overlay_cache() -> &'static Mutex<HashMap<PathBuf, Option<Arc<image::RgbaImage>>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Option<Arc<image::RgbaImage>>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 讀一張要疊上去的圖片（含透明度）；讀不到就記著別再重試
pub fn load_overlay(path: &Path) -> Option<Arc<image::RgbaImage>> {
    let mut cache = overlay_cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = cache.get(path) {
        return hit.clone();
    }
    let got = image::open(path).ok().map(|i| Arc::new(i.to_rgba8()));
    cache.insert(path.to_path_buf(), got.clone());
    got
}

/// 把疊圖片就地畫到照片上（調色之後、文字之前）
pub fn draw_images(img: &mut RgbImage, items: &[ImageItem]) {
    for it in items.iter().filter(|i| i.visible()) {
        if let Some(src) = load_overlay(&it.path) {
            draw_image(img, it, &src);
        }
    }
}

/// 畫一張：目的地是一個旋轉過的矩形，逐點反算回原圖座標做雙線性取樣。
/// 正著畫（掃描原圖、算它落在哪）會在放大時留下空隙
fn draw_image(img: &mut RgbImage, it: &ImageItem, src: &image::RgbaImage) {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let (sw, sh) = (src.width() as f32, src.height() as f32);
    if sw < 1.0 || sh < 1.0 {
        return;
    }
    // 目的地的寬高（像素）與中心
    let dw = (it.scale * iw).max(1.0);
    let dh = dw * sh / sw;
    let (cx, cy) = (it.x * iw, it.y * ih);
    let (sin, cos) = it.rot.to_radians().sin_cos();
    // 旋轉後的外接矩形：四個角轉過去取極值
    let (hx, hy) = (dw / 2.0, dh / 2.0);
    let corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)];
    let (mut lx, mut ly, mut rx, mut ry) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in corners {
        let (px, py) = (cx + x * cos - y * sin, cy + x * sin + y * cos);
        lx = lx.min(px);
        rx = rx.max(px);
        ly = ly.min(py);
        ry = ry.max(py);
    }
    let x0 = (lx.floor().max(0.0) as u32).min(img.width().saturating_sub(1));
    let y0 = (ly.floor().max(0.0) as u32).min(img.height().saturating_sub(1));
    let x1 = (rx.ceil().max(0.0) as u32).min(img.width().saturating_sub(1));
    let y1 = (ry.ceil().max(0.0) as u32).min(img.height().saturating_sub(1));
    let alpha = it.opacity.clamp(0.0, 1.0);
    for y in y0..=y1 {
        for x in x0..=x1 {
            // 畫面座標 → 以圖片中心為原點、轉回沒旋轉的方向
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            let (ux, uy) = (dx * cos + dy * sin, -dx * sin + dy * cos);
            if ux < -hx || ux > hx || uy < -hy || uy > hy {
                continue;
            }
            // → 原圖的像素座標
            let fx = (ux + hx) / dw * sw - 0.5;
            let fy = (uy + hy) / dh * sh - 0.5;
            let Some(s) = sample_rgba(src, fx, fy) else {
                continue;
            };
            let a = s[3] * alpha;
            if a <= 0.002 {
                continue;
            }
            let px = img.get_pixel_mut(x, y);
            for c in 0..3 {
                px[c] = (px[c] as f32 + (s[c] - px[c] as f32) * a)
                    .clamp(0.0, 255.0)
                    .round() as u8;
            }
        }
    }
}

/// 雙線性取樣（回傳 RGB 0~255 與 0~1 的 alpha）；超出範圍就夾回邊界
fn sample_rgba(src: &image::RgbaImage, fx: f32, fy: f32) -> Option<[f32; 4]> {
    let (w, h) = (src.width() as i32, src.height() as i32);
    if w < 1 || h < 1 {
        return None;
    }
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: i32, y: i32| {
        let p = src.get_pixel(
            x.clamp(0, w - 1) as u32,
            y.clamp(0, h - 1) as u32,
        );
        [
            p[0] as f32,
            p[1] as f32,
            p[2] as f32,
            p[3] as f32 / 255.0,
        ]
    };
    let (c00, c10, c01, c11) = (
        at(x0, y0),
        at(x0 + 1, y0),
        at(x0, y0 + 1),
        at(x0 + 1, y0 + 1),
    );
    let mut out = [0f32; 4];
    for c in 0..4 {
        let top = c00[c] + (c10[c] - c00[c]) * tx;
        let bot = c01[c] + (c11[c] - c01[c]) * tx;
        out[c] = top + (bot - top) * ty;
    }
    Some(out)
}

// ---------- 文字 ----------

/// 讀一個字型檔給文字點陣化用。.ttc 這種字型集合固定取第一套
/// （與 egui 載入中文字型的做法一致）
pub fn load_font(path: &Path) -> Option<FontVec> {
    let bytes = std::fs::read(path).ok()?;
    FontVec::try_from_vec_and_index(bytes, 0).ok()
}

/// 「em 的高度」換成 ab_glyph 的 PxScale 要乘的比例。
/// PxScale 是以 ascent−descent 為單位，字級卻是以 em 為單位
fn em_ratio(font: &FontVec) -> f32 {
    match font.units_per_em() {
        Some(u) if u > 0.0 => font.height_unscaled() / u,
        _ => 1.0,
    }
}

/// 一行排好的字：字符與它相對於行首的位移
struct Line {
    glyphs: Vec<(GlyphId, f32)>,
    width: f32,
}

/// 把文字燒進影像（存檔用；預覽是由 egui 直接畫在畫面上，見 `App::ui_smoke_text`）
pub fn draw_texts(img: &mut RgbImage, texts: &[TextItem], style: &SubtitleStyle, font: &FontVec) {
    for t in texts.iter().filter(|t| t.visible()) {
        draw_text(img, t, style, font);
    }
}

fn draw_text(img: &mut RgbImage, item: &TextItem, style: &SubtitleStyle, font: &FontVec) {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    // 字級與外框都以 1080p 高度為基準縮放，與主畫面的 drawtext 一致
    let scale = ih / 1080.0;
    let px = (item.size as f32 * scale).max(2.0);
    // 字級講的是 em 的高度，ab_glyph 的 PxScale 卻是以「ascent−descent」為單位，
    // 兩者差了這個比例。少換算這一下，存出來的字會比預覽大一截
    //（egui 內部也是這樣換算的，見 epaint::text::fonts）
    let px_scale = PxScale::from(px * em_ratio(font));
    let sf = font.as_scaled(px_scale);
    let line_h = sf.ascent() - sf.descent() + sf.line_gap();

    // 排版：每一行各自算寬度，行內靠左、整塊置中（與 egui 的 galley 相同）
    let lines: Vec<Line> = item
        .text
        .trim_end()
        .split('\n')
        .map(|s| {
            let mut glyphs = Vec::new();
            let mut x = 0.0f32;
            let mut prev: Option<GlyphId> = None;
            for ch in s.chars() {
                let id = font.glyph_id(ch);
                if let Some(p) = prev {
                    x += sf.kern(p, id);
                }
                glyphs.push((id, x));
                x += sf.h_advance(id);
                prev = Some(id);
            }
            Line { glyphs, width: x }
        })
        .collect();
    let bw = lines.iter().fold(0.0f32, |m, l| m.max(l.width));
    let bh = line_h * lines.len() as f32;
    if bw <= 0.0 || bh <= 0.0 {
        return;
    }

    let ow = (style.outline_w as f32 * scale).max(0.0);
    let box_pad = if style.boxed {
        (px * 0.25).max(4.0 * scale)
    } else {
        0.0
    };
    let center = [item.x * iw, item.y * ih];
    let angle = item.rot.to_radians();
    let (sin, cos) = angle.sin_cos();
    // 版面座標（原點在文字區塊左上角）↔ 影像座標
    let block_c = [bw * 0.5, bh * 0.5];
    let to_img = |p: [f32; 2]| {
        let (dx, dy) = (p[0] - block_c[0], p[1] - block_c[1]);
        [
            center[0] + dx * cos - dy * sin,
            center[1] + dx * sin + dy * cos,
        ]
    };
    let to_block = |p: [f32; 2]| {
        let (dx, dy) = (p[0] - center[0], p[1] - center[1]);
        [
            block_c[0] + dx * cos + dy * sin,
            block_c[1] - dx * sin + dy * cos,
        ]
    };

    // 要動到的影像範圍：文字區塊（含底框與外框）四個角轉到影像座標後的外接矩形。
    // 只掃這一塊，字級開很大、或整段被拖出畫面時都不會白掃整張照片
    let ext = box_pad.max(ow) + 1.0;
    let corners = [
        [-ext, -ext],
        [bw + ext, -ext],
        [bw + ext, bh + ext],
        [-ext, bh + ext],
    ];
    let Some((dx0, dy0, dx1, dy1)) = bbox(corners.iter().map(|&c| to_img(c)), iw, ih) else {
        return;
    };

    // 再把要掃的那一塊轉回版面座標，只點陣化這個範圍內的字。
    // 沒有這一步，一段被放到很大的文字會配置出遠大於照片的遮罩
    let (mut mx0, mut my0, mut mx1, mut my1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for c in [
        [dx0 as f32, dy0 as f32],
        [dx1 as f32, dy0 as f32],
        [dx1 as f32, dy1 as f32],
        [dx0 as f32, dy1 as f32],
    ] {
        let p = to_block(c);
        mx0 = mx0.min(p[0]);
        my0 = my0.min(p[1]);
        mx1 = mx1.max(p[0]);
        my1 = my1.max(p[1]);
    }
    let pad = ow.ceil() + 2.0;
    let mx0 = (mx0 - 1.0).max(-pad).floor() as i32;
    let my0 = (my0 - 1.0).max(-pad).floor() as i32;
    let mx1 = (mx1 + 1.0).min(bw + pad).ceil() as i32;
    let my1 = (my1 + 1.0).min(bh + pad).ceil() as i32;
    let (mw, mh) = ((mx1 - mx0).max(0) as usize, (my1 - my0).max(0) as usize);
    let has_glyphs = mw > 0 && mh > 0;

    // 字的覆蓋率遮罩（只存一個通道；外框直接在合成時取八個方向的最大值，
    // 不必為了描邊再配置一份）
    let mut mask = vec![0u8; if has_glyphs { mw * mh } else { 0 }];
    if has_glyphs {
        for (i, line) in lines.iter().enumerate() {
            let base = line_h * i as f32 + sf.ascent();
            for &(id, gx) in &line.glyphs {
                let g = id.with_scale_and_position(px_scale, ab_glyph::point(gx, base));
                let Some(og) = font.outline_glyph(g) else {
                    continue;
                };
                let b = og.px_bounds();
                if b.max.x < mx0 as f32
                    || b.min.x > mx1 as f32
                    || b.max.y < my0 as f32
                    || b.min.y > my1 as f32
                {
                    continue;
                }
                og.draw(|gx, gy, c| {
                    let x = b.min.x as i32 + gx as i32 - mx0;
                    let y = b.min.y as i32 + gy as i32 - my0;
                    if x < 0 || y < 0 || x >= mw as i32 || y >= mh as i32 {
                        return;
                    }
                    let at = &mut mask[y as usize * mw + x as usize];
                    // 字符可能相疊（例如注音、合字），取較大的覆蓋率
                    *at = (*at).max((c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                });
            }
        }
    }

    // 遮罩的雙線性取樣（座標為版面座標）
    let sample = |x: f32, y: f32| -> f32 {
        if !has_glyphs {
            return 0.0;
        }
        let (fx, fy) = (x - mx0 as f32 - 0.5, y - my0 as f32 - 0.5);
        let (bx, by) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - bx, fy - by);
        let (bx, by) = (bx as i32, by as i32);
        let get = |x: i32, y: i32| -> f32 {
            if x < 0 || y < 0 || x >= mw as i32 || y >= mh as i32 {
                0.0
            } else {
                mask[y as usize * mw + x as usize] as f32 / 255.0
            }
        };
        let a = get(bx, by) + (get(bx + 1, by) - get(bx, by)) * tx;
        let b = get(bx, by + 1) + (get(bx + 1, by + 1) - get(bx, by + 1)) * tx;
        a + (b - a) * ty
    };

    let fill = style.color.to_srgba_unmultiplied();
    let edge = style.outline_color.to_srgba_unmultiplied();
    // 外框的八個方向與預覽的畫法相同（近似 ffmpeg drawtext 的 borderw）
    const DIRS: [(f32, f32); 8] = [
        (-1.0, 0.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-0.7, -0.7),
        (0.7, -0.7),
        (-0.7, 0.7),
        (0.7, 0.7),
    ];
    let outlined = ow > 0.05 && edge[3] > 0;
    // 離文字區塊超過一個外框寬就一定是全透明，整片跳過不必取樣
    let slack = ow + 1.5;

    for y in dy0..dy1 {
        for x in dx0..dx1 {
            let b = to_block([x as f32 + 0.5, y as f32 + 0.5]);
            let src = img.get_pixel(x as u32, y as u32);
            let mut dst = [src[0] as f32, src[1] as f32, src[2] as f32];
            let mut touched = false;

            // 半透明底框（近似輸出的 box=1）；邊緣做一個像素的漸變才不會有鋸齒
            if box_pad > 0.0 {
                let inside = (b[0] + box_pad)
                    .min(bw + box_pad - b[0])
                    .min(b[1] + box_pad)
                    .min(bh + box_pad - b[1]);
                let cov = (inside + 0.5).clamp(0.0, 1.0);
                if cov > 0.0 {
                    blend(&mut dst, [0.0, 0.0, 0.0], cov * 0.4);
                    touched = true;
                }
            }

            if b[0] > -slack && b[1] > -slack && b[0] < bw + slack && b[1] < bh + slack {
                if outlined {
                    let mut a = 0.0f32;
                    for (ox, oy) in DIRS {
                        a = a.max(sample(b[0] - ox * ow, b[1] - oy * ow));
                        if a >= 1.0 {
                            break;
                        }
                    }
                    if a > 0.0 {
                        blend(
                            &mut dst,
                            [edge[0] as f32, edge[1] as f32, edge[2] as f32],
                            a * edge[3] as f32 / 255.0,
                        );
                        touched = true;
                    }
                }
                let a = sample(b[0], b[1]);
                if a > 0.0 && fill[3] > 0 {
                    blend(
                        &mut dst,
                        [fill[0] as f32, fill[1] as f32, fill[2] as f32],
                        a * fill[3] as f32 / 255.0,
                    );
                    touched = true;
                }
            }

            if touched {
                let p = img.get_pixel_mut(x as u32, y as u32);
                for c in 0..3 {
                    p[c] = dst[c].clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

/// 一組點在影像範圍內的整數外接矩形；完全在畫面外時回傳 None
fn bbox(pts: impl Iterator<Item = [f32; 2]>, iw: f32, ih: f32) -> Option<(i32, i32, i32, i32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in pts {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let x0 = x0.floor().max(0.0) as i32;
    let y0 = y0.floor().max(0.0) as i32;
    let x1 = (x1.ceil() + 1.0).clamp(0.0, iw) as i32;
    let y1 = (y1.ceil() + 1.0).clamp(0.0, ih) as i32;
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// 把顏色以 a（0~1）疊到 dst 上
fn blend(dst: &mut [f32; 3], color: [f32; 3], a: f32) {
    let a = a.clamp(0.0, 1.0);
    for c in 0..3 {
        dst[c] += (color[c] - dst[c]) * a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, image::Rgb(c))
    }

    #[test]
    fn neutral_grade_leaves_every_pixel_untouched() {
        // 滑桿全歸零卻動到畫面，等於使用者「什麼都沒調」照片就變樣了
        let mut img = solid(8, 8, [40, 90, 170]);
        let before = img.clone();
        apply_grade(&mut img, &Adjustments::default());
        assert_eq!(img.as_raw(), before.as_raw());
    }

    #[test]
    fn sliders_move_the_picture_in_the_labelled_direction() {
        // 每條滑桿的方向都與主畫面的濾鏡串同義；反了的話使用者往右拉會變暗
        let base = solid(4, 4, [120, 120, 120]);
        let px = |adj: Adjustments| {
            let mut img = base.clone();
            apply_grade(&mut img, &adj);
            img.get_pixel(0, 0).0
        };
        let up = |v: i32| {
            let mut a = Adjustments::default();
            a.exposure = v;
            px(a)
        };
        assert!(up(50)[0] > 120, "曝光度 + 應該變亮");
        assert!(up(-50)[0] < 120, "曝光度 − 應該變暗");

        let mut warm = Adjustments::default();
        warm.temp = 80;
        let w = px(warm);
        assert!(w[0] > w[2], "色溫 + 要偏暖（紅多於藍）");
        let mut cool = Adjustments::default();
        cool.temp = -80;
        let c = px(cool);
        assert!(c[2] > c[0], "色溫 − 要偏冷（藍多於紅）");

        let mut magenta = Adjustments::default();
        magenta.tint = 80;
        let m = px(magenta);
        assert!(m[1] < m[0] && m[1] < m[2], "色調 + 要偏洋紅（綠被壓下去）");

        let mut bright = Adjustments::default();
        bright.brightness = 100;
        assert!(px(bright)[0] > 120, "亮度 + 應該變亮");
    }

    #[test]
    fn saturation_and_vibrance_only_move_colour_not_grey() {
        // 灰色沒有彩度可加，飽和度與鮮豔度動到它就是在偷改亮度
        let mut grey = solid(4, 4, [128, 128, 128]);
        let mut a = Adjustments::default();
        a.saturation = 100;
        a.vibrance = 100;
        apply_grade(&mut grey, &a);
        for c in grey.get_pixel(0, 0).0 {
            assert!((c as i32 - 128).abs() <= 1, "灰色被調彩度後不該變色");
        }

        let mut color = solid(4, 4, [180, 100, 100]);
        let mut sat = Adjustments::default();
        sat.saturation = 100;
        apply_grade(&mut color, &sat);
        let p = color.get_pixel(0, 0).0;
        assert!(p[0] - p[1] > 80, "飽和度 + 要把色差拉開");
    }

    #[test]
    fn curve_endpoints_stay_put_and_lift_black() {
        // 黑色 + 是「提黑」：純黑要被抬起來，純白不該跟著動
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([0, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([255, 255, 255]));
        let mut a = Adjustments::default();
        a.blacks = 100;
        apply_grade(&mut img, &a);
        assert!(img.get_pixel(0, 0).0[0] > 20, "提黑後純黑要變亮");
        assert_eq!(img.get_pixel(1, 0).0[0], 255, "提黑不該動到純白");
    }

    #[test]
    fn spline_passes_through_its_control_points() {
        let pts = [(0.0, 0.1), (0.25, 0.4), (1.0, 1.0)];
        let sp = Spline::new(&pts);
        for (x, y) in pts {
            assert!((sp.eval(x) - y).abs() < 1e-4, "樣條要通過控制點 ({x}, {y})");
        }
    }

    #[test]
    fn clarity_leaves_a_flat_image_flat() {
        // 局部對比是「原圖減模糊」，平坦的畫面兩者相同，差值必須是零。
        // 邊界處理沒做好的話，四邊會浮出一圈亮框
        let mut img = solid(64, 48, [100, 130, 160]);
        let before = img.clone();
        let mut a = Adjustments::default();
        a.clarity = 100;
        apply_grade(&mut img, &a);
        for (p, q) in img.pixels().zip(before.pixels()) {
            for c in 0..3 {
                assert!((p[c] as i32 - q[c] as i32).abs() <= 1, "平坦畫面不該被清晰度改變");
            }
        }
    }

    /// 系統上第一個可用的字型；沒有字型的環境就讓文字測試自己跳過
    fn any_font() -> Option<FontVec> {
        crate::detect_fonts()
            .first()
            .and_then(|(_, p)| load_font(p))
    }

    #[test]
    fn text_lands_on_the_spot_it_was_placed() {
        let Some(font) = any_font() else { return };
        let mut img = solid(400, 300, [0, 0, 0]);
        let style = SubtitleStyle {
            outline_w: 0,
            ..SubtitleStyle::default()
        };
        let t = TextItem {
            text: "字A".into(),
            x: 0.5,
            y: 0.5,
            size: 200,
            ..TextItem::default()
        };
        draw_texts(&mut img, std::slice::from_ref(&t), &style, &font);
        let bright = |x, y| img.get_pixel(x, y).0[0] > 100;
        assert!(
            (180..220).any(|x| (130..170).any(|y| bright(x, y))),
            "文字要落在指定的中心點附近"
        );
        for (x, y) in [(2u32, 2u32), (397, 2), (2, 297), (397, 297)] {
            assert!(!bright(x, y), "四個角不該被畫到");
        }
    }

    #[test]
    fn text_pushed_off_the_canvas_draws_nothing_and_does_not_panic() {
        // 拖到畫面外的文字要靜靜地不見，不是崩潰、也不是在邊上留一條
        let Some(font) = any_font() else { return };
        let mut img = solid(120, 90, [10, 20, 30]);
        let before = img.clone();
        for (x, y) in [(-3.0f32, 0.5f32), (4.0, 0.5), (0.5, -3.0), (0.5, 4.0)] {
            let t = TextItem {
                text: "邊界".into(),
                x,
                y,
                size: 120,
                rot: 30.0,
            };
            draw_texts(&mut img, std::slice::from_ref(&t), &SubtitleStyle::default(), &font);
        }
        assert_eq!(img.as_raw(), before.as_raw());
    }

    #[test]
    fn a_full_width_glyph_is_about_one_font_size_wide() {
        // 「大小」講的是 em 的高度：1080 高的照片上大小 200 的全形字，
        // 一個字就該佔約 200 像素寬。ab_glyph 的 PxScale 不是以 em 為單位，
        // 少換算一次這裡會大上兩成——預覽（egui 畫的）與存檔就對不起來
        let Some(font) = any_font() else { return };
        let mut img = solid(1200, 1080, [0, 0, 0]);
        let t = TextItem {
            text: "字字字".into(),
            x: 0.5,
            y: 0.5,
            size: 200,
            rot: 0.0,
        };
        let style = SubtitleStyle {
            outline_w: 0,
            ..SubtitleStyle::default()
        };
        draw_texts(&mut img, std::slice::from_ref(&t), &style, &font);
        let (mut x0, mut x1) = (u32::MAX, 0u32);
        for y in 0..img.height() {
            for x in 0..img.width() {
                if img.get_pixel(x, y).0[0] > 100 {
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                }
            }
        }
        assert!(x1 > x0, "應該畫出了東西");
        let per_glyph = (x1 - x0) as f32 / 3.0;
        assert!(
            (per_glyph - 200.0).abs() <= 30.0,
            "一個全形字量到 {per_glyph:.0} 像素，應該接近字級 200"
        );
    }

    #[test]
    fn outline_widens_the_painted_area() {
        // 外框是靠八個方向的偏移堆出來的，寬度沒吃到就等於這個選項沒作用
        let Some(font) = any_font() else { return };
        let painted = |w: i32| {
            let mut img = solid(300, 200, [0, 0, 0]);
            let style = SubtitleStyle {
                color: egui::Color32::WHITE,
                outline_w: w,
                outline_color: egui::Color32::from_rgb(255, 0, 0),
                ..SubtitleStyle::default()
            };
            let t = TextItem {
                text: "邊".into(),
                size: 120,
                y: 0.5,
                ..TextItem::default()
            };
            draw_texts(&mut img, std::slice::from_ref(&t), &style, &font);
            img.pixels().filter(|p| p.0 != [0, 0, 0]).count()
        };
        assert!(painted(8) > painted(0), "外框調寬後上色的面積要變大");
    }

    #[test]
    fn box_blur_of_a_constant_plane_is_that_constant() {
        let plane = vec![0.5f32; 40 * 30];
        for r in [1usize, 5, 17] {
            for v in box_blur(&plane, 40, 30, r) {
                assert!((v - 0.5).abs() < 1e-5, "半徑 {r} 的邊界延伸沒接好");
            }
        }
    }

    /// 主畫面同一組滑桿的濾鏡串在灰階梯上量到的輸出，用來釘住「同一個數字
    /// 在兩邊是同一種效果」。量法：
    /// `ffmpeg -i ramp.png -vf "<濾鏡>" -pix_fmt rgb24 -f rawvideo -`
    /// （ramp.png 為 geq 產生的 0~255 灰階梯）
    fn assert_matches_ffmpeg(adj: Adjustments, want: &[(u8, u8)], tol: i32, what: &str) {
        for &(i, o) in want {
            let mut img = solid(1, 1, [i, i, i]);
            apply_grade(&mut img, &adj);
            let got = img.get_pixel(0, 0).0[0] as i32;
            let d = got - o as i32;
            assert!(
                d.abs() <= tol,
                "{what}：輸入 {i} 時 ffmpeg 給 {o}、這裡給 {got}（差 {d}，容許 {tol}）"
            );
        }
    }

    #[test]
    fn curves_match_the_ffmpeg_filter_they_mirror() {
        // curves=all='0/0 0.024/0 0.25/0.2875 0.9775/1 1/1'
        assert_matches_ffmpeg(
            Adjustments {
                shadows: 25,
                whites: 15,
                blacks: -20,
                ..Adjustments::default()
            },
            &[
                (0, 0), (16, 4), (32, 21), (48, 45), (64, 75), (80, 102), (96, 127),
                (112, 150), (128, 172), (144, 191), (160, 208), (176, 223), (192, 235),
                (208, 244), (224, 250), (240, 254),
            ],
            2,
            "陰影／白色／黑色的曲線",
        );
    }

    #[test]
    fn exposure_contrast_and_brightness_match_the_ffmpeg_filters() {
        // eq=contrast=1.2400:brightness=0.0000:saturation=1.0000
        assert_matches_ffmpeg(
            Adjustments {
                contrast: 30,
                ..Adjustments::default()
            },
            &[
                (0, 0), (32, 7), (64, 48), (96, 86), (128, 127), (160, 165), (192, 206),
                (224, 245),
            ],
            3,
            "對比",
        );
        // eq=contrast=1.0:brightness=0.1000:saturation=1.0（亮度 25 → 0.1）
        assert_matches_ffmpeg(
            Adjustments {
                brightness: 25,
                ..Adjustments::default()
            },
            &[
                (0, 29), (32, 61), (64, 93), (96, 125), (128, 157), (160, 189), (192, 221),
                (224, 253),
            ],
            3,
            "亮度",
        );
        // exposure=exposure=0.750（曝光度 25 → 0.75 EV）
        assert_matches_ffmpeg(
            Adjustments {
                exposure: 25,
                ..Adjustments::default()
            },
            &[(0, 0), (32, 54), (64, 108), (96, 161), (128, 215), (160, 255), (192, 255)],
            3,
            "曝光度",
        );
    }

    #[test]
    fn colour_temperature_matches_the_ffmpeg_filter() {
        // colortemperature=temperature=5100（色溫 40）在灰 128 上量到
        // (128, 115, 105)、在白 255 上量到 (255, 229, 209)
        for (input, want) in [([128u8; 3], [128u8, 115, 105]), ([255; 3], [255, 229, 209])] {
            let mut img = solid(1, 1, input);
            apply_grade(
                &mut img,
                &Adjustments {
                    temp: 40,
                    ..Adjustments::default()
                },
            );
            let p = img.get_pixel(0, 0).0;
            for c in 0..3 {
                assert!(
                    (p[c] as i32 - want[c] as i32).abs() <= 1,
                    "色溫 40 在 {input:?} 上給 {p:?}，ffmpeg 是 {want:?}"
                );
            }
        }
    }

    #[test]
    fn dehaze_matches_the_ffmpeg_filter() {
        // 去朦朧 +50 → colorlevels=rimin=0.1100（減掉那層幕再拉回滿格）
        assert_matches_ffmpeg(
            Adjustments {
                dehaze: 50,
                ..Adjustments::default()
            },
            &[
                (0, 0), (32, 4), (64, 40), (96, 76), (128, 112), (160, 148), (192, 184),
                (224, 220),
            ],
            2,
            "去朦朧（減霧）",
        );
        // 去朦朧 −100 → colorlevels=romin=0.2200（反過來加一層幕上去）
        assert_matches_ffmpeg(
            Adjustments {
                dehaze: -100,
                ..Adjustments::default()
            },
            &[
                (0, 56), (32, 80), (64, 105), (96, 130), (128, 155), (160, 180), (192, 205),
                (224, 230),
            ],
            2,
            "去朦朧（加霧）",
        );
    }

    #[test]
    fn dehaze_brings_colour_back_as_it_lifts_the_veil() {
        // 去朦朧不是只把畫面壓暗：三個通道一起拉開，通道差距會跟著放大，
        // 彩度自己就回來了。只動亮度的話就只是「黑色」滑桿的翻版
        let mut img = solid(4, 4, [150, 130, 120]);
        apply_grade(
            &mut img,
            &Adjustments {
                dehaze: 100,
                ..Adjustments::default()
            },
        );
        let p = img.get_pixel(0, 0).0;
        assert!(
            (p[0] as i32 - p[2] as i32) > (150 - 120),
            "拉開之後通道差距要比原本 30 大，實際是 {}",
            p[0] as i32 - p[2] as i32
        );
    }

    #[test]
    fn the_whole_chain_composes_the_same_way_ffmpeg_does() {
        // 一次疊上多條滑桿，順序錯了或階段之間少夾一次就會整條走偏。
        // ffmpeg 側量的是：
        //   colortemperature=temperature=5100,
        //   exposure=exposure=0.450,
        //   eq=contrast=1.2400:brightness=0.0400:saturation=1.1000,
        //   curves=all='0/0 0.024/0 0.25/0.2875 0.9775/1 1/1',
        //   vibrance=intensity=0.8000
        // 容許 12：單條濾鏡各自都在 ±3 以內（見上面幾個測試），但 ffmpeg 是
        // 一條濾鏡一次 8 位元進出、中間還走一趟有限範圍的 YUV，捨去誤差疊了
        // 五層就到這個量級。這個測試守的是「順序與公式沒跑掉」，不是逐階相同
        let adj = Adjustments {
            temp: 40,
            exposure: 15,
            contrast: 30,
            brightness: 10,
            shadows: 25,
            whites: 15,
            blacks: -20,
            vibrance: 40,
            saturation: 10,
            ..Adjustments::default()
        };
        let want: [(u8, [u8; 3]); 8] = [
            (0, [0, 0, 0]),
            (32, [22, 9, 2]),
            (64, [122, 90, 65]),
            (96, [197, 162, 130]),
            (128, [243, 218, 184]),
            (160, [255, 249, 231]),
            (192, [255, 255, 253]),
            (224, [255, 255, 255]),
        ];
        for (i, w) in want {
            let mut img = solid(1, 1, [i, i, i]);
            apply_grade(&mut img, &adj);
            let got = img.get_pixel(0, 0).0;
            for c in 0..3 {
                assert!(
                    (got[c] as i32 - w[c] as i32).abs() <= 12,
                    "輸入 {i}：整串跑完得到 {got:?}，ffmpeg 是 {w:?}"
                );
            }
        }
    }

    #[test]
    fn vibrance_matches_the_ffmpeg_filter() {
        // vibrance=intensity=0.4000（鮮豔度 20）與 -0.4000 在 (150,100,100) 上
        // 量到 (168,94,94) 與 (137,103,103)
        for (v, want) in [(20, [168u8, 94, 94]), (-20, [137, 103, 103])] {
            let mut img = solid(1, 1, [150, 100, 100]);
            apply_grade(
                &mut img,
                &Adjustments {
                    vibrance: v,
                    ..Adjustments::default()
                },
            );
            let p = img.get_pixel(0, 0).0;
            for c in 0..3 {
                assert!(
                    (p[c] as i32 - want[c] as i32).abs() <= 3,
                    "鮮豔度 {v}：第 {c} 個通道給 {}，ffmpeg 是 {}",
                    p[c],
                    want[c]
                );
            }
        }
    }


    /// 夜空底 + 一團亮煙：測資與實際用法一致（塗掉那團煙）
    fn sky_with_blob(w: u32, h: u32, cx: f32, cy: f32, r: f32) -> RgbImage {
        let mut img = solid(w, h, [12, 14, 20]);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                let d = (dx * dx + dy * dy).sqrt();
                if d < r {
                    // 邊緣柔一點，比較像真的煙
                    let k = (1.0 - d / r).min(1.0);
                    let p = img.get_pixel_mut(x, y);
                    for c in 0..3 {
                        p[c] = (p[c] as f32 + k * 150.0).min(255.0) as u8;
                    }
                }
            }
        }
        img
    }

    #[test]
    fn wipe_density_caps_how_much_gets_cleared() {
        // 濃度不是「清或不清」，而是「最多清到多乾淨」：一半就該留一半的煙。
        // 沒有分層的話這條滑桿等於只有 0 與 100 兩段，拿來輕輕壓淡就沒用了
        let at_center = |density: f32| {
            let mut img = sky_with_blob(120, 90, 60.0, 45.0, 12.0);
            apply_wipe(
                &mut img,
                &[Wipe {
                    pts: vec![[0.5, 0.5]],
                    radius: 18.0 / 120.0,
                    keep_detail: false,
                    density,
                    ..Wipe::default()
                }],
            );
            img.get_pixel(60, 45).0[0] as i32
        };
        let untouched = sky_with_blob(120, 90, 60.0, 45.0, 12.0).get_pixel(60, 45).0[0] as i32;
        let (full, half) = (at_center(1.0), at_center(0.5));
        assert!(
            full < half && half < untouched,
            "濃度要能分出層次：全清 {full}、半清 {half}、原本 {untouched}"
        );
    }

    #[test]
    fn wipe_feather_softens_the_edge_without_moving_it() {
        // 羽化只改「邊緣多柔」，不改筆刷蓋到多大一圈：
        // 靠近邊界的地方羽化高時只清掉一部分，圓心一樣清得乾淨。
        // 兩者混在一起的話，拉羽化會連帶讓筆刷看起來變小
        let sample = |feather: f32, x: u32| {
            let mut img = sky_with_blob(160, 160, 80.0, 80.0, 60.0);
            apply_wipe(
                &mut img,
                &[Wipe {
                    pts: vec![[0.5, 0.5]],
                    radius: 40.0 / 160.0,
                    keep_detail: false,
                    feather,
                    ..Wipe::default()
                }],
            );
            img.get_pixel(x, 80).0[0] as i32
        };
        // 圓心（0.0r）：兩種羽化都在實心核心裡，清得一樣乾淨
        assert!(
            (sample(0.0, 80) - sample(1.0, 80)).abs() <= 3,
            "圓心不該因為羽化而清得比較少"
        );
        // 邊界內側（約 0.8r）：羽化 0 是硬邊、還在核心裡；羽化 1 已經淡出大半
        let (hard, soft) = (sample(0.0, 112), sample(1.0, 112));
        assert!(
            hard < soft,
            "羽化拉高時邊緣要留得比較多：硬邊 {hard}、柔邊 {soft}"
        );
    }

    #[test]
    fn wipe_removes_the_blob_and_leaves_the_rest_alone() {
        // 沒清乾淨的煙塗過去就該不見，四周的夜空不能跟著被動到
        let mut img = sky_with_blob(120, 90, 60.0, 45.0, 12.0);
        let before = img.clone();
        apply_wipe(
            &mut img,
            &[Wipe {
                pts: vec![[0.5, 0.5]],
                radius: 18.0 / 120.0,
                keep_detail: false,
                ..Wipe::default()
            }],
        );
        let mid = img.get_pixel(60, 45).0;
        assert!(
            mid.iter().all(|&v| v < 30),
            "塗到的地方要被夜空補起來，卻還留著 {mid:?}"
        );
        // 離筆跡夠遠的角落一個像素都不該變
        for (x, y) in [(0u32, 0u32), (119, 0), (0, 89), (119, 89)] {
            assert_eq!(
                img.get_pixel(x, y).0,
                before.get_pixel(x, y).0,
                "({x},{y}) 在筆跡外卻被改到"
            );
        }
    }

    #[test]
    fn wipe_works_the_same_on_a_preview_sized_copy() {
        // 預覽是縮圖、存檔是原尺寸：同一筆相對座標要清掉同一塊，
        // 否則會「預覽清乾淨了，存出來還在」
        let same = |w: u32, h: u32| {
            let mut img = sky_with_blob(w, h, w as f32 * 0.5, h as f32 * 0.5, w as f32 * 0.1);
            apply_wipe(
                &mut img,
                &[Wipe {
                    pts: vec![[0.5, 0.5]],
                    radius: 0.15,
                keep_detail: false,
                ..Wipe::default()
                }],
            );
            img.get_pixel(w / 2, h / 2).0
        };
        let small = same(120, 90);
        let big = same(480, 360);
        for c in 0..3 {
            assert!(
                (small[c] as i32 - big[c] as i32).abs() <= 6,
                "縮圖給 {small:?}、原尺寸給 {big:?}，兩邊清得不一樣"
            );
        }
    }

    #[test]
    fn degenerate_wipes_do_nothing() {
        // 沒點、或半徑小到看不見的筆跡不該動到畫面（也不該當掉）
        let mut img = sky_with_blob(40, 30, 20.0, 15.0, 6.0);
        let before = img.clone();
        apply_wipe(
            &mut img,
            &[
                Wipe {
                    pts: vec![],
                    radius: 0.2,
                    keep_detail: false,
                    ..Wipe::default()
                },
                Wipe {
                    pts: vec![[0.5, 0.5]],
                    radius: 0.0,
                    keep_detail: true,
                    ..Wipe::default()
                },
            ],
        );
        assert_eq!(img.as_raw(), before.as_raw());
    }

    #[test]
    fn a_stroke_wipes_along_its_whole_path() {
        // 拖一條線就該整條都清掉，不能只清起點那一個圓
        let mut img = solid(160, 60, [10, 10, 14]);
        for y in 20..40 {
            for x in 20..140 {
                img.put_pixel(x, y, image::Rgb([180, 140, 160]));
            }
        }
        apply_wipe(
            &mut img,
            &[Wipe {
                pts: vec![[0.15, 0.5], [0.5, 0.5], [0.85, 0.5]],
                radius: 20.0 / 160.0,
                keep_detail: false,
                ..Wipe::default()
            }],
        );
        for x in [30u32, 80, 130] {
            let p = img.get_pixel(x, 30).0;
            assert!(p.iter().all(|&v| v < 40), "x={x} 沒被清掉：{p:?}");
        }
    }

    #[test]
    fn keep_detail_saves_the_streaks_but_still_drops_the_smoke() {
        // 煙裡有煙火線條時，勾了「保留煙火紋路」就只該扣掉煙、線條原樣留著；
        // 沒勾則整塊換成夜空，線條也一起不見
        // 尺寸取實際照片的量級：線條粗細與「量煙用的半徑」是等比縮放的，
        // 拿 200px 的縮圖測等於線條佔滿整個取樣窗，量不出煙火與煙的差別
        const N: u32 = 1200;
        let make = || {
            // 夜空中間一大團煙，煙上橫著一條細亮線（煙火線條）
            let mut img = sky_with_blob(N, N, 600.0, 600.0, 260.0);
            for x in 250..950 {
                for y in 598..601 {
                    img.put_pixel(x, y, image::Rgb([250, 240, 220]));
                }
            }
            img
        };
        let wiped = |keep: bool| {
            let mut img = make();
            apply_wipe(
                &mut img,
                &[Wipe {
                    pts: vec![[0.5, 0.5]],
                    radius: 0.22,
                    keep_detail: keep,
                    ..Wipe::default()
                }],
            );
            img
        };
        let (kept, gone) = (wiped(true), wiped(false));
        // 線條會跟著變暗是對的——蓋在它上面的那層煙也被扣掉了；
        // 要看的是它仍然清清楚楚地站在補起來的夜空上
        let on_line = kept.get_pixel(600, 599).0;
        let beside = kept.get_pixel(600, 700).0;
        assert!(
            (0..3).all(|c| on_line[c] as i32 > beside[c] as i32 + 40),
            "勾了保留紋路，煙火線條卻被抹掉：線上 {on_line:?}、旁邊 {beside:?}"
        );
        assert!(
            gone.get_pixel(600, 599).0.iter().all(|&v| v < 60),
            "沒勾就該連線條一起清掉：{:?}",
            gone.get_pixel(600, 599).0
        );
        // 線條旁邊的煙：兩種模式都要清乾淨
        for (name, img) in [("保留紋路", &kept), ("整塊清除", &gone)] {
            let smoke = img.get_pixel(600, 700).0;
            assert!(
                smoke.iter().all(|&v| v < 45),
                "{name}：線條旁邊的煙沒清掉 {smoke:?}"
            );
        }
    }

    #[test]
    fn wipes_apply_the_same_split_as_in_one_go() {
        // 預覽為了不越塗越慢，是「接著上一筆的結果再算新的一筆」
        // （見 SmokeTool::wipe_cache）。這條要成立，分兩次做與一次做完
        // 必須完全一樣，否則塗到一半的畫面會與存檔的成品對不起來
        let base = sky_with_blob(600, 400, 300.0, 200.0, 90.0);
        let mk = |x: f32, y: f32, keep: bool| Wipe {
            pts: vec![[x, y], [x + 0.05, y + 0.03]],
            radius: 0.06,
            keep_detail: keep,
            ..Wipe::default()
        };
        let all = [
            mk(0.35, 0.4, true),
            mk(0.5, 0.5, false),
            mk(0.45, 0.35, true),
        ];
        let mut once = base.clone();
        apply_wipe(&mut once, &all);
        let mut split = base.clone();
        apply_wipe(&mut split, &all[..2]);
        apply_wipe(&mut split, &all[2..]);
        assert_eq!(
            once.as_raw(),
            split.as_raw(),
            "分兩次套用與一次套完結果不同，預覽的中間結果快取就不能用"
        );
    }

    #[test]
    fn a_burst_core_keeps_its_brightness() {
        // 塗到煙火團上時，煙散掉、煙火本身一點都不能暗——這是實照上最刺眼的
        // 症狀：芯比取樣窗還大，量到的「煙」會是芯自己，照扣就把中心壓黑
        const N: u32 = 1200;
        let mut img = sky_with_blob(N, N, 600.0, 600.0, 140.0);
        // 一團連續發亮的芯（半徑 45px，比量煙的取樣窗大得多）
        for y in 0..N {
            for x in 0..N {
                let (dx, dy) = (x as f32 - 600.0, y as f32 - 600.0);
                let d = (dx * dx + dy * dy).sqrt();
                if d < 45.0 {
                    let k = (1.0 - d / 45.0).min(1.0);
                    let p = img.get_pixel_mut(x, y);
                    for c in 0..3 {
                        p[c] = (p[c] as f32 + k * 120.0).min(255.0) as u8;
                    }
                }
            }
        }
        let core_before = img.get_pixel(600, 600).0;
        apply_wipe(
            &mut img,
            &[Wipe {
                pts: vec![[0.5, 0.5]],
                radius: 0.2,
                ..Wipe::default()
            }],
        );
        let core = img.get_pixel(600, 600).0;
        for c in 0..3 {
            assert!(
                core[c] as i32 >= core_before[c] as i32 - 3,
                "煙火中心被壓暗了：{core_before:?} → {core:?}"
            );
        }
        // 芯外面的煙照樣要清掉
        let smoke = img.get_pixel(600, 700).0;
        assert!(
            smoke.iter().all(|&v| v < 45),
            "芯旁邊的煙沒清乾淨：{smoke:?}"
        );
    }

    /// 疊圖片的測試都要一張真的檔案（`ImageItem` 存的是路徑），
    /// 寫到暫存資料夾再讀回來
    fn temp_png(name: &str, img: &image::RgbaImage) -> PathBuf {
        let p = std::env::temp_dir().join(format!("p2v_test_{}_{name}.png", std::process::id()));
        img.save(&p).unwrap();
        // 同一個路徑在不同測試裡可能被重寫，解碼快取要跟著換掉
        overlay_cache().lock().unwrap().remove(&p);
        p
    }

    #[test]
    fn an_overlay_lands_where_it_was_placed_and_keeps_its_size() {
        // 位置與大小都是相對值：中心擺 0.25/0.5、寬度佔四分之一，
        // 800x400 的照片上就該落在 x=200、寬 200
        let mut logo = image::RgbaImage::new(20, 10);
        for p in logo.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let path = temp_png("solid", &logo);
        let mut img = solid(800, 400, [0, 0, 0]);
        draw_images(
            &mut img,
            &[ImageItem {
                path: path.clone(),
                x: 0.25,
                y: 0.5,
                scale: 0.25,
                ..ImageItem::default()
            }],
        );
        // 中心是紅的
        assert_eq!(img.get_pixel(200, 200).0, [255, 0, 0], "圖片沒畫在該在的位置");
        // 寬 200（x 100~300）、高 100（y 150~250）：外面一點都不該被動到
        assert_eq!(img.get_pixel(200, 145).0, [0, 0, 0], "上緣超出去了");
        assert_eq!(img.get_pixel(200, 255).0, [0, 0, 0], "下緣超出去了");
        assert_eq!(img.get_pixel(95, 200).0, [0, 0, 0], "左緣超出去了");
        assert_eq!(img.get_pixel(305, 200).0, [0, 0, 0], "右緣超出去了");
        assert_eq!(img.get_pixel(110, 200).0, [255, 0, 0], "左邊界內該是圖片");
        assert_eq!(img.get_pixel(290, 200).0, [255, 0, 0], "右邊界內該是圖片");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_transparent_overlay_only_paints_where_it_is_opaque() {
        // PNG 的透明背景要留著：全透明的地方照片一個像素都不能變
        let mut logo = image::RgbaImage::new(8, 8);
        for (x, _y, p) in logo.enumerate_pixels_mut() {
            // 左半不透明、右半全透明
            *p = if x < 4 {
                image::Rgba([0, 255, 0, 255])
            } else {
                image::Rgba([0, 255, 0, 0])
            };
        }
        let path = temp_png("alpha", &logo);
        let mut img = solid(200, 200, [30, 30, 30]);
        draw_images(
            &mut img,
            &[ImageItem {
                path: path.clone(),
                x: 0.5,
                y: 0.5,
                scale: 0.5,
                ..ImageItem::default()
            }],
        );
        assert_eq!(img.get_pixel(75, 100).0, [0, 255, 0], "不透明的半邊沒畫上去");
        assert_eq!(img.get_pixel(125, 100).0, [30, 30, 30], "透明的半邊不該被蓋掉");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn overlay_opacity_blends_towards_the_photo() {
        // 不透明度 50% 就該是兩者各半
        let mut logo = image::RgbaImage::new(4, 4);
        for p in logo.pixels_mut() {
            *p = image::Rgba([200, 200, 200, 255]);
        }
        let path = temp_png("half", &logo);
        let mut img = solid(100, 100, [0, 0, 0]);
        draw_images(
            &mut img,
            &[ImageItem {
                path: path.clone(),
                opacity: 0.5,
                scale: 0.5,
                ..ImageItem::default()
            }],
        );
        let mid = img.get_pixel(50, 50).0;
        assert!(
            mid.iter().all(|&v| (v as i32 - 100).abs() <= 2),
            "50% 應該混成一半：{mid:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_rotated_overlay_turns_around_its_own_centre() {
        // 轉 90 度：原本的寬變高、高變寬，中心不動
        let mut logo = image::RgbaImage::new(40, 10);
        for p in logo.pixels_mut() {
            *p = image::Rgba([0, 0, 255, 255]);
        }
        let path = temp_png("rot", &logo);
        let mut img = solid(400, 400, [0, 0, 0]);
        draw_images(
            &mut img,
            &[ImageItem {
                path: path.clone(),
                x: 0.5,
                y: 0.5,
                scale: 0.5,
                rot: 90.0,
                ..ImageItem::default()
            }],
        );
        // 寬 200、高 50 的圖轉 90 度後：縱向 200（y 100~300）、橫向 50（x 175~225）
        assert_eq!(img.get_pixel(200, 120).0, [0, 0, 255], "轉過去之後上下該是圖片");
        assert_eq!(img.get_pixel(200, 280).0, [0, 0, 255], "轉過去之後上下該是圖片");
        assert_eq!(img.get_pixel(120, 200).0, [0, 0, 0], "左右不該還是圖片");
        assert_eq!(img.get_pixel(280, 200).0, [0, 0, 0], "左右不該還是圖片");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn masked_grade_only_touches_where_the_weights_say() {
        // 權重 1 的地方要與整張調色一模一樣、權重 0 的地方一個像素都不動、
        // 中間的權重是兩者的混合——否則「套用遮色片」等於沒有作用
        let img = solid(40, 20, [90, 100, 110]);
        let (w, h) = (img.width() as usize, img.height() as usize);
        let adj = Adjustments {
            exposure: 60,
            ..Adjustments::default()
        };
        let mut whole = img.clone();
        apply_grade(&mut whole, &adj);
        assert_ne!(whole.as_raw(), img.as_raw(), "測試前提：曝光 +60 要看得出差別");
        // 左半 0、右半 1、中間一欄 0.5
        let mut weights = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                weights[y * w + x] = if x < 19 {
                    0.0
                } else if x == 19 {
                    0.5
                } else {
                    1.0
                };
            }
        }
        let mut out = img.clone();
        apply_grade_masked(&mut out, &adj, &weights);
        assert_eq!(out.get_pixel(5, 10).0, img.get_pixel(5, 10).0, "權重 0 的地方動了");
        assert_eq!(out.get_pixel(30, 10).0, whole.get_pixel(30, 10).0, "權重 1 要與整張調色相同");
        let (a, b, m) = (img.get_pixel(19, 10).0, whole.get_pixel(19, 10).0, out.get_pixel(19, 10).0);
        for c in 0..3 {
            let mid = (a[c] as f32 + b[c] as f32) / 2.0;
            assert!((m[c] as f32 - mid).abs() <= 1.0, "權重 0.5 要落在兩者中間：{a:?} {b:?} {m:?}");
        }
        // 滑桿全歸零：不管權重畫成什麼，一個像素都不該動
        let mut out = img.clone();
        apply_grade_masked(&mut out, &Adjustments::default(), &weights);
        assert_eq!(out.as_raw(), img.as_raw());
    }
}
