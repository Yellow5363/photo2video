//! 主體追蹤：在一整批連拍照片裡跟著同一個主體跑（飛鳥、跑動的人、疾駛的車），
//! 算出它在每一張裡的位置，好讓輸出的影片把鏡頭挪過去、主體一直留在畫面中央。
//!
//! 做法是最老實的「樣板比對」：從使用者框起來的那一塊取一片樣板，到下一張
//! 照片裡找最像的位置，找到之後那一塊就慢慢混進樣板。三件事讓它在連拍上
//! 夠準也夠快：
//!
//! * **比的是正規化相關係數（ZNCC）而不是逐點差值**——連拍每張的曝光常常
//!   會跳（逆光、自動測光追不上），比亮度會被那個全域差值淹沒；ZNCC 先把
//!   平均與對比除掉，只認「長得像不像」。
//! * **由粗到細的金字塔搜尋**：先在縮小好幾倍的小圖上大範圍掃，再逐層回到
//!   細的圖上就近修。飛鳥在連拍的兩張之間可以移動大半個畫面，直接在細圖上
//!   全域搜尋要多算上百倍。
//! * **照速度預測下一張的位置**：搜尋範圍以「預測點」為圓心而不是「上一張
//!   的位置」，同樣的範圍就跟得上更快的主體。
//!
//! 只找平移、不找縮放：鏡頭要跟的是主體**在哪**，框多大由使用者用「鏡頭
//! 範圍」決定（見 `App::track_crops`）——每張的框若跟著量出來的主體大小變，
//! 影片就會一直忽遠忽近。

use image::RgbImage;

/// 追蹤用的工作解析度（長邊）。原尺寸完全用不上：主體在 1024px 上還有
/// 幾十個像素，足夠認得出來，而解碼與比對的成本都與面積成正比
pub const WORK_EDGE: u32 = 1024;

/// 樣板取到這麼大就夠用（長邊像素）。比對成本與樣板面積成正比，
/// 主體再大也不必拿每一根羽毛去比
const TMPL_FINE: usize = 48;

/// 粗找那一層的樣板長邊：縮到這麼小才敢在大範圍裡逐點掃
const TMPL_COARSE: usize = 20;

/// 樣板短邊的下限：再小就沒有花紋可認，比出來的是雜訊
const TMPL_MIN: usize = 5;

/// 分數低於這個值就當作「這張沒對上」：改用預測的位置，樣板也不更新
/// （硬把背景收進樣板的話，接下來整段都會跟著背景跑）
const LOST_SCORE: f32 = 0.30;

/// 分數高於這個值才敢拿新的樣子去更新樣板
const LOCK_SCORE: f32 = 0.55;

/// 樣板每次混入新樣子的比例。主體會轉身、拍翅、變大變小，樣板得跟著變；
/// 但混太快會被周圍的背景一點一點帶走（漂移），慢慢混才穩
const TMPL_MIX: f32 = 0.25;

/// 預測下一張位置時，上一次的位移打幾折。飛行速度會變，全額外推容易衝過頭
const VEL_DAMP: f32 = 0.85;

/// 追蹤用的灰階影像（0~255 的浮點數，ZNCC 全程在浮點下算）
#[derive(Clone)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    px: Vec<f32>,
    /// 這張照片的色彩（只有解碼來的原圖有；金字塔縮出來的各層、切出來的
    /// 樣板都沒有）。找鳥頭要用顏色，見 [`find_head`]
    colour: Option<std::sync::Arc<Chroma>>,
}

/// 照片的色彩，存成兩個「對手色」平面：紅綠（R−G）與黃藍（(R+G)/2−B）。
/// 解析度是灰階原圖的一半，正好與差值圖（[`DET_DIFF`] 層）一樣大，
/// 兩者可以逐點對照
pub struct Chroma {
    w: usize,
    h: usize,
    rg: Vec<f32>,
    yb: Vec<f32>,
    /// 原解析度（長寬是 `w`、`h` 的兩倍）每一點有多「純紅」（見 [`redness`]）。
    /// 五色鳥嘴基兩側那對紅點只有幾個像素大，縮一半就跟旁邊的黃、黑混掉了
    red: Vec<f32>,
}

/// 一個像素有多「純紅」：紅要同時明顯高過綠與藍，綠與藍又要差不多，而且要夠亮。
/// 正紅（200, 40, 40）約 160；橘（230, 140, 30）只剩 35；黃、綠、褐都在 0 以下。
///
/// 夠亮這一條擋的是鳥嘴裡叼的果子：實測一批連拍，果子是暗紅褐色（115, 34, 40），
/// 純度跟紅點差不多，十字被拉到果子上；紅點則亮得多（244, 74, 5）
fn redness(r: f32, g: f32, b: f32) -> f32 {
    if r < RED_BRIGHT {
        return -1.0;
    }
    (r - g).min(r - b) - 0.5 * (g - b).abs()
}

impl Chroma {
    /// 從彩色影像縮一半（2×2 取平均）算出對手色
    fn from_rgb(img: &RgbImage) -> Chroma {
        let (iw, ih) = (img.width() as usize, img.height() as usize);
        let (w, h) = ((iw / 2).max(1), (ih / 2).max(1));
        let mut rg = Vec::with_capacity(w * h);
        let mut yb = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = img.get_pixel(
                        ((2 * x + dx).min(iw - 1)) as u32,
                        ((2 * y + dy).min(ih - 1)) as u32,
                    );
                    r += p[0] as f32;
                    g += p[1] as f32;
                    b += p[2] as f32;
                }
                let (r, g, b) = (r / 4.0, g / 4.0, b / 4.0);
                rg.push(r - g);
                yb.push((r + g) / 2.0 - b);
            }
        }
        let mut red = Vec::with_capacity(4 * w * h);
        for y in 0..2 * h {
            for x in 0..2 * w {
                let p = img.get_pixel(x.min(iw - 1) as u32, y.min(ih - 1) as u32);
                red.push(redness(p[0] as f32, p[1] as f32, p[2] as f32));
            }
        }
        Chroma { w, h, rg, yb, red }
    }
}

impl Frame {
    /// 從彩色影像轉灰階（BT.601，與 ffmpeg 在 yuv420p 上用的同一組係數），
    /// 順便留一份色彩（找鳥頭用）
    pub fn from_rgb(img: &RgbImage) -> Frame {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let px = img
            .pixels()
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect();
        Frame { w, h, px, colour: Some(std::sync::Arc::new(Chroma::from_rgb(img))) }
    }

    /// 直接以灰階像素建一張（切塊與測試用）
    fn new(w: usize, h: usize, px: Vec<f32>) -> Frame {
        debug_assert_eq!(px.len(), w * h);
        Frame { w, h, px, colour: None }
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        self.px[y * self.w + x]
    }

    /// 長寬各縮一半（2×2 取平均）。奇數邊的最後一行/列拿自己補，
    /// 縮完至少 1×1
    fn half(&self) -> Frame {
        let (w, h) = ((self.w / 2).max(1), (self.h / 2).max(1));
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            let (y0, y1) = ((2 * y).min(self.h - 1), (2 * y + 1).min(self.h - 1));
            for x in 0..w {
                let (x0, x1) = ((2 * x).min(self.w - 1), (2 * x + 1).min(self.w - 1));
                px.push(
                    (self.at(x0, y0) + self.at(x1, y0) + self.at(x0, y1) + self.at(x1, y1)) * 0.25,
                );
            }
        }
        Frame::new(w, h, px)
    }

    /// 金字塔：`[0]` 是自己，往後每層縮一半，共 `top + 1` 層
    fn pyramid(&self, top: usize) -> Vec<Frame> {
        let mut levels = Vec::with_capacity(top + 1);
        levels.push(self.clone());
        for l in 0..top {
            levels.push(levels[l].half());
        }
        levels
    }

    /// 以 (x, y) 為左上角切一塊 `w`×`h`；超出邊界的取邊界上的像素
    /// （主體貼著畫面邊緣時樣板才不會缺一角）
    fn patch(&self, x: i32, y: i32, w: usize, h: usize) -> Frame {
        let mut px = Vec::with_capacity(w * h);
        for dy in 0..h {
            let sy = (y + dy as i32).clamp(0, self.h as i32 - 1) as usize;
            for dx in 0..w {
                let sx = (x + dx as i32).clamp(0, self.w as i32 - 1) as usize;
                px.push(self.at(sx, sy));
            }
        }
        Frame::new(w, h, px)
    }
}

/// 樣板：像素、平均值，以及「扣掉平均之後的長度」（ZNCC 的分母之一）。
/// 這兩個量只跟樣板有關，每層先算好，比對時每個候選位置就少一輪計算
struct Tmpl {
    f: Frame,
    mean: f32,
    norm: f32,
}

impl Tmpl {
    fn new(f: Frame) -> Tmpl {
        let n = f.px.len().max(1) as f32;
        let mean = f.px.iter().sum::<f32>() / n;
        let norm = f.px.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>().sqrt();
        Tmpl { f, mean, norm }
    }
}

/// 樣板的金字塔（與影像那邊同一種縮法，兩邊層層對得起來）
fn tmpl_pyramid(f: &Frame, top: usize) -> Vec<Tmpl> {
    f.pyramid(top).into_iter().map(Tmpl::new).collect()
}

/// 樣板與 `f` 上以 (x0, y0) 為左上角那一塊的正規化相關係數（−1 ~ 1）。
///
/// 分母帶著候選那一塊自己的對比：一整片沒花紋的天空與任何樣板都「不相關」，
/// 分數自然趨近 0，不必另外擋
fn zncc(f: &Frame, x0: usize, y0: usize, t: &Tmpl) -> f32 {
    let (tw, th) = (t.f.w, t.f.h);
    let (mut sp, mut spp, mut spt) = (0.0f32, 0.0f32, 0.0f32);
    for y in 0..th {
        let frow = (y0 + y) * f.w + x0;
        let trow = y * tw;
        for x in 0..tw {
            let p = f.px[frow + x];
            sp += p;
            spp += p * p;
            spt += p * t.f.px[trow + x];
        }
    }
    let n = (tw * th) as f32;
    // Σ(p−p̄)(t−t̄) 展開後與 p̄ 相關的兩項相消，只剩這一式
    let num = spt - t.mean * sp;
    let var = (spp - sp * sp / n).max(0.0);
    let den = var.sqrt() * t.norm;
    // 分母趨近 0＝其中一邊完全沒有對比，這種「相關」沒有意義
    if den < 1e-3 {
        0.0
    } else {
        (num / den).clamp(-1.0, 1.0)
    }
}

/// 在 `f` 上以 `centre`（該層的像素座標）為圓心、半徑 `r` 的範圍裡找樣板最像
/// 的位置。回傳（中心 x, 中心 y, 分數），峰值兩側用拋物線內插取到次像素——
/// 鏡頭的位置全靠這一步才不會一格一格地跳
fn scan(f: &Frame, t: &Tmpl, centre: (f32, f32), r: i32) -> (f32, f32, f32) {
    let (tw, th) = (t.f.w, t.f.h);
    if f.w < tw || f.h < th {
        return (centre.0, centre.1, -1.0);
    }
    // 候選是「左上角」的位置，範圍夾在整塊樣板都還在影像裡的區間
    let (mx, my) = ((f.w - tw) as i32, (f.h - th) as i32);
    let c0 = (centre.0 - tw as f32 / 2.0).round() as i32;
    let c1 = (centre.1 - th as f32 / 2.0).round() as i32;
    let (lo_x, hi_x) = ((c0 - r).clamp(0, mx), (c0 + r).clamp(0, mx));
    let (lo_y, hi_y) = ((c1 - r).clamp(0, my), (c1 + r).clamp(0, my));
    let (mut bx, mut by, mut best) = (lo_x, lo_y, f32::MIN);
    for y in lo_y..=hi_y {
        for x in lo_x..=hi_x {
            let s = zncc(f, x as usize, y as usize, t);
            if s > best {
                best = s;
                bx = x;
                by = y;
            }
        }
    }
    // 峰值兩側各再量一點，用拋物線把頂點內插出來（頂在邊界就不內插）
    let sub = |a: f32, b: f32, c: f32| -> f32 {
        let d = a - 2.0 * b + c;
        if d.abs() < 1e-6 {
            0.0
        } else {
            ((a - c) / (2.0 * d)).clamp(-1.0, 1.0)
        }
    };
    let ox = if bx > 0 && bx < mx {
        sub(
            zncc(f, bx as usize - 1, by as usize, t),
            best,
            zncc(f, bx as usize + 1, by as usize, t),
        )
    } else {
        0.0
    };
    let oy = if by > 0 && by < my {
        sub(
            zncc(f, bx as usize, by as usize - 1, t),
            best,
            zncc(f, bx as usize, by as usize + 1, t),
        )
    } else {
        0.0
    };
    (
        bx as f32 + tw as f32 / 2.0 + ox,
        by as f32 + th as f32 / 2.0 + oy,
        best,
    )
}

/// 一張照片上的追蹤結果。座標是**相對座標**（0~1，對著旋轉後的畫布），
/// 每張照片的尺寸不同也對得起來
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub cx: f32,
    pub cy: f32,
    /// 這一張比出來的分數（−1 ~ 1）。呼叫端拿它決定「這一張要不要請
    /// 使用者過目」——勉強對上的（一片綠葉上總找得到「還算像」的地方）
    /// 與完全沒對上的一樣不能默默放行
    pub score: f32,
    /// 有沒有真的對上。false＝分數太低，回傳的是照速度預測的位置
    pub locked: bool,
    /// 找到的那一塊與**出發時**的樣子有多像（−1 ~ 1）。
    ///
    /// `score` 比的是一路慢慢混進新樣子的樣板；主體離開畫面後追蹤器還在
    /// 「勉強對上」，就會把背景一點一點混進樣板，混到後來跟葉子比得非常像
    /// （樣板漂移）。拿它判斷「這張還是不是主體」會被騙；`same` 比的是
    /// 最初那隻鳥，漂移騙不了它
    pub same: f32,
}

/// 一條追蹤的狀態。從使用者框的那一張出發，往後（或往前）一張一張餵進去
pub struct Tracker {
    /// 樣板，以及它取自金字塔的哪一層
    tmpl: Frame,
    /// 出發時的樣板（不跟著更新），用來量 [`Hit::same`]
    orig: Frame,
    level: usize,
    /// 上一張抓到的中心（相對座標）
    last: (f32, f32),
    /// 主體框的相對大小（決定搜尋範圍要開多大）
    size: (f32, f32),
    /// 上一次的位移（相對座標）：拿來預測下一張大概會在哪
    vel: (f32, f32),
    /// 連續幾張沒對上（越久沒對上就把搜尋範圍開得越大，好重新找回來）
    lost: u32,
}

impl Tracker {
    /// 用 `frame` 上的框 `rect`（x0, y0, x1, y1 相對座標）起一條追蹤。
    /// 框太小（樣板短邊不到 [`TMPL_MIN`] 個像素）就回 None
    pub fn new(frame: &Frame, rect: [f32; 4]) -> Option<Tracker> {
        if frame.w < TMPL_MIN || frame.h < TMPL_MIN {
            return None;
        }
        let x0 = rect[0].min(rect[2]).clamp(0.0, 1.0);
        let x1 = rect[0].max(rect[2]).clamp(0.0, 1.0);
        let y0 = rect[1].min(rect[3]).clamp(0.0, 1.0);
        let y1 = rect[1].max(rect[3]).clamp(0.0, 1.0);
        let (bw, bh) = ((x1 - x0) * frame.w as f32, (y1 - y0) * frame.h as f32);
        if bw < TMPL_MIN as f32 || bh < TMPL_MIN as f32 {
            return None;
        }
        // 挑一層讓樣板長邊不超過 TMPL_FINE；但短邊不能因此縮到認不出花紋
        let mut level = 0usize;
        while bw.max(bh) / (1 << level) as f32 > TMPL_FINE as f32
            && bw.min(bh) / (1 << (level + 1)) as f32 >= TMPL_MIN as f32
        {
            level += 1;
        }
        let pyr = frame.pyramid(level);
        let f = &pyr[level];
        let k = (1 << level) as f32;
        let tw = ((bw / k).round() as usize).clamp(TMPL_MIN, f.w);
        let th = ((bh / k).round() as usize).clamp(TMPL_MIN, f.h);
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        let tmpl = f.patch(
            (cx * f.w as f32 - tw as f32 / 2.0).round() as i32,
            (cy * f.h as f32 - th as f32 / 2.0).round() as i32,
            tw,
            th,
        );
        Some(Tracker {
            orig: tmpl.clone(),
            tmpl,
            level,
            last: (cx, cy),
            size: (x1 - x0, y1 - y0),
            vel: (0.0, 0.0),
            lost: 0,
        })
    }

    /// 在下一張照片裡找出主體
    pub fn find(&mut self, frame: &Frame) -> Hit {
        // 照上一次的位移預測這一張大概在哪，搜尋範圍以預測點為圓心：
        // 同樣的範圍就跟得上更快的主體
        let pred = (
            (self.last.0 + self.vel.0 * VEL_DAMP).clamp(0.0, 1.0),
            (self.last.1 + self.vel.1 * VEL_DAMP).clamp(0.0, 1.0),
        );
        let speed = self.vel.0.hypot(self.vel.1);
        // 範圍＝主體大小＋這一段的速度＋一點餘裕；連續沒對上就一路開大找回來
        let r_rel = ((self.size.0.max(self.size.1) * 0.9 + speed * 1.2 + 0.03)
            * (1.0 + self.lost as f32 * 0.6))
            .min(0.8);

        // 粗找的那一層：把樣板縮到 TMPL_COARSE 左右，才敢在大範圍裡逐點掃
        let mut up = 0usize;
        while (self.tmpl.w.max(self.tmpl.h) >> up) > TMPL_COARSE
            && (self.tmpl.w.min(self.tmpl.h) >> (up + 1)) >= 4
        {
            up += 1;
        }
        let coarse = self.level + up;
        let pyr = frame.pyramid(coarse);
        let tpyr = tmpl_pyramid(&self.tmpl, up);

        // 先在最粗的那一層大範圍掃，再逐層回到細的圖上就近修
        let long = frame.w.max(frame.h) as f32;
        let top = &pyr[coarse];
        let r = ((r_rel * long) / (1 << coarse) as f32).ceil().max(2.0) as i32;
        let mut best = scan(
            top,
            &tpyr[up],
            (pred.0 * top.w as f32, pred.1 * top.h as f32),
            r,
        );
        for l in (self.level..coarse).rev() {
            best = scan(&pyr[l], &tpyr[l - self.level], (best.0 * 2.0, best.1 * 2.0), 2);
        }

        let fine = &pyr[self.level];
        let found = (
            (best.0 / fine.w as f32).clamp(0.0, 1.0),
            (best.1 / fine.h as f32).clamp(0.0, 1.0),
        );
        let locked = best.2 >= LOST_SCORE;
        let centre = if locked { found } else { pred };
        // 沒對上時速度不重算（拿比錯的位置算速度會把預測甩到天邊），
        // 只讓它照原方向慢慢滑過去
        if locked {
            self.vel = (centre.0 - self.last.0, centre.1 - self.last.1);
            self.lost = 0;
        } else {
            self.vel = (self.vel.0 * VEL_DAMP, self.vel.1 * VEL_DAMP);
            self.lost += 1;
        }
        self.last = centre;
        // 對得夠好才把新的樣子混一點進樣板：主體會轉身、拍翅、變大變小
        if best.2 >= LOCK_SCORE {
            let (tw, th) = (self.tmpl.w, self.tmpl.h);
            let fresh = fine.patch(
                (best.0 - tw as f32 / 2.0).round() as i32,
                (best.1 - th as f32 / 2.0).round() as i32,
                tw,
                th,
            );
            for (old, new) in self.tmpl.px.iter_mut().zip(&fresh.px) {
                *old += (new - *old) * TMPL_MIX;
            }
        }
        // 與出發時的樣子比一次（漂移過的樣板會把背景也當成主體，見 Hit::same）
        let (tw, th) = (self.orig.w, self.orig.h);
        let same = if fine.w >= tw && fine.h >= th {
            let x = ((best.0 - tw as f32 / 2.0).round() as i32).clamp(0, (fine.w - tw) as i32);
            let y = ((best.1 - th as f32 / 2.0).round() as i32).clamp(0, (fine.h - th) as i32);
            zncc(fine, x as usize, y as usize, &Tmpl::new(self.orig.clone()))
        } else {
            -1.0
        };
        Hit { cx: centre.0, cy: centre.1, score: best.2, locked, same }
    }
}

/// 挑完路徑後，重新評估每一張「有多可信」（0~1）。
///
/// 光看那一團差值有多亮是不夠的——被風吹的葉子每一張都很亮、分數都接近滿分，
/// 使用者卻一張都不會想留。真正有意義的問題是「**這一張接得上前後嗎**」：
/// 用前後兩張等速外推出來的位置，與這張選中的位置差多少，除以整段的典型
/// 步幅。接得上的照原本的分數，接不上的往下壓，於是「該檢查的那幾張」就
/// 自己浮出來了（也正是自動補框要優先處理的那些）
///
/// `fit_pos` 有給的話，「接不接得上前後」改用這組位置來算（鏡頭實際要對的
/// 位置，例如鳥頭）。動態框會在翅膀與身體之間跳來跳去，拿它算接不接得上，
/// 明明框到頭的照片也會被當成沒把握。其他幾條（換人、整段不動、被瞬移夾住）
/// 判斷的是「追的是不是同一個東西」，照樣用動態框——那幾條不能省
pub fn path_scores(
    cands: &[Vec<Found>],
    picked: &[Option<usize>],
    fit_pos: Option<&[Option<(f32, f32)>]>,
) -> Vec<f32> {
    let n = picked.len();
    let mut out = vec![0.0f32; n];
    let pos: Vec<Option<(f32, f32)>> = (0..n).map(|i| picked[i].map(|k| track_pos(&cands[i][k]))).collect();
    // 整段的典型步幅：拿它當尺，快飛的連拍與慢慢晃的連拍才用同一套標準
    let mut steps: Vec<f32> = Vec::new();
    for i in 1..n {
        if let (Some(a), Some(b)) = (pos[i - 1], pos[i]) {
            steps.push((b.0 - a.0).hypot(b.1 - a.1));
        }
    }
    steps.sort_by(f32::total_cmp);
    let typical = steps.get(steps.len() / 2).copied().unwrap_or(0.02).max(0.01);
    // 「接不接得上」那一項用的位置有自己的典型步幅（頭與動態框走的距離不同）
    let fit_typical = match fit_pos {
        None => typical,
        Some(f) => {
            let mut s: Vec<f32> = (1..n)
                .filter_map(|i| match (f[i - 1].or(pos[i - 1]), f[i].or(pos[i])) {
                    (Some(a), Some(b)) => Some((b.0 - a.0).hypot(b.1 - a.1)),
                    _ => None,
                })
                .collect();
            s.sort_by(f32::total_cmp);
            s.get(s.len() / 2).copied().unwrap_or(0.02).max(0.01)
        }
    };

    // 再問一次「這一路上還是同一個東西嗎」：把路徑照外觀指紋切成幾段
    // （相鄰兩張長得完全不像＝中途換人了），拿**最長的那一段**當作主體的樣子，
    // 每張再照它與這個樣子有多像打分。
    //
    // 這一步是必要的：主體停下來不動時，逐張比差異完全看不見牠，路徑會平順地
    // 滑到旁邊的樹葉上——位置接得上、差值也很亮，只有「長得不像」會露餡
    let mut runs: Vec<(usize, usize)> = Vec::new(); // [起, 迄)
    let mut start = 0usize;
    for i in 1..n {
        let same = match (picked[i - 1], picked[i]) {
            (Some(a), Some(b)) => sig_corr(&cands[i - 1][a].sig, &cands[i][b].sig) >= SIG_SWITCH,
            _ => false,
        };
        if !same {
            if i > start {
                runs.push((start, i));
            }
            start = i;
        }
    }
    if n > start {
        runs.push((start, n));
    }
    // 最長的那一段代表主體（樹葉那種岔出去的段落通常短得多）
    let main = runs.iter().copied().max_by_key(|(a, b)| b - a);
    let mut refsig = [0.0f32; SIG_N * SIG_N];
    if let Some((a, b)) = main {
        let mut cnt = 0.0f32;
        for i in a..b {
            if let Some(k) = picked[i] {
                for (r, v) in refsig.iter_mut().zip(&cands[i][k].sig) {
                    *r += v;
                }
                cnt += 1.0;
            }
        }
        if cnt > 0.0 {
            let norm = refsig.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-3);
            for r in refsig.iter_mut() {
                *r /= norm;
            }
        }
    }

    // 還有一種騙局，位置與外觀都看不出來：整段**停在原地**。
    //
    // 主體停下來不動時差值圖看不見牠，路徑會滑到旁邊被風吹著顫的樹葉上，
    // 然後乖乖待在那裡——位置接得上、也一直是同一片葉子，只有「整段幾乎
    // 沒在移動，這一批其他張卻都在動」會露餡。實測一段翠鳥連拍，跟丟的
    // 那 15 張速度是全批中位數的四分之一。
    //
    // 先照「速度暴衝」把路徑切成幾段（那是換人的瞬間），再把幾乎不動的
    // 那幾段標成要檢查。整批本來就慢的（主體停在枝頭）不算——那時大多數
    // 張都很慢，中位數自己會跟著低下來
    let mut speeds = vec![0.0f32; n];
    for i in 1..n {
        if let (Some(a), Some(b)) = (pos[i - 1], pos[i]) {
            speeds[i] = (b.0 - a.0).hypot(b.1 - a.1);
        }
    }
    let mut still = vec![false; n];
    // 被「瞬移」夾在中間的小段。主體不會一下子跳到畫面另一頭、過一會兒又
    // 跳回來——那一段多半是跟到了別的東西。實測一批 164 張的連拍：鳥飛出
    // 畫面右緣的下一張，路徑瞬間跳到左上角一片在動的葉子（分數 0.99），
    // 跟了七張之後鳥回來、又瞬間跳回右邊。位置一路平順、差值也很亮，只有
    // 「兩頭都是瞬移」露了餡
    let mut island = vec![false; n];
    // 頭的版本：只拿通過驗證的頭來量，沒有頭的那幾張跳過；而且只認**真正
    // 的瞬移**（一步跳超過畫面的 HEAD_TELEPORT）。飛得快的鳥頭偶爾一步跳
    // 0.06～0.07，拿一般的標準量會把前面一大段誤判成小島（實測一批連拍前
    // 52 張全被標要檢查）；但鳥停在空中、動態偵測看不見牠的那十幾張，路徑
    // 會一下跳到畫面最左邊一團緩緩飄的白色光點（跳了三分之一個畫面），
    // 那裡也湊得出夠強的「頭」，只有兩頭的瞬移露得出馬腳
    let mut head_island = vec![false; n];
    if let Some(f) = fit_pos {
        let seen: Vec<usize> = (0..n).filter(|&i| f[i].is_some()).collect();
        let mut breaks: Vec<usize> = vec![0];
        for t in 1..seen.len() {
            let (a, b) = (f[seen[t - 1]].unwrap(), f[seen[t]].unwrap());
            let gap = (seen[t] - seen[t - 1]) as f32;
            if (b.0 - a.0).hypot(b.1 - a.1) / gap > (fit_typical * 4.0).max(HEAD_TELEPORT) {
                breaks.push(t);
            }
        }
        breaks.push(seen.len());
        let longest = breaks.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        for w in breaks.windows(2) {
            if w[1] - w[0] < longest && (w[1] - w[0]) * 5 < seen.len() {
                for &i in &seen[w[0]..w[1]] {
                    head_island[i] = true;
                }
            }
        }
    }
    {
        let mut breaks: Vec<usize> = vec![0];
        for i in 1..n {
            if speeds[i] > (typical * 4.0).max(0.05) {
                breaks.push(i);
            }
        }
        breaks.push(n);
        let longest = breaks.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        for w in breaks.windows(2) {
            let (a, b) = (w[0], w[1]);
            // 不是最長的那段（主體的主線），又短於全批的兩成：可疑的小島
            if b - a < longest && (b - a) * 5 < n {
                island[a..b].fill(true);
            }
            if b - a < 3 {
                continue; // 太短的段落看不出快慢
            }
            let mut sp: Vec<f32> = (a + 1..b).map(|i| speeds[i]).collect();
            sp.sort_by(f32::total_cmp);
            let run_med = sp[sp.len() / 2];
            // 這一段幾乎不動，而且它不是整批的主旋律
            if run_med < typical * 0.3 && (b - a) * 5 < n * 3 {
                still[a..b].fill(true);
            }
        }
    }

    for i in 0..n {
        let Some(k) = picked[i] else { continue };
        // 「整段不動」「被瞬移夾住」是**不知道框的是不是主體時**的間接懷疑。
        // 那一張若已經找到通過前後驗證的鳥頭（fit_pos 有值），就有了更直接的
        // 證據，不再套這兩條——停在枝頭吃東西的鳥，頭本來就幾乎不動，實測
        // 一批這樣的連拍，套了會把三十幾張明明框到頭的照片標成要檢查
        let verified = fit_pos.is_some_and(|f| f[i].is_some());
        let doubt = if still[i] {
            0.3
        } else if island[i] {
            0.4
        } else {
            1.0
        };
        let base = cands[i][k].score * doubt;
        // 前後都在才算得出「接不接得上」；在邊界就只看那一團自己的分數
        let fp = |j: usize| fit_pos.map_or(pos[j], |f| f[j].or(pos[j]));
        let dev = match (i.checked_sub(1).map(fp).flatten(), fp(i), (i + 1 < n).then(|| fp(i + 1)).flatten())
        {
            (Some(a), Some(b), Some(c)) => {
                let mid = ((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0);
                (b.0 - mid.0).hypot(b.1 - mid.1)
            }
            _ => 0.0,
        };
        // 差一個典型步幅還算正常，差三四個就幾乎不可信了
        let fit = 1.0 / (1.0 + (dev / (fit_typical * 1.5)).powi(2));
        // 與「主體的樣子」像不像：完全不像的直接壓到要檢查的程度
        let look = ((sig_corr(&cands[i][k].sig, &refsig) + 1.0) / 2.0).clamp(0.0, 1.0);
        let look = (look * 1.6).min(1.0);
        out[i] = if verified {
            // 頭已經通過前後驗證：那是最直接的證據。「接不接得上」「長得像
            // 不像」是拿動態框猜的間接分數，鳥從遠拍飛到特寫、外觀變很多時
            // 會亂扣分。只留「被兩次真正的瞬移夾住」（見 head_island）
            cands[i][k].score * if head_island[i] { 0.4 } else { 1.0 }
        } else {
            base * fit * look
        }
        .clamp(0.0, 1.0);
    }
    out
}

/// 把「只有一張突然跳掉」的位置抹掉（相鄰三張取中位數）。
///
/// 真正的移動是連續的：前後兩張都往同一邊走的才留得下來。單張的暴衝多半是
/// 那一張認錯了東西（框到旁邊晃動的樹枝），平滑只會把它抹成一段搖晃，
/// 先用中位數換掉才乾淨——而且中位數取的是**真的出現過的位置**，
/// 不像平均會憑空造出一個誰都不在的點
pub fn despike(pts: &mut [(f32, f32)]) {
    if pts.len() < 3 {
        return;
    }
    let src = pts.to_vec();
    let med3 = |a: f32, b: f32, c: f32| a.max(b).min(a.min(b).max(c));
    for i in 1..pts.len() - 1 {
        pts[i].0 = med3(src[i - 1].0, src[i].0, src[i + 1].0);
        pts[i].1 = med3(src[i - 1].1, src[i].1, src[i + 1].1);
    }
}

/// 把逐張抓出來的位置磨順。
///
/// 比對難免有一兩個像素的抖動，直接拿去當鏡頭位置會整支影片都在震。
/// 來回各跑一次指數平滑（先順著、再倒著）＝零相位：只把抖動磨掉，不會
/// 像單向平滑那樣讓鏡頭整個慢半拍、老是追在主體屁股後面。
///
/// `strength` 0＝原樣不動，1＝最順（鏡頭走得很懶，主體會在框裡晃）
pub fn smooth(pts: &mut [(f32, f32)], strength: f32) {
    let s = strength.clamp(0.0, 1.0);
    if pts.len() < 3 || s <= 0.0 {
        return;
    }
    let a = 1.0 - 0.92 * s;
    for i in 1..pts.len() {
        pts[i].0 = pts[i - 1].0 + (pts[i].0 - pts[i - 1].0) * a;
        pts[i].1 = pts[i - 1].1 + (pts[i].1 - pts[i - 1].1) * a;
    }
    for i in (0..pts.len() - 1).rev() {
        pts[i].0 = pts[i + 1].0 + (pts[i].0 - pts[i + 1].0) * a;
        pts[i].1 = pts[i + 1].1 + (pts[i].1 - pts[i + 1].1) * a;
    }
}

// ───────────────────────── 自動框選主體 ─────────────────────────
//
// 追蹤要有人先框一張才跑得動；「自動框選」則是替**每一張**各找一次主體，
// 讓使用者一開始就有東西可以檢查、修，而不是對著空白畫面自己框。
//
// 靠的是連拍最本質的那個特徵：**主體在動，背景不動**。把相鄰兩張對齊
// （相機自己也會晃，所以先估一個整張的位移補掉）再相減，動的東西就跳出來。
//
// 關鍵是**同時看前後兩張**、取兩張差值的**逐點較小值**：主體在這一張的位置，
// 對前一張與後一張都是「新出現的東西」，兩邊都亮；牠在前一張留下的殘影只在
// 「與前一張的差」裡亮，與後一張的差是暗的，取小值就把殘影消掉了。曝光跳動、
// 一片飄過的雲這類只發生在單邊的變化，同樣被消掉。

/// 算差值圖的工作層（長邊約 512px）。
///
/// 這一層的細緻度決定「多小的主體還看得見」：飛遠的鳥在原圖裡可能只有
/// 0.7% 寬，在 256px 的圖上不到兩個像素，模糊一下就沒了——那正是整段
/// 追丟的原因。512px 上牠還有三四個像素，救得回來
const DET_DIFF: usize = 1;

/// 估「整張位移」的工作層（長邊約 128px）。相機的晃動是大尺度的，
/// 在小圖上估又快又穩
const DET_SHIFT: usize = 3;

/// 估整張位移時的搜尋半徑（`DET_SHIFT` 層的像素）＝原圖的 9% 左右。
/// 連拍之間相機不會移動得比這更多，開太大只是慢
const DET_SHIFT_R: i32 = 12;

/// 差值要比「整張的典型差值」高出這麼多倍，才算是真的有東西在那裡動
const DET_PEAK_OVER: f32 = 4.0;

/// 峰值的絕對下限（0~255 的灰階差）。整張都沒什麼變化時，
/// 相對門檻會把雜訊放大成「主體」，這條擋掉它
const DET_PEAK_MIN: f32 = 10.0;

/// 從峰值往外長到峰值的幾成為止（決定框的大小）
const DET_GROW: f32 = 0.35;

/// 長出來的區域超過整張的這個比例就當作「不是主體」——那多半是整片
/// 曝光變了、或整張糊掉，框住半個畫面對鏡頭一點幫助也沒有
const DET_MAX_AREA: f32 = 0.35;

/// 框往外留的餘裕（各邊各留框寬的這個比例）。差值圖只認得出「在動的那一塊」，
/// 鳥的身體與尾羽動得少、邊緣容易被切掉，留一點才框得住整隻
const DET_MARGIN: f32 = 0.18;

/// 外觀指紋的邊長（取 `SIG_N`×`SIG_N` 的小圖）。認的是「這一塊長得像不像」，
/// 8×8 就夠分辨鳥與葉子，又小到可以整批留著
const SIG_N: usize = 8;

/// 自動框出來的主體：框（相對座標）、有多有把握（0~1），
/// 以及框裡那一塊的外觀指紋
#[derive(Clone, Copy, Debug)]
pub struct Found {
    pub rect: [f32; 4],
    pub score: f32,
    /// 補償相機位移之後，這一團比補償之前亮了幾倍。相機沒動時恆為 1；
    /// 釘在畫面上的浮水印會遠大於 1（見 [`drop_overlays`]）
    pub shift_gain: f32,
    /// 在這一團附近最像鳥頭的幾處：(x, y, 有多像)，相對座標，最像的在前，
    /// 不足的是 0 分（見 [`HeadMaps::find`]）。動態偵測找到的常是拍動的
    /// 翅膀，鏡頭要對的是頭；要挑哪一處見 [`choose_heads`]
    pub heads: [(f32, f32, f32); HEAD_PEAKS],
    /// `heads` 每一處附近的紅點（相對座標），找不到是 None（見 [`red_spot`]）。
    /// 只用來決定十字最後擺哪：挑頭、驗證、判斷要不要檢查一律照 `heads`，
    /// 免得紅點把「這一批的頭清不清楚」之類的判斷帶偏
    pub reds: [Option<(f32, f32)>; HEAD_PEAKS],
    /// 這張照片最清晰那一塊有多清晰（整張共用，見 HeadMaps::sharp_ref）。
    /// 跟整批比低很多＝這張整個糊掉了（失焦或晃動），交給使用者處理
    pub sharp: f32,
    /// 框裡那一塊縮成 8×8、扣掉平均並正規化的樣子。
    /// 拿它跟別張的比對就知道「還是同一個東西嗎」——差值圖只認得出
    /// 「有沒有在動」，認不出「動的是誰」
    sig: [f32; SIG_N * SIG_N],
}

/// 兩個指紋有多像（−1 ~ 1）。兩邊都已經扣過平均、正規化過，內積就是相關係數
fn sig_corr(a: &[f32; SIG_N * SIG_N], b: &[f32; SIG_N * SIG_N]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().clamp(-1.0, 1.0)
}

/// 從 `f` 上切出 `rect`（該層的像素座標）那一塊，縮成 8×8 的指紋
fn signature(f: &Frame, x0: f32, y0: f32, x1: f32, y1: f32) -> [f32; SIG_N * SIG_N] {
    let mut px = [0.0f32; SIG_N * SIG_N];
    let (w, h) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    for gy in 0..SIG_N {
        for gx in 0..SIG_N {
            // 每一格取它中心點的值（框本來就模糊，不必再平均一次）
            let sx = (x0 + w * (gx as f32 + 0.5) / SIG_N as f32).clamp(0.0, f.w as f32 - 1.0);
            let sy = (y0 + h * (gy as f32 + 0.5) / SIG_N as f32).clamp(0.0, f.h as f32 - 1.0);
            px[gy * SIG_N + gx] = f.at(sx as usize, sy as usize);
        }
    }
    let n = px.len() as f32;
    let mean = px.iter().sum::<f32>() / n;
    let norm = px.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>().sqrt();
    if norm > 1e-3 {
        for v in px.iter_mut() {
            *v = (*v - mean) / norm;
        }
    } else {
        px = [0.0; SIG_N * SIG_N];
    }
    px
}

/// 一張照片最多找幾個「正在動」的候選。畫面裡會動的東西不只主體
/// （搖晃的樹枝、被風吹的葉子、閃動的光斑），多留幾個候選，
/// 到底哪一個是主體交給 [`choose_path`] 整批一起決定
pub const DET_CANDIDATES: usize = 6;

/// 在 `cur` 上找出「正在動」的東西，最多 `max` 個，強的排前面。
///
/// `prev`／`next` 是拿來比對的另外兩張。兩張缺一不可地重要：主體在 `cur`
/// 的位置對兩張**都**不一樣，牠在別張留下的殘影卻只對其中一張不一樣，
/// 取兩份差值的較小值就只剩下主體現在的位置。所以第一張與最後一張要**拿
/// 同一側的兩張**來比（例如第 0 張用第 1、2 張），不能只給一張——只給一張
/// 的話殘影與本尊一樣亮，框到哪一個純屬運氣
pub fn detect_candidates(
    prev: Option<&Frame>,
    cur: &Frame,
    next: Option<&Frame>,
    max: usize,
) -> Vec<Found> {
    let cp = cur.pyramid(DET_SHIFT);
    let mut maps: Vec<Frame> = Vec::new();
    // 同樣兩張、但**不補償**相機位移的差值：拿來認出「釘在畫面上」的東西
    // （浮水印、日期戳記、相框），見下面的 overlay 判斷
    let mut raws: Vec<Frame> = Vec::new();
    for other in [prev, next].into_iter().flatten() {
        if other.w != cur.w || other.h != cur.h {
            continue; // 尺寸不同的照片沒得逐點相減
        }
        let op = other.pyramid(DET_SHIFT);
        // 先估相機自己的位移，把它補掉，剩下的才是「畫面裡真的在動的東西」。
        // 位移是在 DET_SHIFT 那層量的，換到 DET_DIFF 層要照層差放大
        let s = best_shift(&cp[DET_SHIFT], &op[DET_SHIFT], DET_SHIFT_R);
        let k = 1 << (DET_SHIFT - DET_DIFF);
        maps.push(diff_at(&cp[DET_DIFF], &op[DET_DIFF], (s.0 * k, s.1 * k)));
        raws.push(diff_at(&cp[DET_DIFF], &op[DET_DIFF], (0, 0)));
    }
    // 兩份差值圖用同一種方式合併（逐點取小值、再糊一下），才比得起來
    let combine = |mut v: Vec<Frame>| -> Option<Frame> {
        let mut f = match v.len() {
            0 => return None,
            1 => v.pop().unwrap(),
            _ => {
                // 逐點取小值：只有「前後兩張都覺得這裡不一樣」的地方留得下來
                let (a, b) = (&v[0], &v[1]);
                let px = a.px.iter().zip(&b.px).map(|(x, y)| x.min(*y)).collect();
                Frame::new(a.w, a.h, px)
            }
        };
        blur3(&mut f);
        Some(f)
    };
    // 單邊（頭尾兩張）少了一層把關，門檻抬高一點免得框到雜訊
    let strict = if maps.len() < 2 { 1.6 } else { 1.0 };
    let Some(m) = combine(maps) else { return Vec::new() };
    let raw = combine(raws).unwrap_or_else(|| m.clone());

    // 「典型差值」用中位數：會動的東西只佔一小塊，中位數代表的就是背景的殘差
    let mut sorted: Vec<f32> = m.px.clone();
    sorted.sort_by(f32::total_cmp);
    let typical = sorted[sorted.len() / 2].max(0.5);
    let floor = (DET_PEAK_MIN * strict).max(typical * DET_PEAK_OVER * strict);
    let level = (m.px.len() as f32 * DET_MAX_AREA) as usize;
    let (fw, fh) = (m.w as f32, m.h as f32);
    // 找鳥頭用的累加表：整張算一次，每個候選共用（沒有色彩的照片就不找）
    let heads = cur.colour.as_ref().map(|c| HeadMaps::new(c, &m, &sharpness(&cp[0], &m), &coarse_detail(&cp[DET_DIFF + 1], &m)));

    let mut used = vec![false; m.px.len()];
    let mut out: Vec<Found> = Vec::new();
    while out.len() < max {
        // 找還沒被前面的候選佔走的最高點
        let (peak, at) = m.px.iter().enumerate().fold((0.0f32, usize::MAX), |(bv, bi), (i, v)| {
            if !used[i] && *v > bv {
                (*v, i)
            } else {
                (bv, bi)
            }
        });
        if at == usize::MAX || peak < floor {
            break;
        }
        // 從峰值往外長：連著的、還有峰值三成以上的格子都算同一團
        let limit = (peak * DET_GROW).max(typical * 2.0);
        let Some(blob) = grow(&m, at, limit, level, &mut used) else {
            continue; // 長太大＝整片都在變（曝光跳動之類），換下一個峰值
        };
        let gain = shift_gain(&blob, &m, &raw);
        // 這一團（鳥身）自己的平均移動量：找頭時拿它當尺（見 HEAD_GATE）
        let bird_motion = blob.cells.iter().map(|&j| m.px[j]).sum::<f32>() / blob.cells.len().max(1) as f32;
        let (x0, y0, x1, y1) = blob.bbox;
        // 位置用**差值加權的重心**而不是外接框的中心：同一隻鳥的外接框會
        // 隨著翅膀張合忽胖忽瘦，中心跟著左右跳；重心穩得多
        let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.0f32);
        for &i in &blob.cells {
            let w = m.px[i];
            sx += (i % m.w) as f32 * w;
            sy += (i / m.w) as f32 * w;
            sw += w;
        }
        let (cx, cy) = if sw > 0.0 {
            (sx / sw + 0.5, sy / sw + 0.5)
        } else {
            ((x0 + x1) as f32 / 2.0, (y0 + y1) as f32 / 2.0)
        };
        // 差值只認得出「動得最多的那一塊」，往外留一點餘裕才框得住整隻
        let bw = (x1 - x0 + 1) as f32 * (1.0 + DET_MARGIN * 2.0);
        let bh = (y1 - y0 + 1) as f32 * (1.0 + DET_MARGIN * 2.0);
        let rect = [
            ((cx - bw / 2.0) / fw).clamp(0.0, 1.0),
            ((cy - bh / 2.0) / fh).clamp(0.0, 1.0),
            ((cx + bw / 2.0) / fw).clamp(0.0, 1.0),
            ((cy + bh / 2.0) / fh).clamp(0.0, 1.0),
        ];
        // 把握程度：峰值比背景高出越多越有把握，長出來的東西越大則越可疑
        let contrast = ((peak / (typical * DET_PEAK_OVER) - 1.0) / 2.0).clamp(0.0, 1.0);
        let area = (rect[2] - rect[0]) * (rect[3] - rect[1]);
        let compact = (1.0 - area / DET_MAX_AREA).clamp(0.0, 1.0);
        // 指紋取自**照片本身**（不是差值圖）：要認的是這一塊長什麼樣子。
        // cp 是 cur 的金字塔，第 DET_DIFF 層正是差值圖那一層的灰階原圖
        let sig = signature(&cp[DET_DIFF], rect[0] * fw, rect[1] * fh, rect[2] * fw, rect[3] * fh);
        let found_heads = heads.as_ref().map_or([(0.0, 0.0, 0.0); HEAD_PEAKS], |hm| hm.find(rect, bird_motion));
        out.push(Found {
            rect,
            score: (contrast * 0.65 + compact * 0.35).clamp(0.0, 1.0),
            sig,
            shift_gain: gain,
            heads: found_heads,
            reds: match cur.colour.as_deref() {
                Some(c) => found_heads.map(|(x, y, s)| (s > 0.0).then(|| red_spot(c, x, y)).flatten()),
                None => [None; HEAD_PEAKS],
            },
            sharp: heads.as_ref().map_or(0.0, |hm| hm.sharp_ref),
        });
    }
    out
}

/// 找鳥頭時搜尋的範圍：動態框的幾倍大（至少畫面的 [`HEAD_WIN_MIN`]）。
///
/// 動態框常常落在拍動的翅膀上，頭在旁邊；鳥飛近時翅膀很大，頭可以離框中心
/// 好幾個框寬，範圍開小了就找不到
const HEAD_WIN: f32 = 5.0;
const HEAD_WIN_MIN: f32 = 0.15;

/// 頭的大小（佔長邊的比例）：算「這一小塊有多少種顏色」用的半徑
const HEAD_RADIUS: f32 = 0.012;

/// 附近的移動量要到「這隻鳥自己的移動量」的幾成，才算跟著鳥一起在動（滿分）。
///
/// 不能拿背景的典型殘差當尺：實測一批背景雜亂的連拍，黃色葉子挨著深色
/// 樹枝，「一小塊裡顏色很多樣」這點跟鳥頭一樣，被風吹得微微晃動也早就超過
/// 背景殘差的好幾倍——框就被搬到葉子上。鳥頭是跟著鳥整隻一起移動的，
/// 移動量跟鳥身差不多；葉子只是晃，遠不到鳥身的一半
const HEAD_GATE: f32 = 0.5;

/// 色彩變化量（0~1）至少要到這裡才算找到頭
const HEAD_MIN: f32 = 0.06;

/// 頭所在的那一塊要有「整張最清晰那一塊」的幾成清晰度（滿分）
const HEAD_SHARP: f32 = 0.5;

/// 頭所在的那一塊，「細節 ÷ 輪廓」要有整張最高的幾成才算滿分（見
/// [`HeadMaps::crisp_at`]）。門檻刻意放低：只淘汰真的糊掉的。攝影師對焦的
/// 就是鳥，常常整張只有鳥是清楚的；散景裡天空透過樹葉的白色亮邊，邊緣強度
/// 跟鳥一樣高（光看 [`HEAD_SHARP`] 那一關擋不住，實測一批連拍連續十幾張框在
/// 左邊的散景裡），但細節遠不如輪廓，這一關擋得住
const HEAD_CRISP: f32 = 0.2;

/// 清晰度圖：每一點的拉普拉斯絕對值（細節越銳利越大），在最細的那層算、
/// 再 2×2 取平均縮到 `to` 那一層的大小（與差值圖、色彩對齊）。
///
/// 要在最細的那層算：散景與對焦清楚的地方，差別全在最細的那些邊緣上，
/// 先縮小再算就都一樣糊了
fn sharpness(fine: &Frame, to: &Frame) -> Frame {
    let (w, h) = (fine.w, fine.h);
    let lap = lap_abs(fine);
    let mut px = Vec::with_capacity(to.w * to.h);
    for y in 0..to.h {
        for x in 0..to.w {
            let (x0, y0) = ((2 * x).min(w - 1), (2 * y).min(h - 1));
            let (x1, y1) = ((2 * x + 1).min(w - 1), (2 * y + 1).min(h - 1));
            px.push((lap[y0 * w + x0] + lap[y0 * w + x1] + lap[y1 * w + x0] + lap[y1 * w + x1]) * 0.25);
        }
    }
    Frame::new(to.w, to.h, px)
}

/// 每一點拉普拉斯的絕對值（邊上一圈是 0）
fn lap_abs(f: &Frame) -> Vec<f32> {
    let (w, h) = (f.w, f.h);
    let mut lap = vec![0.0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let c = f.at(x, y);
            let v = 4.0 * c - f.at(x - 1, y) - f.at(x + 1, y) - f.at(x, y - 1) - f.at(x, y + 1);
            lap[y * w + x] = v.abs();
        }
    }
    lap
}

/// 粗一級的輪廓圖：在比 `to` 再小一半的那層算拉普拉斯，放大回 `to` 的大小。
///
/// 拿來當 [`sharpness`] 的分母：光看最細的邊緣有多強會被**對比**騙——散景裡
/// 白色天空透過樹葉的亮邊，糊歸糊，邊緣強度照樣比鳥頭高（實測一批連拍，
/// 左邊散景的「清晰度」跟整張最清晰的鳥一樣）。糊掉的邊緣只剩粗輪廓、沒有
/// 細節；對焦清楚的地方兩者都有。兩者相除，就只剩「銳不銳利」，與亮暗對比無關
fn coarse_detail(coarse: &Frame, to: &Frame) -> Frame {
    let (w, h) = (coarse.w, coarse.h);
    let lap = lap_abs(coarse);
    let mut px = Vec::with_capacity(to.w * to.h);
    for y in 0..to.h {
        for x in 0..to.w {
            px.push(lap[(y / 2).min(h.saturating_sub(1)) * w + (x / 2).min(w.saturating_sub(1))]);
        }
    }
    Frame::new(to.w, to.h, px)
}

/// 找鳥頭用的累加表（整張的對手色、移動量與清晰度），一張照片算一次、
/// 所有候選共用
struct HeadMaps {
    w: usize,
    h: usize,
    s_rg: Vec<f64>,
    s_yb: Vec<f64>,
    q_rg: Vec<f64>,
    q_yb: Vec<f64>,
    s_m: Vec<f64>,
    s_sharp: Vec<f64>,
    /// 粗一級輪廓的累加表（清晰度的分母，見 [`coarse_detail`]）
    s_coarse: Vec<f64>,
    /// 分母的底：平坦的天空、散景裡只有雜訊，細節與輪廓都接近 0，相除會亂跳。
    /// 每一點的輪廓至少算這麼多（整張輪廓的中位數）
    floor: f64,
    /// 這張照片「最清晰的那一塊」有多清晰（頭那麼大一塊的平均，取全張的
    /// 前 1%）。找頭時拿它當尺；整張都糊的照片它也就很低
    sharp_ref: f32,
    /// 「細節 ÷ 輪廓」的前 1%（見 [`HeadMaps::crisp_at`]）
    crisp_ref: f32,
}

impl HeadMaps {
    fn new(c: &Chroma, m: &Frame, sharp: &Frame, coarse: &Frame) -> HeadMaps {
        let (w, h) = (c.w.min(m.w).min(sharp.w).min(coarse.w), c.h.min(m.h).min(sharp.h).min(coarse.h));
        let n = (w + 1) * (h + 1);
        let mut maps = HeadMaps {
            w,
            h,
            s_rg: vec![0.0; n],
            s_yb: vec![0.0; n],
            q_rg: vec![0.0; n],
            q_yb: vec![0.0; n],
            s_m: vec![0.0; n],
            s_sharp: vec![0.0; n],
            s_coarse: vec![0.0; n],
            floor: 0.0,
            sharp_ref: 0.0,
            crisp_ref: 0.0,
        };
        for y in 0..h {
            let (mut a, mut b, mut qa, mut qb, mut mm) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let (mut ss, mut sc) = (0.0f64, 0.0f64);
            for x in 0..w {
                let rg = c.rg[y * c.w + x] as f64;
                let yb = c.yb[y * c.w + x] as f64;
                a += rg;
                b += yb;
                qa += rg * rg;
                qb += yb * yb;
                mm += m.px[y * m.w + x] as f64;
                ss += sharp.px[y * sharp.w + x] as f64;
                sc += coarse.px[y * coarse.w + x] as f64;
                let k = (y + 1) * (w + 1) + x + 1;
                let up = y * (w + 1) + x + 1;
                maps.s_rg[k] = maps.s_rg[up] + a;
                maps.s_yb[k] = maps.s_yb[up] + b;
                maps.q_rg[k] = maps.q_rg[up] + qa;
                maps.q_yb[k] = maps.q_yb[up] + qb;
                maps.s_m[k] = maps.s_m[up] + mm;
                maps.s_sharp[k] = maps.s_sharp[up] + ss;
                maps.s_coarse[k] = maps.s_coarse[up] + sc;
            }
        }
        let mut cs: Vec<f32> = coarse.px.clone();
        cs.sort_by(f32::total_cmp);
        maps.floor = cs.get(cs.len() / 2).copied().unwrap_or(0.0).max(1.0) as f64;
        // 最清晰的那一塊：以頭的大小為窗，掃一遍取前 1%（取最大值會被
        // 單一顆亮點或雜訊帶走）
        let r = ((w.max(h) as f32 * HEAD_RADIUS).round() as usize).max(2);
        if w > 2 * r + 1 && h > 2 * r + 1 {
            let (mut v, mut q): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
            let n = ((2 * r + 1) * (2 * r + 1)) as f64;
            let mut y = r;
            while y + r + 1 < h {
                let mut x = r;
                while x + r + 1 < w {
                    v.push((maps.sum(&maps.s_sharp, x - r, y - r, x + r + 1, y + r + 1) / n) as f32);
                    q.push(maps.crisp_at(x - r, y - r, x + r + 1, y + r + 1) as f32);
                    x += 2;
                }
                y += 2;
            }
            v.sort_by(f32::total_cmp);
            q.sort_by(f32::total_cmp);
            maps.sharp_ref = v.get(v.len() * 99 / 100).copied().unwrap_or(0.0);
            maps.crisp_ref = q.get(q.len() * 99 / 100).copied().unwrap_or(0.0);
        }
        maps
    }

    /// [x0, x1) × [y0, y1) 這一塊是不是糊的：最細的細節 ÷ 粗一級的輪廓
    /// （見 [`coarse_detail`]）。與亮暗對比無關，對焦清楚的地方遠高於散景。
    ///
    /// 只拿來**否決糊掉的**，不拿來比誰比較清楚：鳥頭是大塊的藍、黃、紅，
    /// 粗輪廓本來就強，這個比例反而常輸給細碎的羽毛與樹枝（實測一批近距離
    /// 的連拍，拿它打分，十幾張的十字被拉到頭旁邊的翅膀上）
    fn crisp_at(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let n = ((x1 - x0) * (y1 - y0)) as f64;
        self.sum(&self.s_sharp, x0, y0, x1, y1) / (self.sum(&self.s_coarse, x0, y0, x1, y1) + self.floor * n)
    }

    /// [x0, x1) × [y0, y1) 的總和
    fn sum(&self, v: &[f64], x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let w1 = self.w + 1;
        v[y1 * w1 + x1] + v[y0 * w1 + x0] - v[y0 * w1 + x1] - v[y1 * w1 + x0]
    }

    /// 在動態框 `rect` 附近找鳥頭，回傳頭的中心（相對座標）。
    ///
    /// 鳥頭的特徵是「**很小的範圍裡擠了很多種顏色**」——五色鳥的藍頭頂、黃臉、
    /// 紅喉、黑嘴全在一起；翅膀與背景的顏色則單一。但光看顏色會被散景騙：
    /// 藍天從綠葉縫透出來，同樣是一小塊裡有兩種顏色。所以再乘上「那裡有沒有
    /// 在動」——頭跟著鳥一起移動，散景不會。
    ///
    /// 回傳最像頭的前幾處（彼此至少隔一個頭寬，最像的在前，不足的補 0 分）。
    /// 單張最像的不一定是頭：五色鳥的翅膀也泛著藍綠光，張開又清楚時會贏過
    /// 頭。留幾個備選，交給整批的連貫性來挑（見 [`choose_heads`]）
    fn find(&self, rect: [f32; 4], bird_motion: f32) -> [(f32, f32, f32); HEAD_PEAKS] {
        let mut out = [(0.0, 0.0, 0.0); HEAD_PEAKS];
        let (w, h) = (self.w, self.h);
        if w < 8 || h < 8 {
            return out;
        }
        let (cx, cy) = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0);
        let ww = ((rect[2] - rect[0]) * HEAD_WIN).max(HEAD_WIN_MIN);
        let wh = ((rect[3] - rect[1]) * HEAD_WIN).max(HEAD_WIN_MIN);
        let r = ((w.max(h) as f32 * HEAD_RADIUS).round() as usize).max(2);
        let px = |u: f32, n: usize| ((u.clamp(0.0, 1.0) * n as f32) as usize).min(n);
        let (x0, x1) = (px(cx - ww / 2.0, w).max(r), px(cx + ww / 2.0, w).min(w - r));
        let (y0, y1) = (px(cy - wh / 2.0, h).max(r), px(cy + wh / 2.0, h).min(h - r));
        let gate_full = (bird_motion * HEAD_GATE).max(1.0) as f64;
        let sharp_full = (self.sharp_ref * HEAD_SHARP).max(1e-3) as f64;
        let crisp_full = (self.crisp_ref * HEAD_CRISP).max(1e-6) as f64;
        let mut all: Vec<(f32, usize, usize)> = Vec::new();
        let mut y = y0;
        while y < y1 {
            let mut x = x0;
            while x < x1 {
                let (a0, b0, a1, b1) = (x - r, y - r, x + r + 1, y + r + 1);
                let n = ((a1 - a0) * (b1 - b0)) as f64;
                let mean_rg = self.sum(&self.s_rg, a0, b0, a1, b1) / n;
                let mean_yb = self.sum(&self.s_yb, a0, b0, a1, b1) / n;
                let var = (self.sum(&self.q_rg, a0, b0, a1, b1) / n - mean_rg * mean_rg)
                    + (self.sum(&self.q_yb, a0, b0, a1, b1) / n - mean_yb * mean_yb);
                let motion = self.sum(&self.s_m, a0, b0, a1, b1) / n;
                // 頭幾乎都在整張最清晰的地方：攝影師對焦的就是頭與眼。
                // 散景裡的葉縫透光、拍動中糊掉的翅膀都過不了這一關
                let sharp = self.sum(&self.s_sharp, a0, b0, a1, b1) / n;
                let crisp = self.crisp_at(a0, b0, a1, b1);
                let score = var.max(0.0).sqrt() / 255.0
                    * (motion / gate_full).min(1.0)
                    * (sharp / sharp_full).min(1.0)
                    * (crisp / crisp_full).min(1.0);
                if score as f32 >= HEAD_MIN {
                    all.push((score as f32, x, y));
                }
                x += 2;
            }
            y += 2;
        }
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        let apart = (3 * r) as f32;
        let mut got = 0;
        for (s, x, y) in all {
            let far = out[..got].iter().all(|&(u, v, _)| {
                (u * w as f32 - x as f32 - 0.5).hypot(v * h as f32 - y as f32 - 0.5) >= apart
            });
            if far {
                out[got] = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32, s);
                got += 1;
                if got == HEAD_PEAKS {
                    break;
                }
            }
        }
        out
    }
}

/// 圖庫比對的取樣格數（每邊）
const LIB_GRID: usize = 12;

/// 一塊鳥頭的「長相」：以某一點為中心、邊長 `size` 的方塊縮成 12×12，
/// 灰階扣平均再正規化（只看明暗花紋），兩個對手色保留原值（五色鳥頭上的藍、
/// 黃、紅本身就是線索）
#[derive(Clone)]
pub struct HeadLook {
    v: Vec<f32>,
}

/// 色彩在比對裡的份量（灰階已正規化成長度 1）
const LIB_COLOUR_W: f32 = 1.0 / 400.0;

impl Frame {
    /// 雙線性取樣（超出邊界的取邊界）
    fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let a = self.px[y0 * self.w + x0] * (1.0 - fx) + self.px[y0 * self.w + x1] * fx;
        let b = self.px[y1 * self.w + x0] * (1.0 - fx) + self.px[y1 * self.w + x1] * fx;
        a * (1.0 - fy) + b * fy
    }

    /// 以 (cx, cy)（工作圖像素）為中心、邊長 `size` 的那一塊長什麼樣子
    pub fn head_look(&self, cx: f32, cy: f32, size: f32) -> Option<HeadLook> {
        self.head_look_m(cx, cy, size, false)
    }

    /// 同 [`Frame::head_look`]，`mirror` 為真時左右翻轉（朝左飛的範本也能拿來
    /// 比朝右飛的鳥；標記點在正中央，翻過來還是在正中央）
    pub fn head_look_m(&self, cx: f32, cy: f32, size: f32, mirror: bool) -> Option<HeadLook> {
        let c = self.colour.as_deref()?;
        let g = LIB_GRID;
        let step = size / g as f32;
        let mut lum = Vec::with_capacity(g * g);
        let mut col = Vec::with_capacity(2 * g * g);
        for j in 0..g {
            for i in 0..g {
                let ii = if mirror { g - 1 - i } else { i };
                let x = cx - size / 2.0 + (ii as f32 + 0.5) * step;
                let y = cy - size / 2.0 + (j as f32 + 0.5) * step;
                lum.push(self.sample(x, y));
                let (qx, qy) = (((x / 2.0) as usize).min(c.w - 1), ((y / 2.0) as usize).min(c.h - 1));
                col.push(c.rg[qy * c.w + qx] * LIB_COLOUR_W);
                col.push(c.yb[qy * c.w + qx] * LIB_COLOUR_W);
            }
        }
        let m = lum.iter().sum::<f32>() / lum.len() as f32;
        let norm = lum.iter().map(|v| (v - m) * (v - m)).sum::<f32>().sqrt().max(1e-3);
        let mut v: Vec<f32> = lum.iter().map(|x| (x - m) / norm).collect();
        v.extend(col);
        Some(HeadLook { v })
    }
}

impl HeadLook {
    /// 兩塊長相差多遠（越小越像）
    pub fn dist(&self, o: &HeadLook) -> f32 {
        self.v.iter().zip(o.v.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
    }
}

/// 拿圖庫裡標好的鳥頭，在工作圖 `f` 的 (cx, cy) 附近找最像的那一處，回傳
/// (x, y, 距離, 用了第幾張範本)；x, y 是範本標記點對應的位置（工作圖像素）。
///
/// 範本是「以使用者標的那一點為中心」切下來的，所以找到最像的那一塊，它的
/// 中心就是使用者會標的位置：側面時是眼睛，正面時是兩眼之間——不必再分
/// 正面側面。鳥遠近不同，所以範本大小從 `sizes` 裡逐一試
pub fn match_library(
    f: &Frame,
    cx: f32,
    cy: f32,
    sizes: &[f32],
    lib: &[HeadLook],
    skip: &dyn Fn(usize) -> bool,
) -> Option<(f32, f32, f32, usize)> {
    let mut best: Option<(f32, f32, f32, usize)> = None;
    for &s in sizes {
        let reach = s * 1.5;
        let step = (s / 8.0).max(1.0);
        let n = (reach / step) as i32;
        for j in -n..=n {
            for i in -n..=n {
                let (x, y) = (cx + i as f32 * step, cy + j as f32 * step);
                if x < 0.0 || y < 0.0 || x >= f.w as f32 || y >= f.h as f32 {
                    continue;
                }
                let Some(look) = f.head_look(x, y, s) else { return None };
                for (k, t) in lib.iter().enumerate() {
                    if skip(k) {
                        continue;
                    }
                    let d = look.dist(t);
                    if best.is_none_or(|b| d < b.2) {
                        best = Some((x, y, d, k));
                    }
                }
            }
        }
    }
    best
}
/// 第 i 張挑中的頭 (x, y)（[`choose_heads`] 回傳的那一處）附近的紅點
pub fn red_of(cands: &[Found], x: f32, y: f32) -> Option<(f32, f32)> {
    cands.iter().find_map(|c| {
        c.heads
            .iter()
            .zip(c.reds.iter())
            .find(|(h, _)| h.2 > 0.0 && (h.0 - x).abs() < 1e-6 && (h.1 - y).abs() < 1e-6)
            .and_then(|(_, r)| *r)
    })
}

/// 紅點的紅色值至少要這麼亮（0 到 255，見 [`redness`]）
const RED_BRIGHT: f32 = 150.0;

/// 紅點要多紅才算（見 [`redness`]）
const RED_MIN: f32 = 60.0;

/// 在頭附近多遠的範圍裡找紅點（[`HEAD_RADIUS`] 的幾倍）
const RED_SEARCH: f32 = 4.0;

/// 一塊紅至少要有幾個像素（原解析度）才算紅點
const RED_PIXELS: usize = 4;

/// 成對的兩塊紅點，面積最多差幾倍
const RED_PAIR_AREA: f32 = 4.0;

/// 成對的兩塊紅點，高低差最多是左右距離的幾倍（頭歪一點也認得）
const RED_PAIR_TILT: f32 = 0.6;

/// 成對的兩塊紅點，左右距離最多是一塊寬度的幾倍（中間夾著的是嘴基）
const RED_PAIR_GAP: f32 = 6.0;

/// 在找到的頭 (x, y)（相對座標）附近找五色鳥嘴基兩側的那對紅點，回傳兩點的
/// 中點（相對座標）；找不到是 None。
///
/// 「顏色最雜的一小塊」只說得出頭大概在哪：近拍時常落在喉嚨的紅黃色塊、或
/// 頭頂與背景的交界。那對紅點的位置是固定的——正面看兩點夾著嘴基、兩眼就在
/// 它們外側，中點正好在兩眼之間（使用者提供的辨識特徵）。
///
/// 一定要**成對**才算：五色鳥脖子兩側還有一塊較大的紅斑，側面時只看得到
/// 一邊，那時只找單一一塊紅，十字會被拉到脖子上、離眼睛一大段（實測一批
/// 側飛的連拍，二十幾張都這樣）。成對＝兩塊大小相近、高度相近、左右排開；
/// 胸口那一圈紅是一整條，湊不成一對
fn red_spot(c: &Chroma, x: f32, y: f32) -> Option<(f32, f32)> {
    let (rw, rh) = (2 * c.w, 2 * c.h);
    if c.red.len() != rw * rh || rw < 8 || rh < 8 {
        return None;
    }
    let r = (rw.max(rh) as f32 * HEAD_RADIUS).max(2.0);
    let rad = r * RED_SEARCH;
    let (cx, cy) = (x * rw as f32, y * rh as f32);
    let x0 = (cx - rad).max(0.0) as usize;
    let x1 = ((cx + rad) as usize).min(rw - 1);
    let y0 = (cy - rad).max(0.0) as usize;
    let y1 = ((cy + rad) as usize).min(rh - 1);
    let (ww, wh) = (x1 - x0 + 1, y1 - y0 + 1);
    // 搜尋範圍裡的紅色像素連成一塊一塊（四鄰接）
    let mut label = vec![0u32; ww * wh];
    // 每一塊：(像素數, x 總和, y 總和, 最左, 最右)
    let mut blobs: Vec<(usize, f32, f32, usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for sy in 0..wh {
        for sx in 0..ww {
            if label[sy * ww + sx] != 0 || c.red[(y0 + sy) * rw + x0 + sx] <= RED_MIN {
                continue;
            }
            let id = blobs.len() as u32 + 1;
            let mut b = (0usize, 0.0f32, 0.0f32, usize::MAX, 0usize);
            label[sy * ww + sx] = id;
            stack.push((sx, sy));
            while let Some((px, py)) = stack.pop() {
                b.0 += 1;
                b.1 += px as f32 + 0.5;
                b.2 += py as f32 + 0.5;
                b.3 = b.3.min(px);
                b.4 = b.4.max(px);
                let mut visit = |qx: usize, qy: usize| {
                    let k = qy * ww + qx;
                    if label[k] == 0 && c.red[(y0 + qy) * rw + x0 + qx] > RED_MIN {
                        label[k] = id;
                        stack.push((qx, qy));
                    }
                };
                if px > 0 {
                    visit(px - 1, py);
                }
                if px + 1 < ww {
                    visit(px + 1, py);
                }
                if py > 0 {
                    visit(px, py - 1);
                }
                if py + 1 < wh {
                    visit(px, py + 1);
                }
            }
            blobs.push(b);
        }
    }
    let spots: Vec<(f32, f32, f32, f32)> = blobs
        .iter()
        .filter(|b| b.0 >= RED_PIXELS)
        .map(|b| (b.1 / b.0 as f32, b.2 / b.0 as f32, b.0 as f32, (b.4 - b.3 + 1) as f32))
        .collect();
    // 挑最像一對、中點又離頭最近的
    let mut best: Option<(f32, (f32, f32))> = None;
    for i in 0..spots.len() {
        for j in i + 1..spots.len() {
            let (a, b) = (spots[i], spots[j]);
            let (dx, dy) = ((a.0 - b.0).abs(), (a.1 - b.1).abs());
            let size = a.3.max(b.3);
            let pair = a.2.max(b.2) <= a.2.min(b.2) * RED_PAIR_AREA
                && dy <= dx * RED_PAIR_TILT
                && dx >= size
                && dx <= size * RED_PAIR_GAP;
            if !pair {
                continue;
            }
            let mid = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            let d = (mid.0 + x0 as f32 - cx).hypot(mid.1 + y0 as f32 - cy);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, mid));
            }
        }
    }
    best.map(|(_, (mx, my))| ((mx + x0 as f32) / rw as f32, (my + y0 as f32) / rh as f32))
}

/// 這一團「補償相機位移之後比補償之前亮了幾倍」（見 [`Found::shift_gain`]）
fn shift_gain(blob: &Blob, comp: &Frame, raw: &Frame) -> f32 {
    let n = blob.cells.len().max(1) as f32;
    let comp_mean = blob.cells.iter().map(|&i| comp.px[i]).sum::<f32>() / n;
    let raw_mean = blob.cells.iter().map(|&i| raw.px[i]).sum::<f32>() / n;
    comp_mean / raw_mean.max(0.5)
}

/// 同一個位置（佔畫面的比例）以內就算「同一個地方」
const OVERLAY_RADIUS: f32 = 0.04;

/// 在整批裡有這麼大比例的照片、同一個位置都出現候選，才可能是釘在畫面上的
const OVERLAY_SHARE: f32 = 0.10;

/// 那些候選「補償後放大」的倍數中位數要到這裡（相機沒動時恆為 1）
const OVERLAY_GAIN: f32 = 2.5;

/// 剔除**釘在畫面上**的東西（浮水印、日期戳記、相框）。
///
/// 相機一移動，程式會把整張的位移補掉，好讓背景對齊、只剩真正在動的主體。
/// 但浮水印是跟著畫面走的——背景對齊了，它反而錯開了，於是變成整張最亮的
/// 「移動物體」。實測一批 164 張的連拍，鳥飛出畫面後最後 10 張整段黏在
/// 攝影師的簽名上，而且分數都很高。
///
/// 單看一張分不出來：半透明的浮水印疊在變動的背景上，補償前的差值也不是 0
/// （實測 0.9～16），而相機在移動時，真正的鳥補償後也會放大個三到七倍。
/// 但浮水印有一個鳥絕對沒有的特徵——**每一張都在畫面上同一個位置**。所以
/// 整批一起看：在固定位置反覆出現（超過一成的照片）、而且那些候選多半是
/// 靠補償相機位移才亮起來的，就是釘在畫面上的東西。
///
/// 相機沒動的連拍（鳥在巢洞口、同一處動了幾十張）放大倍數都是 1，不會被誤殺
pub fn drop_overlays(cands: &mut [Vec<Found>]) {
    let n = cands.len();
    let need = ((n as f32 * OVERLAY_SHARE).ceil() as usize).max(4);
    let centre = |f: &Found| ((f.rect[0] + f.rect[2]) / 2.0, (f.rect[1] + f.rect[3]) / 2.0);
    let mut kill: Vec<Vec<bool>> = cands.iter().map(|c| vec![false; c.len()]).collect();
    for i in 0..n {
        for (k, c) in cands[i].iter().enumerate() {
            let (cx, cy) = centre(c);
            // 其他照片在同一個位置的候選：數有幾張照片有、收集它們的放大倍數
            let mut frames = 0usize;
            let mut gains = vec![c.shift_gain];
            for (j, other) in cands.iter().enumerate() {
                if j == i {
                    continue;
                }
                let near: Vec<f32> = other
                    .iter()
                    .filter(|o| {
                        let (ox, oy) = centre(o);
                        (ox - cx).abs() < OVERLAY_RADIUS && (oy - cy).abs() < OVERLAY_RADIUS
                    })
                    .map(|o| o.shift_gain)
                    .collect();
                if !near.is_empty() {
                    frames += 1;
                    gains.extend(near);
                }
            }
            if frames + 1 < need {
                continue;
            }
            gains.sort_by(f32::total_cmp);
            if gains[gains.len() / 2] >= OVERLAY_GAIN {
                kill[i][k] = true;
            }
        }
    }
    for (c, k) in cands.iter_mut().zip(kill) {
        let mut it = k.into_iter();
        c.retain(|_| !it.next().unwrap_or(false));
    }
}

/// 只要最強的那一個候選（單張使用時的簡便版）
pub fn detect(prev: Option<&Frame>, cur: &Frame, next: Option<&Frame>) -> Option<Found> {
    detect_candidates(prev, cur, next, 1).into_iter().next()
}

/// 候選本身的不確定度在總代價裡佔多少。位置的單位是「畫面的幾分之幾」，
/// 所以 0.15 代表「分數差一整級」約等於「位置差 15% 畫面」
const PATH_SCORE_W: f32 = 0.15;

/// 頭一步跳超過畫面的這個比例，才算「瞬移」（見 [`path_scores`] 的頭版小島）
const HEAD_TELEPORT: f32 = 0.15;

/// 單一步加速度最多算到這裡（佔畫面的比例）。
///
/// 偶爾一兩張偵測不到主體時，路徑只能暫時跳到別的東西上再跳回來；不設上限
/// 的話一進一出代價太大，整段乾脆都留在旁邊那團樹葉上更「便宜」（實測一批
/// 連拍，鳥在 14 張裡只有 3 張沒偵測到，整段 14 張就全被拉走）。跳出去的那
/// 一兩張之後會被頭的前後檢查抓出來，標成要檢查
const PATH_ACC_CAP: f32 = 0.12;

/// 「附近找不到清楚的頭」要罰多少（與 [`PATH_SCORE_W`] 同一把尺）。
///
/// 只看軌跡平不平順會被騙：鳥迎面振翅時，每張偵測到的那一團在翅膀與身體
/// 之間跳來跳去，旁邊一團被風吹著緩緩飄的樹葉反而平順得多，整段路徑就被
/// 拉到樹葉上（實測一批連拍，連續十幾張框在左邊的散景裡）。附近有沒有
/// 清楚的頭是另一條獨立的證據；主體不是鳥（全都找不到頭）時每個候選罰得
/// 一樣，不影響挑選
const PATH_HEAD_W: f32 = 0.15;

/// 「有在移動」這件事值多少折扣。
///
/// 沒有這一項的話，光看軌跡平不平順會**偏好原地不動的東西**：被風吹得
/// 微微顫動的葉子，加速度幾乎是 0，比真的在飛的鳥還「便宜」。但這個功能
/// 要跟的本來就是**快速移動的主體**，所以每一步依速度給折扣，讓「一路飛
/// 過去」的那條路徑贏過「在原地抖」的那條
const PATH_MOVE_W: f32 = 1.0;

/// 速度折扣的上限（每張移動畫面的幾分之幾）。再快也只給到這麼多，
/// 免得亂跳的路徑靠「跳很遠」賺折扣
const PATH_MOVE_CAP: f32 = 0.08;

/// 大小忽大忽小要罰多少（用面積比的對數，放大一倍與縮小一半罰得一樣重）。
/// 同一隻鳥的大小是漸變的，一下子胖三倍多半是換到別的東西上了
const PATH_SIZE_W: f32 = 0.05;

/// 相鄰兩張的外觀指紋低於這個相關係數，就當作「中途換人了」
const SIG_SWITCH: f32 = 0.35;

/// 從每張的候選裡挑出一條「自始至終都是同一個東西」的路徑，
/// 回傳每張選中的候選編號（沒有候選的那張是 None）。
///
/// 每張各自挑最亮的那一團是不夠的：畫面裡會動的東西不只主體，逐張各挑各的
/// 就會在牠們之間跳來跳去——那正是成品裡看到的「主體在抖」。這裡改成整批
/// 一起挑：用動態規劃找總代價最低的一條路徑，代價 = 每一步的**加速度**
/// （位置的二階差分）＋ 候選本身的不確定度。等速直線飛行的加速度是 0，
/// 所以真正的主體軌跡幾乎一定會勝出，偶爾更亮的葉子則會被那一下「急轉彎」
/// 的代價擋掉。
///
/// 狀態帶著「前一張選了哪個」才算得出加速度（二階的維特比）：
/// 候選 4 個時每張 16 個狀態、64 條轉移，上千張也是一瞬間
pub fn choose_path(cands: &[Vec<Found>]) -> Vec<Option<usize>> {
    let mut out = vec![None; cands.len()];
    // 只在「有候選」的那幾張之間接力（中間空的張數要算進加速度的間距）
    let live: Vec<usize> = (0..cands.len()).filter(|&i| !cands[i].is_empty()).collect();
    if live.is_empty() {
        return out;
    }
    if live.len() <= 2 {
        for &i in &live {
            out[i] = Some(0);
        }
        return out;
    }
    let centre = |i: usize, k: usize| track_pos(&cands[i][k]);
    let area = |i: usize, k: usize| -> f32 {
        let r = cands[i][k].rect;
        ((r[2] - r[0]) * (r[3] - r[1])).max(1e-6)
    };
    let unc = |i: usize, k: usize| {
        let c = &cands[i][k];
        (1.0 - c.score) * PATH_SCORE_W + (1.0 - (c.heads[0].2 / HEAD_FULL).min(1.0)) * PATH_HEAD_W
    };
    let pick = viterbi2(&live, |i| cands[i].len(), unc, |(ip, a), (iq, b), (ir, c)| {
        let (pa, pb, pc) = (centre(ip, a), centre(iq, b), centre(ir, c));
        let d2 = (ir - iq) as f32;
        // 有在移動的給折扣（否則原地顫動的葉子最便宜），
        // 大小忽大忽小的要罰（多半是換到別的東西上了）
        let speed = ((pc.0 - pb.0).hypot(pc.1 - pb.1) / d2).min(PATH_MOVE_CAP);
        let grow = (area(ir, c) / area(iq, b)).ln().abs().min(3.0);
        accel(pa, pb, pc, (iq - ip) as f32, d2).min(PATH_ACC_CAP) + unc(ir, c) + grow * PATH_SIZE_W
            - speed * PATH_MOVE_W
    });
    for (t, &i) in live.iter().enumerate() {
        out[i] = Some(pick[t]);
    }
    out
}

/// 依序在 pa、pb、pc 的三張（間隔 d1、d2 張）：照前兩張等速外推出來的位置，
/// 與實際位置差多少＝這一步的加速度。間距不等時（中間有空著的張數）按比例外推
fn accel(pa: (f32, f32), pb: (f32, f32), pc: (f32, f32), d1: f32, d2: f32) -> f32 {
    let v = ((pb.0 - pa.0) / d1, (pb.1 - pa.1) / d1);
    let pred = (pb.0 + v.0 * d2, pb.1 + v.1 * d2);
    (pc.0 - pred.0).hypot(pc.1 - pred.1)
}

/// 二階的維特比：`live` 裡的每一張各有 `count(i)` 個選項，挑總代價最低的一條，
/// 回傳每張選了第幾個（與 `live` 一一對應）。
///
/// `own(i, k)` 是選項本身的代價（只用在頭兩張），`step` 是連著三張選
/// (ip, a)、(iq, b)、(ir, c) 時最後這一步要付的代價（含 c 本身的）。
/// 狀態帶著「前一張選了哪個」才算得出加速度
fn viterbi2(
    live: &[usize],
    count: impl Fn(usize) -> usize,
    own: impl Fn(usize, usize) -> f32,
    step: impl Fn((usize, usize), (usize, usize), (usize, usize)) -> f32,
) -> Vec<usize> {
    if live.len() <= 2 {
        return live
            .iter()
            .map(|&i| (0..count(i)).min_by(|&a, &b| own(i, a).total_cmp(&own(i, b))).unwrap_or(0))
            .collect();
    }
    // best[(a, b)]＝「上上張選 a、上一張選 b」這條路走到這裡的最低總代價
    let (n0, n1) = (count(live[0]), count(live[1]));
    let mut best: Vec<f32> = Vec::with_capacity(n0 * n1);
    for a in 0..n0 {
        for b in 0..n1 {
            best.push(own(live[0], a) + own(live[1], b));
        }
    }
    let mut width = n1;
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(live.len());
    for t in 2..live.len() {
        let (ip, iq, ir) = (live[t - 2], live[t - 1], live[t]);
        let (np, nq, nr) = (count(ip), count(iq), count(ir));
        let mut next = vec![f32::MAX; nq * nr];
        let mut from = vec![0usize; nq * nr];
        for b in 0..nq {
            for c in 0..nr {
                let slot = b * nr + c;
                for a in 0..np {
                    let prev = best[a * width + b];
                    if prev == f32::MAX {
                        continue;
                    }
                    let total = prev + step((ip, a), (iq, b), (ir, c));
                    if total < next[slot] {
                        next[slot] = total;
                        from[slot] = a;
                    }
                }
            }
        }
        best = next;
        width = nr;
        back.push(from);
    }
    // 回溯：先找終點那一對，再一路往回走
    let (mut bi, mut bv) = (0usize, f32::MAX);
    for (i, v) in best.iter().enumerate() {
        if *v < bv {
            bv = *v;
            bi = i;
        }
    }
    let mut out = vec![0usize; live.len()];
    let (mut b, mut c) = (bi / width, bi % width);
    out[live.len() - 1] = c;
    out[live.len() - 2] = b;
    for t in (2..live.len()).rev() {
        let a = back[t - 2][b * count(live[t]) + c];
        out[t - 2] = a;
        c = b;
        b = a;
    }
    out
}

/// 每一團附近留幾處「像頭」的地方當備選（見 [`HeadMaps::find`]）
pub const HEAD_PEAKS: usize = 3;

/// 找到的鳥頭離畫面邊緣不到這個比例，就當作頭被切掉了（不算抓到）
pub const HEAD_EDGE: f32 = 0.04;

/// 每張最多拿幾處備選去挑路徑
const HEAD_POOL: usize = 6;

/// 不同團找到的頭相距這麼近（佔畫面的比例）就算同一處
const HEAD_SAME: f32 = 0.02;

/// 頭的強度到這裡就算十足像頭（挑路徑時的不確定度歸零）
const HEAD_FULL: f32 = 0.4;

/// 每張挑一處當鳥頭：回傳 (x, y, 有多像)，選中的那一團附近找不到頭就是 None。
///
/// 單張各挑最像的不夠：五色鳥張開的翅膀泛著藍綠光，對焦又清楚時會贏過頭
/// （實測一批連拍，就有一張的「頭」從前一張的位置跳出去、下一張又跳回來，
/// 落在翅膀上）；動態框挑中的那一團也不一定離頭最近，旁邊那一團找到的才是
/// 真的頭。所以把「選中那一團的搜尋範圍裡、所有候選找到的頭」都當備選，
/// 跟 [`choose_path`] 一樣整批挑一條加速度最小、又最像頭的路徑——真的頭
/// 跟著鳥平順移動，翅膀上的亮點忽左忽右
pub fn choose_heads(cands: &[Vec<Found>], picked: &[Option<usize>]) -> Vec<Option<(f32, f32, f32)>> {
    let n = cands.len();
    let edge = |v: f32| v > HEAD_EDGE && v < 1.0 - HEAD_EDGE;
    let pool: Vec<Vec<(f32, f32, f32)>> = (0..n)
        .map(|i| {
            let Some(k) = picked.get(i).copied().flatten() else { return Vec::new() };
            let r = cands[i][k].rect;
            let (cx, cy) = ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0);
            let ww = ((r[2] - r[0]) * HEAD_WIN).max(HEAD_WIN_MIN) / 2.0;
            let wh = ((r[3] - r[1]) * HEAD_WIN).max(HEAD_WIN_MIN) / 2.0;
            let mut v: Vec<(f32, f32, f32)> = Vec::new();
            for &(x, y, s) in cands[i].iter().flat_map(|c| c.heads.iter()) {
                // 貼在畫面邊緣的不算：多半被切掉一半（例如鳥飛過鏡頭上方時，
                // 畫面頂端只剩紅色的喉嚨）
                if s <= 0.0 || (x - cx).abs() > ww || (y - cy).abs() > wh || !edge(x) || !edge(y) {
                    continue;
                }
                match v.iter_mut().find(|h| (h.0 - x).hypot(h.1 - y) < HEAD_SAME) {
                    Some(h) if s > h.2 => *h = (x, y, s),
                    Some(_) => {}
                    None => v.push((x, y, s)),
                }
            }
            v.sort_by(|a, b| b.2.total_cmp(&a.2));
            v.truncate(HEAD_POOL);
            v
        })
        .collect();
    let live: Vec<usize> = (0..n).filter(|&i| !pool[i].is_empty()).collect();
    let pos = |i: usize, k: usize| (pool[i][k].0, pool[i][k].1);
    let own = |i: usize, k: usize| (1.0 - (pool[i][k].2 / HEAD_FULL).min(1.0)) * PATH_SCORE_W;
    let pick = viterbi2(&live, |i| pool[i].len(), own, |(ip, a), (iq, b), (ir, c)| {
        accel(pos(ip, a), pos(iq, b), pos(ir, c), (iq - ip) as f32, (ir - iq) as f32) + own(ir, c)
    });
    let mut out = vec![None; n];
    for (t, &i) in live.iter().enumerate() {
        out[i] = Some(pool[i][pick[t]]);
    }
    out
}

/// 量軌跡用的位置：附近找得到頭就用頭，否則用那一團的中心。
///
/// 鳥迎面振翅時，偵測到的那一團在翅膀與身體之間跳來跳去，但每一團找到的頭
/// 都在同一處——拿那一團的中心量，真的鳥反而顯得亂跳，輸給旁邊一團緩緩飄的
/// 樹葉（實測一批連拍，連續十幾張框在左邊的散景裡）
fn track_pos(c: &Found) -> (f32, f32) {
    if c.heads[0].2 > 0.0 {
        return (c.heads[0].0, c.heads[0].1);
    }
    let r = c.rect;
    ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)
}

/// 估「b 要平移多少才會對上 a」（回傳的是 a 的座標系裡的位移，該層的像素）。
/// 比的是逐點差值的平均——同一台相機連拍，兩張的曝光多半一樣，
/// 這裡要的也只是個大概
fn best_shift(a: &Frame, b: &Frame, r: i32) -> (i32, i32) {
    let (mut best, mut at) = (f32::MAX, (0i32, 0i32));
    for dy in -r..=r {
        for dx in -r..=r {
            let (mut sum, mut n) = (0.0f32, 0u32);
            let mut y = 0;
            while y < a.h {
                let sy = y as i32 + dy;
                if sy >= 0 && (sy as usize) < b.h {
                    let mut x = 0;
                    while x < a.w {
                        let sx = x as i32 + dx;
                        if sx >= 0 && (sx as usize) < b.w {
                            sum += (a.at(x, y) - b.at(sx as usize, sy as usize)).abs();
                            n += 1;
                        }
                        x += 2;
                    }
                }
                y += 2;
            }
            // 重疊太少的位移不算數（挪到只剩一角時平均差值當然低）
            if n as usize * 8 < a.w * a.h && n > 0 {
                continue;
            }
            if n > 0 {
                let avg = sum / n as f32;
                if avg < best {
                    best = avg;
                    at = (dx, dy);
                }
            }
        }
    }
    at
}

/// `a` 與「平移了 `shift` 的 `b`」的逐點差值；`b` 那邊落在畫面外就給 0
/// （沒有東西可比的地方不能算成「有變化」）
fn diff_at(a: &Frame, b: &Frame, shift: (i32, i32)) -> Frame {
    let mut px = vec![0.0f32; a.w * a.h];
    for y in 0..a.h {
        let sy = y as i32 + shift.1;
        if sy < 0 || sy as usize >= b.h {
            continue;
        }
        for x in 0..a.w {
            let sx = x as i32 + shift.0;
            if sx < 0 || sx as usize >= b.w {
                continue;
            }
            px[y * a.w + x] = (a.at(x, y) - b.at(sx as usize, sy as usize)).abs();
        }
    }
    Frame::new(a.w, a.h, px)
}

/// 3×3 方框模糊（就地）。差值圖常是一小塊一小塊的，糊一下才連得成一團
fn blur3(f: &mut Frame) {
    let src = f.px.clone();
    for y in 0..f.h {
        for x in 0..f.w {
            let (mut sum, mut n) = (0.0f32, 0.0f32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                    if sx >= 0 && sy >= 0 && (sx as usize) < f.w && (sy as usize) < f.h {
                        sum += src[sy as usize * f.w + sx as usize];
                        n += 1.0;
                    }
                }
            }
            f.px[y * f.w + x] = sum / n;
        }
    }
}

/// 長出來的一團：格子清單與外接框（x0, y0, x1, y1，含端點）
struct Blob {
    cells: Vec<usize>,
    bbox: (usize, usize, usize, usize),
}

/// 從 `seed` 這一格往外長，把連著的、值在 `limit` 以上的格子都收進來。
///
/// 走過的格子一律在 `used` 上記號（包含長太大而作廢的那些）：下一個候選就
/// 不會又從同一團裡挑一個峰值出來。長得比 `max_cells` 還大就當作「這不是
/// 一個主體」而回 None——那多半是整片曝光變了、或整張糊掉
fn grow(
    f: &Frame,
    seed: usize,
    limit: f32,
    max_cells: usize,
    used: &mut [bool],
) -> Option<Blob> {
    let mut cells = vec![seed];
    let mut stack = vec![seed];
    used[seed] = true;
    let (mut x0, mut y0) = (seed % f.w, seed / f.w);
    let (mut x1, mut y1) = (x0, y0);
    let mut too_big = false;
    while let Some(i) = stack.pop() {
        let (x, y) = (i % f.w, i / f.w);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
        // 太大就別再長了，但已經走過的格子仍留著記號
        if cells.len() > max_cells {
            too_big = true;
            continue;
        }
        let mut push = |nx: usize, ny: usize, stack: &mut Vec<usize>, cells: &mut Vec<usize>| {
            let j = ny * f.w + nx;
            if !used[j] && f.px[j] >= limit {
                used[j] = true;
                cells.push(j);
                stack.push(j);
            }
        };
        if x > 0 {
            push(x - 1, y, &mut stack, &mut cells);
        }
        if x + 1 < f.w {
            push(x + 1, y, &mut stack, &mut cells);
        }
        if y > 0 {
            push(x, y - 1, &mut stack, &mut cells);
        }
        if y + 1 < f.h {
            push(x, y + 1, &mut stack, &mut cells);
        }
    }
    (!too_big).then_some(Blob { cells, bbox: (x0, y0, x1, y1) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 畫一張「灰底＋一塊有花紋的方塊」的測試影像，方塊中心在 (cx, cy) 像素
    fn scene(w: usize, h: usize, cx: f32, cy: f32, half: f32) -> Frame {
        let mut px = vec![90.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                if dx.abs() <= half && dy.abs() <= half {
                    // 有花紋才追得動：純色方塊在自己內部處處都一樣像
                    px[y * w + x] = if (x / 3 + y / 2) % 2 == 0 { 235.0 } else { 30.0 };
                }
            }
        }
        Frame::new(w, h, px)
    }

    #[test]
    fn follows_a_fast_moving_subject() {
        let (w, h) = (320usize, 240usize);
        let half = 14.0;
        let first = scene(w, h, 60.0, 120.0, half);
        let rect = [
            (60.0 - half) / w as f32,
            (120.0 - half) / h as f32,
            (60.0 + half) / w as f32,
            (120.0 + half) / h as f32,
        ];
        let mut t = Tracker::new(&first, rect).expect("框夠大，應該起得了追蹤");
        // 每張移動 18px（比主體本身還大一截），中途再讓整張變亮一級
        for k in 1..=6 {
            let (cx, cy) = (60.0 + 18.0 * k as f32, 120.0 + 6.0 * k as f32);
            let mut f = scene(w, h, cx, cy, half);
            if k >= 3 {
                for p in f.px.iter_mut() {
                    *p = (*p * 1.15 + 12.0).min(255.0);
                }
            }
            let hit = t.find(&f);
            assert!(hit.locked, "第 {k} 張應該對得上（分數 {}）", hit.score);
            let (gx, gy) = (cx / w as f32, cy / h as f32);
            assert!(
                (hit.cx - gx).abs() < 0.02 && (hit.cy - gy).abs() < 0.02,
                "第 {k} 張位置差太多：{:?} 應該接近 ({gx}, {gy})",
                (hit.cx, hit.cy)
            );
        }
    }

    #[test]
    fn coasts_along_when_the_subject_vanishes() {
        let (w, h) = (320usize, 240usize);
        let first = scene(w, h, 60.0, 120.0, 14.0);
        let rect = [46.0 / 320.0, 106.0 / 240.0, 74.0 / 320.0, 134.0 / 240.0];
        let mut t = Tracker::new(&first, rect).unwrap();
        // 先跟兩張建立速度，再餵一張完全沒有主體的空景
        t.find(&scene(w, h, 80.0, 120.0, 14.0));
        let before = t.find(&scene(w, h, 100.0, 120.0, 14.0));
        assert!(before.locked);
        let blank = Frame::new(w, h, vec![90.0; w * h]);
        let hit = t.find(&blank);
        assert!(!hit.locked, "沒有主體卻宣稱對上了");
        // 沒對上就照原方向滑過去，不會跳回原地也不會亂飛
        assert!(hit.cx > before.cx && hit.cx < before.cx + 0.15);
    }

    #[test]
    fn tiny_box_is_rejected() {
        let f = scene(320, 240, 60.0, 120.0, 14.0);
        assert!(Tracker::new(&f, [0.5, 0.5, 0.501, 0.501]).is_none());
    }

    #[test]
    fn smoothing_removes_jitter_without_lagging() {
        // 等速前進＋逐張左右抖一格
        let mut pts: Vec<(f32, f32)> = (0..40)
            .map(|i| {
                let j = if i % 2 == 0 { 0.01 } else { -0.01 };
                (0.1 + i as f32 * 0.01 + j, 0.5)
            })
            .collect();
        let raw = pts.clone();
        smooth(&mut pts, 0.7);
        // 抖動磨掉了：相鄰兩張的位移不再忽正忽負
        let flips = pts.windows(2).filter(|w| w[1].0 <= w[0].0).count();
        assert_eq!(flips, 0, "平滑後仍在來回抖：{pts:?}");
        // 也沒有整條落後（零相位）：中段與原始的等速線幾乎重合
        for i in 10..30 {
            let straight = raw[i].0 - if i % 2 == 0 { 0.01 } else { -0.01 };
            assert!((pts[i].0 - straight).abs() < 0.004);
        }
    }

    /// 一張連拍：背景是有紋理的樹葉（相機每張往右挪 `pan` 個像素），
    /// 主體是一塊飛到 `bx` 的方塊
    fn burst(pan: i32, bx: f32) -> Frame {
        let (w, h) = (320usize, 240usize);
        let mut px = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as i32 + pan, y as i32);
                px[y * w + x] = 60.0
                    + 25.0 * (((u * 7 + v * 13) % 23) as f32 / 23.0)
                    + 12.0 * (((u / 5 + v / 3) % 3) as f32);
                if (x as f32 - bx).abs() <= 11.0 && (y as f32 - 120.0).abs() <= 8.0 {
                    px[y * w + x] = if (x / 2 + y / 2) % 2 == 0 { 240.0 } else { 20.0 };
                }
            }
        }
        Frame::new(w, h, px)
    }

    /// 有背景紋理、相機還在平移的一組連拍：自動框選要框到那隻在飛的東西，
    /// 而不是被整片背景的位移騙走
    #[test]
    fn detect_finds_the_thing_that_moves_by_itself() {
        let (a, b, c) = (burst(0, 90.0), burst(3, 112.0), burst(6, 134.0));
        let f = detect(Some(&a), &b, Some(&c)).expect("應該要框到飛過去的主體");
        let (cx, cy) = ((f.rect[0] + f.rect[2]) / 2.0, (f.rect[1] + f.rect[3]) / 2.0);
        assert!(
            (cx - 112.0 / 320.0).abs() < 0.05 && (cy - 0.5).abs() < 0.05,
            "框到別的地方去了：{:?}",
            f.rect
        );
        // 框大概是主體那麼大，不是整片背景
        let area = (f.rect[2] - f.rect[0]) * (f.rect[3] - f.rect[1]);
        assert!(area < 0.15, "框太大了（{area}）：{:?}", f.rect);
        assert!(f.score > 0.3, "應該要有點把握：{}", f.score);
    }

    /// 整批都是同一個畫面（沒有東西在動）時不能亂框一個出來——
    /// 那種照片就是該讓使用者刪掉的
    #[test]
    fn detect_gives_up_when_nothing_moves() {
        let still = scene(320, 240, 160.0, 120.0, 20.0);
        assert!(detect(Some(&still), &still, Some(&still)).is_none());
    }

    /// 第一張（與最後一張）沒有「另一邊」可比，就拿**同一側的兩張**來比：
    /// 主體在這張的位置對兩張都不一樣，牠在別張留下的殘影卻只對其中一張
    /// 不一樣，取小值就只剩本尊。只給一張的話兩者一樣亮，框到哪個純屬運氣
    #[test]
    fn detect_uses_two_neighbours_on_the_same_side_for_the_first_photo() {
        let (cur, n1, n2) = (burst(0, 90.0), burst(3, 112.0), burst(6, 134.0));
        let f = detect(Some(&n1), &cur, Some(&n2)).expect("第一張也該框得到");
        let cx = (f.rect[0] + f.rect[2]) / 2.0;
        assert!(
            (cx - 90.0 / 320.0).abs() < 0.06,
            "框到殘影去了（應該在 {:.3}）：{:?}",
            90.0 / 320.0,
            f.rect
        );
    }

    /// 測試用的候選：`look` 不同就代表「長得不一樣的東西」
    fn cand(cx: f32, cy: f32, score: f32, look: f32) -> Found {
        let mut sig = [0.0f32; SIG_N * SIG_N];
        for (i, v) in sig.iter_mut().enumerate() {
            *v = (i as f32 * 0.7 + look * 2.3).sin();
        }
        let mean = sig.iter().sum::<f32>() / sig.len() as f32;
        let norm = sig.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>().sqrt();
        for v in sig.iter_mut() {
            *v = (*v - mean) / norm;
        }
        Found { rect: [cx - 0.04, cy - 0.04, cx + 0.04, cy + 0.04], score, sig, shift_gain: 1.0, heads: [(0.0, 0.0, 0.0); HEAD_PEAKS], reds: [None; HEAD_PEAKS], sharp: 0.0 }
    }

    /// 畫面裡不只主體在動：整批一起挑路徑時，要挑那條**連貫**的，
    /// 而不是每張各自最亮的那一團——後者正是成品裡「主體在抖」的來源
    #[test]
    fn path_follows_the_consistent_subject_not_the_loudest_blob() {
        let at = |cx: f32, cy: f32, score: f32| cand(cx, cy, score, 0.0);
        // 候選 0（分數最高）每張亂跳；候選 1 是等速往右下飛的真主體
        let cands: Vec<Vec<Found>> = (0..12)
            .map(|i| {
                let t = i as f32;
                let jump = if i % 2 == 0 { 0.85 } else { 0.15 };
                vec![
                    at(jump, 0.2 + 0.05 * (i % 3) as f32, 0.98),
                    at(0.12 + 0.05 * t, 0.30 + 0.03 * t, 0.80),
                ]
            })
            .collect();
        let picked = choose_path(&cands);
        for (i, k) in picked.iter().enumerate() {
            assert_eq!(*k, Some(1), "第 {i} 張挑到亂跳的那一團了");
        }
    }

    /// 沒有候選的那幾張要留空（呼叫端會用前後內插補），不能亂指一個
    #[test]
    fn path_leaves_empty_frames_alone() {
        let at = |cx: f32| cand(cx, 0.5, 0.9, 0.0);
        let cands = vec![
            vec![at(0.10)],
            Vec::new(),
            vec![at(0.20)],
            vec![at(0.25)],
            Vec::new(),
        ];
        let picked = choose_path(&cands);
        assert_eq!(picked[1], None);
        assert_eq!(picked[4], None);
        assert_eq!(picked[0], Some(0));
        assert_eq!(picked[3], Some(0));
    }

    /// 主體停下來不動時，差值圖看不見牠，路徑會平順地滑到旁邊的樹葉上——
    /// 位置接得上、差值也很亮，只有「長得不像」會露餡。那幾張要被標成
    /// 沒把握，後面才會拿它們去補框、也才會請使用者過目
    #[test]
    fn path_scores_flag_the_stretch_that_changed_identity() {
        // 0~5 與 10~15 是主體（look 0），6~9 換成長得完全不同的東西（look 1）
        let cands: Vec<Vec<Found>> = (0..16)
            .map(|i| {
                let t = i as f32;
                let look = if (6..10).contains(&i) { 1.0 } else { 0.0 };
                vec![cand(0.15 + 0.03 * t, 0.4 + 0.01 * t, 0.95, look)]
            })
            .collect();
        let picked: Vec<Option<usize>> = (0..16).map(|_| Some(0)).collect();
        let s = path_scores(&cands, &picked, None);
        for i in 6..10 {
            assert!(s[i] < 0.4, "第 {i} 張換了東西卻還很有把握（{}）", s[i]);
        }
        for i in [1usize, 3, 12, 14] {
            assert!(s[i] > 0.6, "第 {i} 張是主體本人，不該被壓分（{}）", s[i]);
        }
    }

    /// 單張暴衝的離群值要被前後夾掉，連續的移動則原樣留著
    #[test]
    fn despike_drops_the_single_frame_jump() {
        let mut pts = vec![
            (0.10, 0.5),
            (0.15, 0.5),
            (0.80, 0.5), // 這張認錯了東西
            (0.25, 0.5),
            (0.30, 0.5),
        ];
        despike(&mut pts);
        assert!((pts[2].0 - 0.25).abs() < 1e-6, "離群值沒被夾掉：{:?}", pts[2]);
        // 頭尾不動、連續的移動也不該被改
        assert_eq!(pts[0], (0.10, 0.5));
        assert_eq!(pts[4], (0.30, 0.5));
        assert!((pts[1].0 - 0.15).abs() < 1e-6);
    }

    /// 浮水印：相機在移動時，固定在畫面同一處、靠補償相機位移才亮起來的
    /// 東西要整批剔除；真正在飛的鳥留著。相機沒動的連拍（鳥在巢洞口同一處
    /// 動了幾十張，放大倍數恆為 1）不能被誤殺
    #[test]
    fn overlays_fixed_on_the_frame_are_dropped() {
        let with_gain = |cx: f32, cy: f32, g: f32| {
            let mut f = cand(cx, cy, 0.95, 0.0);
            f.shift_gain = g;
            f
        };
        let mut cands: Vec<Vec<Found>> = (0..20)
            .map(|i| {
                vec![
                    with_gain(0.90, 0.95, 6.0),                     // 右下角的簽名
                    with_gain(0.10 + 0.03 * i as f32, 0.50, 1.5), // 飛過去的鳥
                ]
            })
            .collect();
        drop_overlays(&mut cands);
        for (i, c) in cands.iter().enumerate() {
            assert_eq!(c.len(), 1, "第 {i} 張還留著浮水印：{c:?}");
            assert!((c[0].rect[1] + c[0].rect[3]) / 2.0 < 0.6, "第 {i} 張把鳥剔掉了");
        }

        // 相機沒動：同一處反覆出現、但放大倍數是 1——那是停在巢洞口的鳥
        let mut nest: Vec<Vec<Found>> = (0..20).map(|_| vec![with_gain(0.80, 0.48, 1.0)]).collect();
        drop_overlays(&mut nest);
        assert!(nest.iter().all(|c| c.len() == 1), "相機沒動時不該剔除任何東西");
    }

    /// 單張最像頭的不一定是頭：中間那張翅膀上的亮點比頭還強，但它一下跳出去、
    /// 下一張又跳回來；頭跟著鳥平順移動，整批挑的時候要挑頭。
    /// 另一團（不是路徑選中的那一團）找到的頭也要拿來當備選
    #[test]
    fn heads_follow_the_smooth_path_not_the_brightest_spot() {
        let n = 7;
        let mut cands: Vec<Vec<Found>> = Vec::new();
        for i in 0..n {
            let hx = 0.3 + 0.02 * i as f32;
            let mut body = cand(hx + 0.05, 0.5, 0.9, 0.0);
            body.heads[0] = (hx, 0.45, 0.30);
            if i == 3 {
                // 翅膀上的亮點比頭強，頭只排第二
                body.heads[0] = (hx + 0.09, 0.53, 0.40);
                body.heads[1] = (hx, 0.45, 0.25);
            }
            let mut wing = cand(hx + 0.07, 0.55, 0.5, 1.0);
            if i == 5 {
                // 選中那一團這張沒找到頭，旁邊那一團找到了
                body.heads[0] = (0.0, 0.0, 0.0);
                wing.heads[0] = (hx, 0.45, 0.28);
            }
            cands.push(vec![body, wing]);
        }
        let picked = vec![Some(0); n];
        let heads = choose_heads(&cands, &picked);
        for (i, h) in heads.iter().enumerate() {
            let (x, y, _) = h.expect("每張都該找得到頭");
            let hx = 0.3 + 0.02 * i as f32;
            assert!((x - hx).abs() < 1e-4 && (y - 0.45).abs() < 1e-4, "第 {i} 張挑到 ({x}, {y})");
        }
    }

    /// 被兩次「瞬移」夾在中間的小段要被標成沒把握：主體不會一下子跳到畫面
    /// 另一頭、過幾張又跳回來（實測：鳥飛出畫面後路徑跳到一片葉子上跟了七張）
    #[test]
    fn a_short_stretch_between_two_jumps_is_doubted() {
        let cands: Vec<Vec<Found>> = (0..60)
            .map(|i| {
                let t = i as f32;
                let (x, y) = if (30..36).contains(&i) {
                    (0.10 + 0.03 * (t - 30.0), 0.15) // 跳到左上角的葉子
                } else {
                    (0.30 + 0.008 * t, 0.50 + 0.003 * t) // 主體的主線
                };
                vec![cand(x, y, 0.99, 0.0)]
            })
            .collect();
        let picked: Vec<Option<usize>> = (0..60).map(|_| Some(0)).collect();
        let s = path_scores(&cands, &picked, None);
        for i in 30..36 {
            assert!(s[i] < 0.45, "第 {i} 張在兩次瞬移之間，卻還很有把握（{}）", s[i]);
        }
        for i in [5usize, 15, 45, 55] {
            assert!(s[i] > 0.6, "第 {i} 張是主線上的，不該被壓分（{}）", s[i]);
        }
    }

    /// 追蹤器的 `same` 永遠拿**出發時的樣子**來比：主體不見了、換成別的
    /// 東西時，不管樣板一路混成什麼樣子，`same` 都要說「不像」
    #[test]
    fn same_compares_against_the_original_look() {
        let (w, h) = (320usize, 240usize);
        let first = scene(w, h, 120.0, 120.0, 14.0);
        let rect = [106.0 / 320.0, 106.0 / 240.0, 134.0 / 320.0, 134.0 / 240.0];
        let mut t = Tracker::new(&first, rect).unwrap();
        let hit = t.find(&scene(w, h, 125.0, 120.0, 14.0));
        assert!(hit.same > 0.9, "同一個東西應該很像（{}）", hit.same);
        // 主體離開，原地換成一塊花紋完全不同的東西，連續好幾張
        let other = |cx: f32| {
            let mut px = vec![90.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    if (x as f32 - cx).abs() <= 14.0 && (y as f32 - 120.0).abs() <= 14.0 {
                        px[y * w + x] = if (x / 7 + y / 9) % 2 == 0 { 30.0 } else { 220.0 };
                    }
                }
            }
            Frame::new(w, h, px)
        };
        for k in 0..6 {
            let hit = t.find(&other(130.0 + k as f32));
            assert!(hit.same < 0.6, "換成別的東西了，same 卻說很像（第 {k} 張 {}）", hit.same);
        }
    }

    /// 找鳥頭：要挑「顏色很多樣、又跟著鳥一起在動」的那一塊。
    /// 顏色單一的翅膀（動得再兇也不是頭）、顏色很雜但只是微微晃的葉子
    /// （樹叢裡透光的那種）都不能被當成頭
    /// 範本取自一張照片的鳥頭，換到下一張（鳥移了位置、左右翻了面）也要找得到，
    /// 而且回傳的是範本標記點（正中央）對應的位置
    #[test]
    fn library_match_finds_the_marked_spot_in_another_photo() {
        let head = |img: &mut RgbImage, x0: i64, y0: i64, flip: bool| {
            for dy in 0..30i64 {
                for dx in 0..30i64 {
                    let ex = if flip { 29 - dx } else { dx };
                    let c = match (ex / 10, dy / 10) {
                        (0, _) => [40, 90, 200],
                        (1, 0) => [230, 200, 40],
                        (1, _) => [220, 40, 40],
                        _ => [20, 20, 20],
                    };
                    img.put_pixel((x0 + dx) as u32, (y0 + dy) as u32, image::Rgb(c));
                }
            }
        };
        let mut a = RgbImage::from_pixel(200, 150, image::Rgb([70, 130, 60]));
        head(&mut a, 40, 40, false);
        let mut b = RgbImage::from_pixel(200, 150, image::Rgb([70, 130, 60]));
        head(&mut b, 110, 70, true);
        let (fa, fb) = (Frame::from_rgb(&a), Frame::from_rgb(&b));
        // 標記點在頭的正中央 (55, 55)
        let lib: Vec<HeadLook> = [false, true].iter().filter_map(|&m| fa.head_look_m(55.0, 55.0, 30.0, m)).collect();
        // 從偏了一段的地方出發
        let (x, y, _, _) = match_library(&fb, 115.0, 95.0, &[24.0, 30.0, 36.0], &lib, &|_| false).expect("應該找得到");
        assert!((x - 125.0).abs() <= 3.0 && (y - 85.0).abs() <= 3.0, "找到 ({x}, {y})");
    }

    #[test]
    fn red_spots_beside_the_beak_pull_the_head_to_between_the_eyes() {
        // 原解析度 1000×500：正面的鳥，嘴基兩側各一個紅點；下方遠一點有一圈
        // 紅色的胸口，不能被它帶走
        let (rw, rh) = (1000usize, 500usize);
        let mut img = RgbImage::from_pixel(rw as u32, rh as u32, image::Rgb([60, 140, 40]));
        for (sx, sy) in [(480u32, 200u32), (510, 200)] {
            for dy in 0..6 {
                for dx in 0..6 {
                    img.put_pixel(sx + dx, sy + dy, image::Rgb([210, 40, 40]));
                }
            }
        }
        for x in 460..540u32 {
            for y in 290..300u32 {
                img.put_pixel(x, y, image::Rgb([200, 50, 45]));
            }
        }
        let c = Chroma::from_rgb(&img);
        // 「顏色最雜」找到的頭偏在右上方一點
        let (x, y) = red_spot(&c, 515.0 / rw as f32, 185.0 / rh as f32).expect("應該找得到紅點");
        let (px, py) = (x * rw as f32, y * rh as f32);
        assert!((px - 498.0).abs() < 5.0 && (py - 203.0).abs() < 5.0, "紅點重心 ({px}, {py})");
        // 附近沒有紅點：不動
        assert!(red_spot(&c, 100.0 / rw as f32, 100.0 / rh as f32).is_none());
        // 側面：只有脖子上一塊紅斑，湊不成一對，不動
        let mut side = RgbImage::from_pixel(rw as u32, rh as u32, image::Rgb([60, 140, 40]));
        for x in 300..330u32 {
            for y in 300..315u32 {
                side.put_pixel(x, y, image::Rgb([220, 40, 40]));
            }
        }
        assert!(red_spot(&Chroma::from_rgb(&side), 320.0 / rw as f32, 280.0 / rh as f32).is_none());
    }

    #[test]
    fn head_is_the_colourful_spot_that_moves_with_the_bird() {
        let (w, h) = (96usize, 64usize);
        let mut rg = vec![0.0f32; w * h];
        let mut yb = vec![0.0f32; w * h];
        let mut motion = vec![1.0f32; w * h];
        let put = |x0: usize, y0: usize, f: &mut dyn FnMut(usize, usize)| {
            for y in y0..y0 + 10 {
                for x in x0..x0 + 10 {
                    f(x, y);
                }
            }
        };
        // 鳥頭：藍、黃、紅、黑擠在一起，跟著鳥移動
        put(20, 20, &mut |x, y| {
            let k = y * w + x;
            rg[k] = if (x + y) % 2 == 0 { 90.0 } else { -60.0 };
            yb[k] = if (x / 2 + y) % 2 == 0 { 80.0 } else { -90.0 };
            motion[k] = 40.0;
        });
        // 翅膀：顏色單一，但拍得比頭還兇
        put(45, 20, &mut |x, y| {
            let k = y * w + x;
            rg[k] = -30.0;
            yb[k] = -10.0;
            motion[k] = 70.0;
        });
        // 葉子縫透光：顏色一樣雜，但只是微微晃
        put(70, 20, &mut |x, y| {
            let k = y * w + x;
            rg[k] = if (x + y) % 2 == 0 { 90.0 } else { -60.0 };
            yb[k] = if (x / 2 + y) % 2 == 0 { 80.0 } else { -90.0 };
            motion[k] = 6.0;
        });
        // 糊掉的「頭」：顏色雜、也在動，但沒對焦（散景、或拍動中糊掉的那一塊）
        put(20, 42, &mut |x, y| {
            let k = y * w + x;
            rg[k] = if (x + y) % 2 == 0 { 90.0 } else { -60.0 };
            yb[k] = if (x / 2 + y) % 2 == 0 { 80.0 } else { -90.0 };
            motion[k] = 40.0;
        });
        // 清晰度：只有真的鳥頭那一塊對焦清楚。糊掉的那塊對比很強（散景裡
        // 天空透過樹葉的亮邊），最細的邊緣強度跟頭一樣，但粗輪廓強得多
        let mut sharp = vec![2.0f32; w * h];
        let mut coarse = vec![10.0f32; w * h];
        for y in 20..30 {
            for x in 20..30 {
                sharp[y * w + x] = 20.0;
            }
        }
        for y in 42..52 {
            for x in 20..30 {
                sharp[y * w + x] = 20.0;
                coarse[y * w + x] = 200.0;
            }
        }
        let c = Chroma { w, h, rg, yb, red: Vec::new() };
        let m = Frame::new(w, h, motion);
        let maps = HeadMaps::new(&c, &m, &Frame::new(w, h, sharp), &Frame::new(w, h, coarse));
        // 動態框落在翅膀上，鳥身的平均移動量 40
        let rect = [45.0 / 96.0, 20.0 / 64.0, 55.0 / 96.0, 30.0 / 64.0];
        let (hx, hy, s) = maps.find(rect, 40.0)[0];
        assert!(s > 0.0, "應該找得到頭");
        let (px, py) = (hx * w as f32, hy * h as f32);
        assert!(
            (20.0..30.0).contains(&px) && (20.0..30.0).contains(&py),
            "頭找錯地方了：({px:.1}, {py:.1})"
        );
    }

    #[test]
    fn smoothing_at_zero_keeps_the_raw_track() {
        let mut pts = vec![(0.1, 0.2), (0.5, 0.6), (0.3, 0.4)];
        let raw = pts.clone();
        smooth(&mut pts, 0.0);
        assert_eq!(pts, raw);
    }
}
