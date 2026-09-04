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
}

impl Frame {
    /// 從彩色影像轉灰階（BT.601，與 ffmpeg 在 yuv420p 上用的同一組係數）
    pub fn from_rgb(img: &RgbImage) -> Frame {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let px = img
            .pixels()
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect();
        Frame { w, h, px }
    }

    /// 直接以灰階像素建一張（切塊與測試用）
    fn new(w: usize, h: usize, px: Vec<f32>) -> Frame {
        debug_assert_eq!(px.len(), w * h);
        Frame { w, h, px }
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
}

/// 一條追蹤的狀態。從使用者框的那一張出發，往後（或往前）一張一張餵進去
pub struct Tracker {
    /// 樣板，以及它取自金字塔的哪一層
    tmpl: Frame,
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
        Hit { cx: centre.0, cy: centre.1, score: best.2, locked }
    }
}

/// 挑完路徑後，重新評估每一張「有多可信」（0~1）。
///
/// 光看那一團差值有多亮是不夠的——被風吹的葉子每一張都很亮、分數都接近滿分，
/// 使用者卻一張都不會想留。真正有意義的問題是「**這一張接得上前後嗎**」：
/// 用前後兩張等速外推出來的位置，與這張選中的位置差多少，除以整段的典型
/// 步幅。接得上的照原本的分數，接不上的往下壓，於是「該檢查的那幾張」就
/// 自己浮出來了（也正是自動補框要優先處理的那些）
pub fn path_scores(cands: &[Vec<Found>], picked: &[Option<usize>]) -> Vec<f32> {
    let n = picked.len();
    let mut out = vec![0.0f32; n];
    let pos: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| {
            picked[i].map(|k| {
                let r = cands[i][k].rect;
                ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)
            })
        })
        .collect();
    // 整段的典型步幅：拿它當尺，快飛的連拍與慢慢晃的連拍才用同一套標準
    let mut steps: Vec<f32> = Vec::new();
    for i in 1..n {
        if let (Some(a), Some(b)) = (pos[i - 1], pos[i]) {
            steps.push((b.0 - a.0).hypot(b.1 - a.1));
        }
    }
    steps.sort_by(f32::total_cmp);
    let typical = steps.get(steps.len() / 2).copied().unwrap_or(0.02).max(0.01);

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
    {
        let mut breaks: Vec<usize> = vec![0];
        for i in 1..n {
            if speeds[i] > (typical * 4.0).max(0.05) {
                breaks.push(i);
            }
        }
        breaks.push(n);
        for w in breaks.windows(2) {
            let (a, b) = (w[0], w[1]);
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
        let base = cands[i][k].score * if still[i] { 0.3 } else { 1.0 };
        // 前後都在才算得出「接不接得上」；在邊界就只看那一團自己的分數
        let dev = match (i.checked_sub(1).and_then(|j| pos[j]), pos[i], pos.get(i + 1).copied().flatten())
        {
            (Some(a), Some(b), Some(c)) => {
                let mid = ((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0);
                (b.0 - mid.0).hypot(b.1 - mid.1)
            }
            _ => 0.0,
        };
        // 差一個典型步幅還算正常，差三四個就幾乎不可信了
        let fit = 1.0 / (1.0 + (dev / (typical * 1.5)).powi(2));
        // 與「主體的樣子」像不像：完全不像的直接壓到要檢查的程度
        let look = ((sig_corr(&cands[i][k].sig, &refsig) + 1.0) / 2.0).clamp(0.0, 1.0);
        let look = (look * 1.6).min(1.0);
        out[i] = (base * fit * look).clamp(0.0, 1.0);
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
    }
    let mut m = match maps.len() {
        0 => return Vec::new(),
        1 => maps.pop().unwrap(),
        _ => {
            // 逐點取小值：只有「前後兩張都覺得這裡不一樣」的地方留得下來
            let (a, b) = (&maps[0], &maps[1]);
            let px = a.px.iter().zip(&b.px).map(|(x, y)| x.min(*y)).collect();
            Frame::new(a.w, a.h, px)
        }
    };
    // 單邊（頭尾兩張）少了一層把關，門檻抬高一點免得框到雜訊
    let strict = if maps.len() < 2 { 1.6 } else { 1.0 };
    blur3(&mut m);

    // 「典型差值」用中位數：會動的東西只佔一小塊，中位數代表的就是背景的殘差
    let mut sorted: Vec<f32> = m.px.clone();
    sorted.sort_by(f32::total_cmp);
    let typical = sorted[sorted.len() / 2].max(0.5);
    let floor = (DET_PEAK_MIN * strict).max(typical * DET_PEAK_OVER * strict);
    let level = (m.px.len() as f32 * DET_MAX_AREA) as usize;
    let (fw, fh) = (m.w as f32, m.h as f32);

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
        out.push(Found {
            rect,
            score: (contrast * 0.65 + compact * 0.35).clamp(0.0, 1.0),
            sig,
        });
    }
    out
}

/// 只要最強的那一個候選（單張使用時的簡便版）
pub fn detect(prev: Option<&Frame>, cur: &Frame, next: Option<&Frame>) -> Option<Found> {
    detect_candidates(prev, cur, next, 1).into_iter().next()
}

/// 候選本身的不確定度在總代價裡佔多少。位置的單位是「畫面的幾分之幾」，
/// 所以 0.15 代表「分數差一整級」約等於「位置差 15% 畫面」
const PATH_SCORE_W: f32 = 0.15;

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
    let centre = |i: usize, k: usize| -> (f32, f32) {
        let r = cands[i][k].rect;
        ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)
    };
    let area = |i: usize, k: usize| -> f32 {
        let r = cands[i][k].rect;
        ((r[2] - r[0]) * (r[3] - r[1])).max(1e-6)
    };
    let unc = |i: usize, k: usize| (1.0 - cands[i][k].score) * PATH_SCORE_W;

    // best[(a, b)]＝「上上張選 a、上一張選 b」這條路走到這裡的最低總代價
    let (n0, n1) = (cands[live[0]].len(), cands[live[1]].len());
    let mut best: Vec<f32> = Vec::with_capacity(n0 * n1);
    for a in 0..n0 {
        for b in 0..n1 {
            best.push(unc(live[0], a) + unc(live[1], b));
        }
    }
    let mut width = n1;
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(live.len());
    for t in 2..live.len() {
        let (ip, iq, ir) = (live[t - 2], live[t - 1], live[t]);
        let (np, nq, nr) = (cands[ip].len(), cands[iq].len(), cands[ir].len());
        // 間距不等時（中間有沒候選的張數）按比例外推
        let (d1, d2) = ((iq - ip) as f32, (ir - iq) as f32);
        let mut next = vec![f32::MAX; nq * nr];
        let mut from = vec![0usize; nq * nr];
        for b in 0..nq {
            let pb = centre(iq, b);
            for c in 0..nr {
                let pc = centre(ir, c);
                let cost_c = unc(ir, c);
                let slot = b * nr + c;
                for a in 0..np {
                    let prev = best[a * width + b];
                    if prev == f32::MAX {
                        continue;
                    }
                    let pa = centre(ip, a);
                    // 等速外推出來的位置與實際位置差多少＝這一步的加速度
                    let v = ((pb.0 - pa.0) / d1, (pb.1 - pa.1) / d1);
                    let pred = (pb.0 + v.0 * d2, pb.1 + v.1 * d2);
                    let acc = (pc.0 - pred.0).hypot(pc.1 - pred.1);
                    // 有在移動的給折扣（否則原地顫動的葉子最便宜），
                    // 大小忽大忽小的要罰（多半是換到別的東西上了）
                    let speed = ((pc.0 - pb.0).hypot(pc.1 - pb.1) / d2).min(PATH_MOVE_CAP);
                    let grow = (area(ir, c) / area(iq, b)).ln().abs().min(3.0);
                    let total =
                        prev + acc + cost_c + grow * PATH_SIZE_W - speed * PATH_MOVE_W;
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
    let (mut b, mut c) = (bi / width, bi % width);
    out[live[live.len() - 1]] = Some(c);
    out[live[live.len() - 2]] = Some(b);
    for t in (2..live.len()).rev() {
        let from = &back[t - 2];
        let nr = cands[live[t]].len();
        let a = from[b * nr + c];
        out[live[t - 2]] = Some(a);
        c = b;
        b = a;
    }
    out
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
        Found { rect: [cx - 0.04, cy - 0.04, cx + 0.04, cy + 0.04], score, sig }
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
        let s = path_scores(&cands, &picked);
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

    #[test]
    fn smoothing_at_zero_keeps_the_raw_track() {
        let mut pts = vec![(0.1, 0.2), (0.5, 0.6), (0.3, 0.4)];
        let raw = pts.clone();
        smooth(&mut pts, 0.0);
        assert_eq!(pts, raw);
    }
}
