//! 「優化影像」模組的影像分析與主體強化。
//!
//! 兩件事分開做，理由與去煙霧那邊一樣是「重算成本差了一個數量級」：
//!
//! 1. [`analyze`]：**量一張照片該怎麼調色**，回推那十二條滑桿該停在哪裡。
//!    只在開檔時每張各量一次（拿縮圖量，見 [`WORK_LONG_EDGE`]），
//!    量完就存著；之後拖滑桿都不必再量。
//! 2. [`apply_local`]：**主體強化與柔膚**。這一段要對著眼前這張圖跑，
//!    預覽與存檔各跑一次（尺寸不同）。
//!
//! 調色本身仍走 [`crate::edit::apply_grade`]——與另外三個模組同一份公式，
//! 同一個數字在哪裡都是同一種效果，這裡只負責「決定那個數字」。
//!
//! ## 主體是怎麼判出來的
//!
//! 程式裡沒有神經網路，也不打算為了這個功能塞一個進來（模型檔比整支程式
//! 還大，離線更新也麻煩）。判的是**攝影上「主體長什麼樣」的統計特徵**：
//!
//! - **細節密度**：主體是對到焦的那一塊，高頻能量遠高於天空、水面與散景。
//!   拍鳥、拍動物尤其明顯——背景幾乎糊成一片。
//! - **與周圍的反差**：主體的亮度與顏色和大範圍平均值拉得開
//!   （中心—周圍對比，視覺顯著度最常用的那一條）。
//! - **膚色**：人像另外走 YCbCr 的膚色範圍（見 [`skin_mask`]），
//!   臉與手直接圈得出來，不必靠上面兩條猜。
//!
//! 三者合成一張 0~1 的權重圖，再取最大的幾團連通區域當主體
//! （見 [`keep_main_blobs`]）——散落在背景的枝葉、雜訊會被這一步濾掉。
//!
//! 這一套判得出「哪一塊是重點」，判不出「那是什麼東西」——膚色那一條
//! 尤其如此（被燈光打亮的煙與雲，顏色和皮膚幾乎一樣，見 [`Auto::has_skin`]）。
//! 所以兩件事一開始就設計進去：強化是**依權重漸進**套上去的，不是二值開關，
//! 判偏了也只是輕重之差；而且「顯示主體範圍」把兩張遮罩都畫出來
//! （見 [`mask_overlay`]），對不對使用者自己一眼就看得到。

use image::RgbImage;

use crate::Adjustments;

/// 分析用的工作長邊。量的都是大尺度的統計量（分位數、平均彩度、霧的厚度），
/// 縮到這裡就夠準，開一整批照片才不必等
const WORK_LONG_EDGE: u32 = 640;

/// 主體／膚色遮罩的工作長邊。遮罩只要定得出「哪一塊」，
/// 邊界的一兩個像素無關緊要——套用時是雙線性取樣上去的，本來就會平滑
const MASK_LONG_EDGE: u32 = 384;

/// 優化的類型。三種的差別不在「用不用得到哪一段程式」，而在**目標值**：
/// 同一套量測、不同的期望亮度、對比、彩度與銳利度
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    /// 鳥類：主體小、背景多半是天空或散景，靠細節密度就圈得很準。
    /// 調色偏「通透 ＋ 對比」，主體再另外提亮、加銳
    Bird,
    /// 一般風景：整片畫面都是主體，強化不集中在某一塊。
    /// 調色偏「層次 ＋ 去朦朧」，主體強化預設較輕
    Landscape,
    /// 風景人像：人在景裡。膚色圈得出來就以人為主體，
    /// **臉與皮膚要柔、不要銳**（見 [`apply_local`]）
    Portrait,
}

impl Preset {
    /// 選單順序
    pub const ALL: [Preset; 3] = [Preset::Bird, Preset::Landscape, Preset::Portrait];

    pub fn label(self) -> &'static str {
        match self {
            Preset::Bird => "鳥類",
            Preset::Landscape => "一般風景",
            Preset::Portrait => "風景人像",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Preset::Bird => {
                "主體是鳥：背景（天空、散景）壓得乾淨，鳥身提亮、加清晰與銳利，羽毛紋理跳出來"
            }
            Preset::Landscape => {
                "整片都是主體：以層次、光影與通透度為主，主體強化只做輕微的重點加強"
            }
            Preset::Portrait => {
                "景裡有人：自動找出膚色範圍，臉與皮膚適度柔化、勻膚，清晰度與銳利度跟著調低；\n\
                 頭髮、眼睛、衣服與背景不受影響"
            }
        }
    }

    /// 設定檔裡的穩定識別字串（label 是給人看的中文，不適合存檔）
    pub fn id(self) -> &'static str {
        match self {
            Preset::Bird => "bird",
            Preset::Landscape => "landscape",
            Preset::Portrait => "portrait",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.id() == s)
    }

    /// 主體強化的預設強度。風景整片都是主體，重點加強要收斂一點
    pub fn default_subject(self) -> i32 {
        match self {
            Preset::Bird => 65,
            Preset::Landscape => 30,
            Preset::Portrait => 45,
        }
    }

    /// 柔膚的預設強度。只有人像預設就開著，另外兩種要自己拉
    /// （拍鳥、拍風景照樣可能入鏡一個人，但那不是重點）
    pub fn default_skin(self) -> i32 {
        match self {
            Preset::Portrait => 55,
            _ => 0,
        }
    }
}

/// 量一張照片得到的建議。`grade` 就是那十二條滑桿該停的位置；
/// 另外兩個佔比是拿來在畫面上講一句「這張測到什麼」用的
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Auto {
    pub grade: Adjustments,
    /// 主體佔畫面的比例 0~1
    pub subject_cover: f32,
    /// 膚色佔畫面的比例 0~1（見 [`Auto::has_skin`]：這是顏色的判斷，
    /// 不等於「畫面裡有人」）
    pub skin_cover: f32,
}

/// 膚色要佔畫面多少，柔膚才值得跑一趟。低於這個數的多半是雜訊或
/// 背景裡的一小塊，抹了也看不出差別
pub const SKIN_PRESENT: f32 = 0.012;

impl Auto {
    /// 這張有沒有測到夠大一塊膚色。柔膚要不要作用看它。
    ///
    /// **講的是「膚色」不是「人」**：顏色判得出膚色範圍，判不出那是不是人。
    /// 被燈光打亮的煙、雲，或米色的沙灘與牆面，顏色和皮膚幾乎一樣
    /// （見 [`skin_mask`] 擋掉的那幾種）。所以畫面上一律照實說是「膚色範圍」，
    /// 並且讓使用者用「顯示主體範圍」看得到圈在哪裡——
    /// 圈錯了把柔膚關掉就好，不會毀掉照片
    pub fn has_skin(&self) -> bool {
        self.skin_cover >= SKIN_PRESENT
    }
}

/// 主體強化與柔膚的強度（都是 0~100，0＝不做）
#[derive(Clone, Copy, PartialEq)]
pub struct Local {
    /// 主體強化：提亮、加清晰、加銳利，把主體從背景裡拉出來
    pub subject: i32,
    /// 柔膚：皮膚的細紋與色斑抹勻，邊緣（眼、唇、髮際）留著
    pub skin: i32,
}

impl Local {
    /// 兩樣都不做時整段運算可以跳過
    pub fn is_none(&self) -> bool {
        self.subject <= 0 && self.skin <= 0
    }
}

// ---------- 一、量照片，回推十二條滑桿 ----------

/// 量一張照片，回推調色滑桿該停在哪裡。
///
/// `img` 可以是縮圖（GUI 就是拿縮圖來量）：量的全是分位數與平均值這類
/// 大尺度統計量，縮圖與原圖差不了多少。
///
/// 每一條的推法都寫在下面各段的註解裡，共通的原則有兩條：
///
/// - **往目標值靠，不是往極端拉**。目標值照類型給（見 [`Targets`]），
///   算出來的差距再乘一個小於 1 的係數——寧可調不夠讓人自己補，
///   也不要一鍵把照片調過頭。
/// - **每條都夾在保守的上下限內**。自動判參數最怕的是遇到刻意的低調
///   （逆光剪影、夜景）或高調照片，硬把它拉回「正常」反而毀掉。
pub fn analyze(img: &RgbImage, preset: Preset) -> Auto {
    let small = shrink(img, WORK_LONG_EDGE);
    let (w, h) = (small.width() as usize, small.height() as usize);
    if w < 16 || h < 16 {
        return Auto::default();
    }
    let n = w * h;
    let t = Targets::of(preset);

    // 亮度取編碼值（0~1 的 sRGB）而不是線性值：十二條滑桿全都是對編碼值
    // 動手的（見 edit::apply_grade），在同一個空間裡量才對得起來
    let px: Vec<[f32; 3]> = small
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    let y: Vec<f32> = px.iter().map(|c| luma(*c)).collect();

    let mut sorted = y.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f32| sorted[((n - 1) as f32 * f.clamp(0.0, 1.0)).round() as usize];
    let (p01, p05, p25, p50, p95, p99) = (q(0.01), q(0.05), q(0.25), q(0.5), q(0.95), q(0.99));

    let mut adj = Adjustments::default();

    // --- 低調與高調照片：別硬把它拉回「正常」 ---
    //
    // 上面那組目標值是照「一般光線的照片」訂的。夜景、逆光剪影、黑背景的
    // 生態照本來就大半是暗的，硬套目標亮度會把夜空提成一片灰；雪景、白底
    // 商品照則反過來。
    //
    // 判據有兩條，**兩條都要過才算一般照片**：
    //
    // - 中位數本身。實測一張夜間煙火照的中位數是 0.063；照目標值算下去是
    //   2.8 EV 的提亮，等於把整片夜空提成灰的。照片本來就這麼暗，就是
    //   「它該有的樣子」，不是曝光失誤。
    // - 有多少畫面貼在最暗那一端（同一張有六成以上的像素在 0.10 以下）。
    //
    // 1.0＝一般照片照常調；接近 0＝這是刻意的低調／高調，幾乎不要動
    let frac = |lo: f32, hi: f32| {
        y.iter().filter(|v| **v > lo && **v < hi).count() as f32 / n as f32
    };
    let dark_frac = frac(-1.0, 0.10);
    let bright_frac = frac(0.88, 2.0);
    let low_key = smoothstep(0.06, 0.22, p50) * (1.0 - smoothstep(0.55, 0.90, dark_frac));
    let high_key =
        (1.0 - smoothstep(0.72, 0.92, p50)) * (1.0 - smoothstep(0.45, 0.80, bright_frac));

    // --- 白平衡：只修「明顯的色偏」，而且只修一部分 ---
    //
    // 灰世界假設（整張的平均應該是灰的）套在風景上很危險：夕陽、藍調時刻、
    // 秋天的樹林本來就整片偏色，硬拉回中性等於把作品的味道洗掉。
    // 所以這裡算出來的色偏只採用一部分（見 Targets::wb），上下限也壓得很低；
    // 真正偏掉的照片（陰天發青、鎢絲燈發橘）改個十來格就看得出差別了
    let mut sum = [0.0f32; 3];
    let mut cnt = 0.0f32;
    for (c, yv) in px.iter().zip(y.iter()) {
        // 死黑與死白沒有顏色資訊，把它們算進平均只會稀釋掉真正的色偏
        if *yv > 0.08 && *yv < 0.92 {
            for i in 0..3 {
                sum[i] += c[i];
            }
            cnt += 1.0;
        }
    }
    if cnt > n as f32 * 0.05 {
        let m = [sum[0] / cnt, sum[1] / cnt, sum[2] / cnt];
        let avg = (m[0] + m[1] + m[2]) / 3.0;
        if avg > 0.02 {
            // 偏藍（B 高於 R）就要加暖，所以色溫與 (B − R) 同號
            let cool = (m[2] - m[0]) / avg;
            // 偏綠（G 高於 R、B 的平均）就要加洋紅，色調與它同號
            let green = (m[1] - (m[0] + m[2]) / 2.0) / avg;
            adj.temp = round_clamp(cool * t.wb, -t.wb_max, t.wb_max);
            adj.tint = round_clamp(green * t.wb, -t.wb_max, t.wb_max);
        }
    }

    // --- 去朦朧：先量那層白幕有多厚 ---
    //
    // 薄霧是一層加在畫面上的白幕，特徵是「每個像素的三通道最小值都被墊高」
    // （暗通道先驗）。乾淨的照片裡總有一些像素某個通道接近 0；霧越厚，
    // 那個下限被抬得越高。取低分位數而不是最小值，單顆暗雜訊點才不會說了算
    let mut dark: Vec<f32> = px.iter().map(|c| c[0].min(c[1]).min(c[2])).collect();
    dark.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 取 15% 分位：不是最暗那一點（雜訊），也不是中位數（那是畫面本身的暗部）
    let veil = dark[((n - 1) as f32 * 0.15) as usize];
    // 換算成滑桿：去朦朧拉到 100 正好減掉 DEHAZE_MAX_VEIL 那麼厚的一層
    let veil_slider = (veil / crate::DEHAZE_MAX_VEIL as f32 * 100.0).max(0.0);
    adj.dehaze = round_clamp(veil_slider * t.dehaze, 0.0, t.dehaze_max);
    // 去朦朧已經把黑點往下拉了這麼多，後面的「黑色」只補剩下的那一截，
    // 否則兩條一起壓會把暗部整個壓死
    let veil_taken = adj.dehaze as f32 / 100.0 * crate::DEHAZE_MAX_VEIL as f32;

    // --- 黑色與白色：把直方圖撐滿 ---
    //
    // 「層次感」說穿了就是最暗的地方夠黑、最亮的地方夠白。量最暗與最亮的
    // 那一小撮離兩端還有多遠，不足的用黑色／白色補回來
    let black_gap = (p01 - veil_taken).max(0.0);
    if black_gap > 0.004 {
        // 壓黑是把輸入黑點往右移，位移量 ＝ −blacks/100 × 0.12（見 filter_chain）
        adj.blacks = -round_clamp(black_gap / 0.12 * 100.0 * t.black, 0.0, t.black_max);
    } else if p05 < 0.012 {
        // 暗部已經黏在 0 上（拍死了或壓過頭）：抬一點黑點把細節拉回來
        adj.blacks = round_clamp((0.012 - p05) / 0.15 * 100.0 * 0.8, 0.0, 12.0);
    }
    let white_gap = (1.0 - p99).max(0.0);
    if white_gap > 0.01 {
        // 提白是把輸入白點往左移，位移量 ＝ whites/100 × 0.15
        adj.whites = round_clamp(white_gap / 0.15 * 100.0 * t.white, 0.0, t.white_max);
    } else if p95 > 0.985 {
        // 高光整片黏在 255：往回收一點，雲的層次才回得來
        adj.whites = -round_clamp((p95 - 0.985) / 0.15 * 100.0 * 2.0, 0.0, 12.0);
    }

    // --- 曝光度：把中間調帶到目標亮度 ---
    //
    // 用中位數而不是平均：一片死黑的夜空或一大片白牆會把平均整個帶偏，
    // 中位數說的才是「畫面主要落在哪個亮度」。
    // 曝光是乘上 2^EV（見 filter_chain），所以差距要用比值取對數
    if p50 > 0.004 {
        let ev = (t.mid / p50).log2();
        // 要提亮的是低調照片、要壓暗的是高調照片，各自照自己那一邊收力道
        let damp = if ev > 0.0 { low_key } else { high_key };
        adj.exposure = round_clamp(
            ev / 3.0 * 100.0 * t.exposure * damp,
            -t.exposure_max,
            t.exposure_max,
        );
    }

    // --- 對比：量現在的反差有多大 ---
    //
    // 用 p95 − p05 當「反差」而不是標準差：標準差會被大片同色區域拉低
    // （半張照片是天空的話標準差一定小，但那張不見得平），
    // 分位數差說的是「大部分的內容分佈在多寬的亮度範圍裡」
    let spread = (p95 - p05).max(0.0);
    adj.contrast = round_clamp(
        (t.spread - spread) / t.spread.max(0.05) * 100.0 * t.contrast,
        -t.contrast_max * 0.5,
        t.contrast_max,
    );

    // --- 陰影：暗部有沒有塞太多東西 ---
    //
    // 前面的黑色是「把最暗的一小撮壓到底」，這裡是「把中低調那一大塊抬起來」
    // ——逆光、樹蔭下、日出前的照片就是靠這條把細節撈回來的。
    // 判據用 p25：四分之一的畫面比它暗，它太低就代表暗部擠成一團
    //
    // 這一條同樣要看低調與否：夜景的 p25 一定很低，抬起來就是把夜空洗成灰的
    if p25 < t.shadow_floor {
        adj.shadows = round_clamp(
            (t.shadow_floor - p25) / t.shadow_floor * 100.0 * t.shadow * low_key,
            0.0,
            t.shadow_max,
        );
    }

    // --- 鮮豔度與飽和度：量現在有多濃 ---
    //
    // 彩度取 (max − min)／max（HSV 的 S），對亮度不敏感。
    // 主力給鮮豔度不給飽和度：鮮豔度對本來就濃的顏色施力小，
    // 天空、皮膚不會先爆掉；飽和度只補一點點打底
    let mut chroma = 0.0f32;
    let mut ccnt = 0.0f32;
    for c in &px {
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        if mx > 0.06 {
            chroma += (mx - mn) / mx;
            ccnt += 1.0;
        }
    }
    if ccnt > n as f32 * 0.05 {
        let cur = chroma / ccnt;
        let gap = (t.chroma - cur) / t.chroma.max(0.05);
        adj.vibrance = round_clamp(
            gap * 100.0 * t.vibrance,
            -t.vibrance_max * 0.6,
            t.vibrance_max,
        );
        adj.saturation = round_clamp(gap * 100.0 * t.sat, -t.sat_max * 0.6, t.sat_max);
    }

    // --- 清晰度：量現在的局部反差 ---
    //
    // 局部反差＝亮度與它自己的大半徑模糊差多少。本來就很紮實的照片
    // （逆光的樹林、密集的建築）再加清晰只會髒掉，所以同樣是往目標值靠。
    // 人像的目標值本來就低（見 Targets），臉不會被這條刻出皺紋
    let blur = box_blur(&y, w, h, (w.max(h) / 24).max(2));
    let local: f32 = y.iter().zip(blur.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / n as f32;
    adj.clarity = round_clamp(
        (t.local - local) / t.local.max(0.01) * 100.0 * t.clarity,
        -t.clarity_max * 0.6,
        t.clarity_max,
    );

    // 亮度不參與自動判斷：它與曝光度做的是同一件事（整體提亮），
    // 兩條一起動只會讓人搞不清楚是誰的效果。留給使用者自己微調

    // --- 主體與膚色的佔比（畫面上要講一句「測到什麼」） ---
    let small_mask = shrink(img, MASK_LONG_EDGE);
    let (skin, sw, sh) = skin_mask(&small_mask);
    let skin_cover = skin.iter().sum::<f32>() / (sw * sh).max(1) as f32;
    let (mask, mw, mh) = subject_mask_of(&small_mask, preset, &skin, sw, sh);
    let subject_cover = mask.iter().sum::<f32>() / (mw * mh).max(1) as f32;

    Auto {
        grade: adj.clamped(),
        subject_cover,
        skin_cover,
    }
}

/// 各類型的目標值與施力係數。`*_max` 是那一條的上下限
/// （自動判斷最怕遇到刻意的低調／高調照片，夾住才不會毀掉它）
struct Targets {
    /// 中間調的目標亮度
    mid: f32,
    exposure: f32,
    exposure_max: f32,
    /// p95 − p05 的目標寬度
    spread: f32,
    contrast: f32,
    contrast_max: f32,
    /// p25 低於這個值才抬陰影
    shadow_floor: f32,
    shadow: f32,
    shadow_max: f32,
    black: f32,
    black_max: f32,
    white: f32,
    white_max: f32,
    dehaze: f32,
    dehaze_max: f32,
    /// 平均彩度的目標
    chroma: f32,
    vibrance: f32,
    vibrance_max: f32,
    sat: f32,
    sat_max: f32,
    /// 局部反差的目標
    local: f32,
    clarity: f32,
    clarity_max: f32,
    /// 色偏要修掉幾成
    wb: f32,
    wb_max: f32,
}

impl Targets {
    fn of(p: Preset) -> Self {
        match p {
            // 鳥多半在天空或散景前面：畫面乾淨、反差不足，通透與對比可以給得大方；
            // 主體另外有局部強化，全域清晰度就不必拉太高（背景會跟著髒）
            Preset::Bird => Targets {
                mid: 0.44,
                exposure: 0.55,
                exposure_max: 22.0,
                spread: 0.72,
                contrast: 0.30,
                contrast_max: 26.0,
                shadow_floor: 0.16,
                shadow: 0.55,
                shadow_max: 30.0,
                black: 0.85,
                black_max: 30.0,
                white: 0.85,
                white_max: 32.0,
                dehaze: 0.55,
                dehaze_max: 26.0,
                chroma: 0.30,
                vibrance: 0.40,
                vibrance_max: 26.0,
                sat: 0.12,
                sat_max: 10.0,
                local: 0.055,
                clarity: 0.35,
                clarity_max: 22.0,
                wb: 22.0,
                wb_max: 12.0,
            },
            // 風景要的是層次：陰影抬得多、去朦朧與清晰度給得足，
            // 曝光反而動得少（風景的明暗分佈常常是刻意的）
            Preset::Landscape => Targets {
                mid: 0.42,
                exposure: 0.45,
                exposure_max: 18.0,
                spread: 0.76,
                contrast: 0.32,
                contrast_max: 28.0,
                shadow_floor: 0.18,
                shadow: 0.60,
                shadow_max: 34.0,
                black: 0.90,
                black_max: 32.0,
                white: 0.90,
                white_max: 34.0,
                dehaze: 0.70,
                dehaze_max: 32.0,
                chroma: 0.32,
                vibrance: 0.45,
                vibrance_max: 30.0,
                sat: 0.15,
                sat_max: 12.0,
                local: 0.060,
                clarity: 0.45,
                clarity_max: 28.0,
                wb: 20.0,
                wb_max: 10.0,
            },
            // 人像：亮一點、對比與清晰度都收斂（那兩條專門刻毛孔與皺紋），
            // 彩度也不能高，否則臉先紅掉
            Preset::Portrait => Targets {
                mid: 0.50,
                exposure: 0.60,
                exposure_max: 24.0,
                spread: 0.68,
                contrast: 0.20,
                contrast_max: 16.0,
                shadow_floor: 0.20,
                shadow: 0.65,
                shadow_max: 36.0,
                black: 0.70,
                black_max: 22.0,
                white: 0.70,
                white_max: 24.0,
                dehaze: 0.45,
                dehaze_max: 18.0,
                chroma: 0.26,
                vibrance: 0.30,
                vibrance_max: 18.0,
                sat: 0.08,
                sat_max: 6.0,
                // 目標局部反差刻意設得比一般照片還低：算出來多半是負的，
                // 也就是「主動柔一點」——與柔膚同一個方向
                local: 0.042,
                clarity: 0.40,
                clarity_max: 10.0,
                wb: 26.0,
                wb_max: 12.0,
            },
        }
    }
}

// ---------- 二、主體強化與柔膚 ----------

/// 把主體強化與柔膚套到 `img` 上（就地改）。
///
/// 預覽與存檔都走這裡，差別只在傳進來的尺寸：所有半徑都照長邊等比換算
/// （與 [`crate::edit::apply_grade`] 的清晰度同一個作法），
/// 一張 6000px 的原圖與它 1600px 的預覽才會呈現同一種效果。
///
/// 順序是**先柔膚、再強化**：反過來的話剛加上去的銳利度會被柔膚抹掉一半，
/// 等於白做。人像模式下強化本身也會避開皮膚（見 `avoid_skin`）——
/// 使用者要的是「頭髮、眼睛、衣服清楚，臉柔」，不是整張一起銳
pub fn apply_local(img: &mut RgbImage, preset: Preset, l: Local) {
    if l.is_none() {
        return;
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 8 || h < 8 {
        return;
    }
    let long = w.max(h);

    // 兩張遮罩都在小圖上算，套用時再雙線性取樣回原尺寸
    let small = shrink(img, MASK_LONG_EDGE);
    let (skin, sw, sh) = skin_mask(&small);
    let skin_amt = l.skin as f32 / 100.0;
    // 柔膚只在真的有皮膚時才做：整張沒人卻硬跑一次，只是白花時間
    let skin_cover = skin.iter().sum::<f32>() / (sw * sh).max(1) as f32;
    if skin_amt > 0.0 && skin_cover >= SKIN_PRESENT {
        smooth_skin(img, &skin, sw, sh, skin_amt, long);
    }

    if l.subject > 0 {
        let (mask, mw, mh) = subject_mask_of(&small, preset, &skin, sw, sh);
        // 人像模式下皮膚要避開銳化：把膚色從強化權重裡扣掉，
        // 剩下的（頭髮、眼睛、衣服、背景裡的樹）照常加清晰與銳利
        let avoid_skin = preset == Preset::Portrait;
        sharpen_subject(
            img,
            &mask,
            mw,
            mh,
            &skin,
            sw,
            sh,
            avoid_skin,
            l.subject as f32 / 100.0,
            long,
        );
    }
}

/// 主體強化：提亮 ＋ 局部對比 ＋ 銳利 ＋ 一點點彩度。
///
/// 四樣一起做才像「主體跳出來」：只提亮會糊、只加銳會乾。
/// 每一樣的施力都乘上遮罩權重，邊界是漸進的，不會出現剪貼般的接縫
#[allow(clippy::too_many_arguments)]
fn sharpen_subject(
    img: &mut RgbImage,
    mask: &[f32],
    mw: usize,
    mh: usize,
    skin: &[f32],
    sw: usize,
    sh: usize,
    avoid_skin: bool,
    amount: f32,
    long: usize,
) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let y: Vec<f32> = img
        .pixels()
        .map(|p| {
            luma([
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ])
        })
        .collect();
    // 局部對比用大半徑（與 edit.rs 的清晰度同一個尺規：長邊 ÷ 300），
    // 銳利用小半徑——兩者疊起來就是「輪廓清楚、體積感也在」。
    //
    // 兩份差值當場合成一張 `detail` 就好，不要把兩張模糊圖都留著：
    // 四千五百萬像素的照片，一張浮點平面就是 180MB
    let r_local = (long / 300).clamp(1, 60);
    let r_sharp = (long / 900).clamp(1, 12);
    // 係數照清晰度滑桿的尺規訂（+100 相當於 1.5），這裡各取一半上下，
    // 主體不會刻得太用力
    let mut detail = box_blur(&y, w, h, r_local);
    for (d, y0) in detail.iter_mut().zip(y.iter()) {
        *d = (*y0 - *d) * 0.75;
    }
    {
        let sharp = box_blur(&y, w, h, r_sharp);
        for ((d, y0), s) in detail.iter_mut().zip(y.iter()).zip(sharp.iter()) {
            *d += (*y0 - *s) * 0.55;
        }
    }
    // 亮度在下面的迴圈裡由像素自己算得出來，這一份可以先放掉
    drop(y);

    for (i, p) in img.pixels_mut().enumerate() {
        let (x, yy) = (i % w, i / w);
        let mut m = sample(mask, mw, mh, x, yy, w, h);
        if avoid_skin {
            m *= 1.0 - sample(skin, sw, sh, x, yy, w, h);
        }
        if m <= 0.002 {
            continue;
        }
        let k = m * amount;
        let c = [
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        ];
        let base = luma(c);
        // 提亮：乘法而不是加法，暗部提得多、亮部不會先爆掉
        let gain = 1.0 + 0.18 * k;
        let ny = base * gain + detail[i] * k;
        // 彩度只加一成：主體本來就該比背景實一些，但過頭就假了
        let s = 1.0 + 0.10 * k;
        for ch in 0..3 {
            let v = ny + (c[ch] - base) * s;
            p[ch] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
}

/// 柔膚：低頻留著、高頻按權重扣掉（頻率分離的簡化版）。
///
/// 直接對皮膚做高斯模糊會把眼睛、嘴唇、髮際線一起糊掉，看起來像塑膠。
/// 這裡多一道「邊緣保護」：高頻本身**振幅越大的地方扣得越少**——
/// 毛孔、細紋的振幅小，會被抹平；睫毛、唇線、輪廓的振幅大，原樣留著。
///
/// 走法是「先算好每個像素要抹多少，再一個通道一個通道抹」。
/// 三個通道的模糊圖同時攤開的話，一張四千五百萬像素的照片光是這三份就
/// 540MB；分開做則同時只有一份，而且**邊緣保護算得出來**——
/// 模糊與亮度都是線性運算，「模糊後的亮度」等於「亮度的模糊」，
/// 不必先把三個通道都模糊完才知道高頻有多大
fn smooth_skin(img: &mut RgbImage, skin: &[f32], sw: usize, sh: usize, amount: f32, long: usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    // 半徑照長邊等比：預覽與原圖要抹掉的是同一個尺度的東西（毛孔、細紋）
    let r = (long / 260).clamp(1, 90);

    // 第一步：每個像素要抹多少。存成 u8（0~255 對 0~1）——這是個混色比例，
    // 1/255 的級距看不出來，卻省下四分之三的記憶體
    let strength: Vec<u8> = {
        let y: Vec<f32> = img
            .pixels()
            .map(|p| {
                luma([
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ])
            })
            .collect();
        let lo = box_blur(&y, w, h, r);
        (0..w * h)
            .map(|i| {
                let m = sample(skin, sw, sh, i % w, i / w, w, h);
                if m <= 0.004 {
                    return 0;
                }
                // 邊緣保護：亮度上的高頻振幅越大，越當成「五官」而不是「皮膚紋理」
                let keep = smoothstep(0.045, 0.16, (y[i] - lo[i]).abs());
                // 0.85 是上限：留一成五的紋理，皮膚才不會變成一塊塑膠
                let k = m * amount * (1.0 - keep) * 0.85;
                (k.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            })
            .collect()
    };

    // 第二步：一次抹一個通道。高頻按 k 扣掉＝在低頻與原圖之間依 k 內插
    for ch in 0..3 {
        let plane: Vec<f32> = img.pixels().map(|p| p[ch] as f32 / 255.0).collect();
        let lo = box_blur(&plane, w, h, r);
        for (i, p) in img.pixels_mut().enumerate() {
            let k = strength[i] as f32 / 255.0;
            if k <= 0.0 {
                continue;
            }
            let v = lo[i] + (plane[i] - lo[i]) * (1.0 - k);
            p[ch] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
}

// ---------- 三、遮罩 ----------

/// 把兩張遮罩畫成一張看得見的圖（給「顯示主體範圍」用）：
/// **主體維持原樣、膚色染粉紅、其餘壓暗染藍**。
///
/// 有這個檢視才敢說「自動判斷主體」——判得對不對，使用者自己一眼就看得到，
/// 不必憑感覺猜（比照去煙霧模組的「顯示遮色片」）。膚色也一起畫出來是
/// 必要的：顏色判得出膚色範圍卻判不出那是不是人（見 [`Auto::has_skin`]），
/// 柔膚要抹在哪裡總得先讓人看見
pub fn mask_overlay(img: &RgbImage, preset: Preset) -> RgbImage {
    let small = shrink(img, MASK_LONG_EDGE);
    let (skin, sw, sh) = skin_mask(&small);
    let (mask, mw, mh) = subject_mask_of(&small, preset, &skin, sw, sh);
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = img.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        let (x, y) = (i % w, i / w);
        let m = sample(&mask, mw, mh, x, y, w, h);
        let s = sample(&skin, sw, sh, x, y, w, h);
        // 主體以外壓到三成亮度、往藍色帶（紅色留給去煙霧的遮色片，
        // 兩個模組的檢視畫面才不會看混）；膚色那一塊改染粉紅
        let dim = 0.30 + 0.70 * m.max(s);
        let cold = [0.35, 0.55, 1.0];
        let warm = [1.0, 0.45, 0.62];
        for ch in 0..3 {
            // 主體（m）保持原色，其餘依「是不是膚色」在兩種染色之間取
            let tint = cold[ch] + (warm[ch] - cold[ch]) * s;
            let v = p[ch] as f32 / 255.0 * dim * (m + (1.0 - m) * tint);
            p[ch] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    out
}

/// 主體遮罩的本體。三個線索合成一張權重圖：
///
/// 1. **細節密度**（對到焦的那一塊）
/// 2. **中心—周圍反差**（亮度與顏色和大範圍平均差多少）
/// 3. **膚色**（人像才加權；臉與手直接圈得出來）
///
/// 合成後取一個分位數當門檻做二值化，只留最大的幾團連通區域
/// （見 [`keep_main_blobs`]），最後模糊成漸進的邊緣
fn subject_mask_of(
    small: &RgbImage,
    preset: Preset,
    skin: &[f32],
    sw: usize,
    sh: usize,
) -> (Vec<f32>, usize, usize) {
    let (w, h) = (small.width() as usize, small.height() as usize);
    let n = w * h;
    if n == 0 {
        return (Vec::new(), 0, 0);
    }
    let px: Vec<[f32; 3]> = small
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    let y: Vec<f32> = px.iter().map(|c| luma(*c)).collect();

    // --- 1. 細節密度 ---
    // 先取「亮度與小半徑模糊的差」＝高頻能量，再用中半徑把它抹開，
    // 得到的是「這一帶有多少細節」而不是「這一點是不是邊緣」。
    // 主體對到焦、背景散掉時，兩者差距非常明顯
    let r_fine = (w.max(h) / 128).max(1);
    let fine = box_blur(&y, w, h, r_fine);
    let edge: Vec<f32> = y
        .iter()
        .zip(fine.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let detail = box_blur(&edge, w, h, (w.max(h) / 20).max(2));

    // --- 2. 中心—周圍反差 ---
    // 亮度與顏色各算一份：白鳥在藍天前是亮度差、紅花在綠葉前是顏色差，
    // 少了任何一邊都會漏掉一種常見的情況
    let r_wide = (w.max(h) / 5).max(3);
    let wide_y = box_blur(&y, w, h, r_wide);
    // 顏色用兩條對立軸（紅—綠、黃—藍），這是分辨顏色差異最省事的表示法
    let rg: Vec<f32> = px.iter().map(|c| c[0] - c[1]).collect();
    let yb: Vec<f32> = px.iter().map(|c| (c[0] + c[1]) / 2.0 - c[2]).collect();
    let wide_rg = box_blur(&rg, w, h, r_wide);
    let wide_yb = box_blur(&yb, w, h, r_wide);
    let cs: Vec<f32> = (0..n)
        .map(|i| {
            let dy = (y[i] - wide_y[i]).abs();
            let dc = ((rg[i] - wide_rg[i]).powi(2) + (yb[i] - wide_yb[i]).powi(2)).sqrt();
            dy + dc
        })
        .collect();
    let cs = box_blur(&cs, w, h, (w.max(h) / 32).max(2));

    // --- 3. 合成 ---
    let dn = norm(&detail);
    let cn = norm(&cs);
    let mut sal: Vec<f32> = (0..n).map(|i| dn[i] * 0.60 + cn[i] * 0.40).collect();
    // 很輕的中央加權：構圖上主體確實偏中間，但拍鳥常常刻意留白，
    // 權重給重了會把邊上的鳥判掉——所以只讓它在平手時當個裁判
    for (i, v) in sal.iter_mut().enumerate() {
        let (x, yy) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
        let d = ((x - 0.5).powi(2) + (yy - 0.5).powi(2)).sqrt() / 0.707;
        *v *= 0.86 + 0.14 * (1.0 - d);
    }
    // 人像：膚色直接算主體。臉是最確定的一塊，不必靠統計去猜
    if preset == Preset::Portrait {
        for (i, v) in sal.iter_mut().enumerate() {
            let (x, yy) = (i % w, i / w);
            *v = v.max(sample(skin, sw, sh, x, yy, w, h));
        }
    }

    // --- 4. 門檻與連通區域 ---
    // 門檻取分位數而不是固定值：每張照片的顯著度尺度都不一樣，
    // 固定門檻會在乾淨的照片上圈進整片天空、在雜亂的照片上什麼都圈不到
    let cover = match preset {
        Preset::Bird => 0.16,
        Preset::Landscape => 0.32,
        Preset::Portrait => 0.26,
    };
    let mut sorted = sal.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thr = sorted[((n - 1) as f32 * (1.0 - cover)) as usize];
    let mut m: Vec<f32> = sal
        .iter()
        .map(|v| smoothstep(thr * 0.75, thr * 1.15, *v))
        .collect();
    // 風景不做連通區域：整片畫面本來就是主體，硬留一團反而變成怪異的亮斑
    if preset != Preset::Landscape {
        keep_main_blobs(&mut m, w, h, 0.20, 0.0);
    }
    // 最後抹開：邊界漸進，強化才不會沿著遮罩邊緣留下一圈痕跡
    let m = box_blur(&m, w, h, (w.max(h) / 40).max(2));
    (m, w, h)
}

/// 濾掉零碎的小團塊。通過門檻的東西不只有主體——背景的枝葉、水面反光、
/// 雜訊也會中，但它們是散落的小塊；主體（或一張臉）則是連成一大片。
///
/// 一團要留下來，面積必須同時大於：
/// - `rel` × 最大那一團（主體常被前景樹枝切成好幾塊，只留最大的會把
///   鳥的尾巴、人的手切掉，所以留一個比例而不是只留第一名）
/// - `floor` × 整張畫面（絕對下限。膚色那邊沒有「最大的一團」可以當基準
///   ——兩張臉一大一小都該算數——靠的就是這一條）
///
/// 用的是四連通的標記，以堆疊展開、不遞迴：一張 384px 的圖最深可以
/// 連到十萬層，遞迴會直接爆堆疊
fn keep_main_blobs(m: &mut [f32], w: usize, h: usize, rel: f32, floor: f32) {
    let n = w * h;
    let mut label = vec![u32::MAX; n];
    let mut areas: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if m[start] < 0.5 || label[start] != u32::MAX {
            continue;
        }
        let id = areas.len() as u32;
        let mut area = 0usize;
        label[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            area += 1;
            let (x, y) = (i % w, i / w);
            let mut neighbours = [usize::MAX; 4];
            if x > 0 {
                neighbours[0] = i - 1;
            }
            if x + 1 < w {
                neighbours[1] = i + 1;
            }
            if y > 0 {
                neighbours[2] = i - w;
            }
            if y + 1 < h {
                neighbours[3] = i + w;
            }
            for j in neighbours {
                if j != usize::MAX && m[j] >= 0.5 && label[j] == u32::MAX {
                    label[j] = id;
                    stack.push(j);
                }
            }
        }
        areas.push(area);
    }
    // 一團也沒有（門檻抓得太緊）就原樣留著，總比全部歸零好——
    // 全歸零等於強化默默失效，使用者只會覺得「滑桿沒反應」
    let Some(&biggest) = areas.iter().max() else {
        return;
    };
    let keep = ((biggest as f32 * rel) as usize)
        .max((n as f32 * floor) as usize)
        .max(1);
    for i in 0..n {
        match label[i] {
            u32::MAX => {}
            id if areas[id as usize] >= keep => {}
            // 沒過門檻的小碎塊：留一點點權重而不是砍成 0，
            // 免得遮罩邊上出現一顆一顆的洞
            _ => m[i] *= 0.15,
        }
    }
}

/// 膚色遮罩（0~1）。用的是 YCbCr 上那塊眾所皆知的膚色範圍：
/// 不同人種的差別主要在亮度（Y），色度（Cb、Cr）落在同一塊區域裡，
/// 所以判色度、放寬亮度。
///
/// 邊界不是硬切的：越靠近範圍中心權重越高，柔膚才不會在臉的邊緣斷一刀。
///
/// 光靠 YCbCr 的範圍會誤判一大票東西，所以另外要求兩件事：
///
/// - **R > G > B**（皮膚一定偏紅）。少了這條，木頭、磚牆、沙灘都會中。
/// - **彩度不能太高**。皮膚的彩度落在 0.15~0.45，很少超過 0.5；煙火的
///   橘紅軌跡、夕陽、紅色衣服則動輒 0.7 以上。實測一張夜間煙火照，
///   少了這條會有兩成畫面被當成皮膚——那會讓人像模式對著煙火做柔膚
fn skin_mask(small: &RgbImage) -> (Vec<f32>, usize, usize) {
    let (w, h) = (small.width() as usize, small.height() as usize);
    let mut m = Vec::with_capacity(w * h);
    for p in small.pixels() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        // 太暗與死白都判不出顏色。下限訂在 45~75（而不是貼著黑）是刻意的：
        // 一張暗到這個程度的臉就是逆光剪影，本來也不該對它柔膚；反過來，
        // 夜景裡大片昏暗的暖色（煙火餘燼的輝光、遠處街燈映的天空）
        // 會通過顏色那幾關，就靠這一條擋下來
        let wy = smoothstep(45.0, 75.0, y) * (1.0 - smoothstep(238.0, 252.0, y));
        // Cb 77~127、Cr 133~173 是膚色的經典範圍，兩端各留一段漸進
        let wcb = smoothstep(72.0, 84.0, cb) * (1.0 - smoothstep(120.0, 132.0, cb));
        let wcr = smoothstep(128.0, 138.0, cr) * (1.0 - smoothstep(168.0, 178.0, cr));
        // 偏紅程度：R 比 G 高、G 比 B 高
        let wrg = smoothstep(4.0, 16.0, r - g) * smoothstep(0.0, 10.0, g - b);
        // 彩度要落在一個帶子裡，兩端都不是皮膚：
        // 太濃（> 0.5）是紅衣服、夕陽、煙火軌跡；
        // 太淡（< 0.15）是帶暖調的白——被燈光打亮的煙、雲、米色的牆
        let mx = r.max(g).max(b);
        let sat = if mx > 1.0 { (mx - r.min(g).min(b)) / mx } else { 0.0 };
        let wsat = smoothstep(0.12, 0.20, sat) * (1.0 - smoothstep(0.42, 0.60, sat));
        m.push(wy * wcb * wcr * wrg * wsat);
    }
    // 抹開：單顆像素過關的多半是雜訊，成片過關的才是臉
    let m = box_blur(&m, w, h, (w.max(h) / 60).max(1));
    // 模糊之後邊緣會拖出一圈很淡的尾巴，壓掉它才不會把嘴唇周圍一起柔掉
    let mut m: Vec<f32> = m.iter().map(|v| smoothstep(0.12, 0.45, *v)).collect();
    // 最後濾掉零碎的小團：一張臉是連成一片的，就算在風景裡只佔一點點，
    // 也遠大於千分之一的畫面。散在各處的暖色雜訊（煙火的餘燼、遠處燈火、
    // 秋葉的縫隙）過得了顏色那一關，卻過不了這一關
    keep_main_blobs(&mut m, w, h, 0.0, 0.001);
    (m, w, h)
}

// ---------- 共用小工具 ----------

/// BT.709 亮度（與 edit.rs 的鮮豔度同一組係數）
fn luma(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

fn smoothstep(e0: f32, e1: f32, v: f32) -> f32 {
    if e1 <= e0 {
        return if v >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 把 f32 四捨五入成滑桿的整數值並夾住範圍
fn round_clamp(v: f32, lo: f32, hi: f32) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    v.clamp(lo, hi).round() as i32
}

/// 正規化到 0~1（用 98% 分位數當上限，單顆極端值不會把整張壓平）
fn norm(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let hi = s[((s.len() - 1) as f32 * 0.98) as usize].max(1e-6);
    v.iter().map(|x| (x / hi).clamp(0.0, 1.0)).collect()
}

/// 在小圖遮罩上取樣（雙線性）。遮罩本來就是平滑的低頻資料，
/// 雙線性放大看不出接縫，也省下一張原尺寸遮罩的記憶體
fn sample(m: &[f32], mw: usize, mh: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    if m.is_empty() || mw == 0 || mh == 0 {
        return 0.0;
    }
    if mw == w && mh == h {
        return m[y * w + x];
    }
    let fx = (x as f32 + 0.5) * mw as f32 / w as f32 - 0.5;
    let fy = (y as f32 + 0.5) * mh as f32 / h as f32 - 0.5;
    let x0 = fx.floor().max(0.0) as usize;
    let y0 = fy.floor().max(0.0) as usize;
    let x1 = (x0 + 1).min(mw - 1);
    let y1 = (y0 + 1).min(mh - 1);
    let x0 = x0.min(mw - 1);
    let y0 = y0.min(mh - 1);
    let tx = (fx - x0 as f32).clamp(0.0, 1.0);
    let ty = (fy - y0 as f32).clamp(0.0, 1.0);
    let a = m[y0 * mw + x0] * (1.0 - tx) + m[y0 * mw + x1] * tx;
    let b = m[y1 * mw + x0] * (1.0 - tx) + m[y1 * mw + x1] * tx;
    a * (1.0 - ty) + b * ty
}

/// 可分離的盒狀模糊（先橫再直）。與半徑無關都是 O(n)——
/// 原尺寸的照片才跑得動。中間轉置一次，兩趟的記憶體存取都是連續的
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 || src.len() != w * h {
        return src.to_vec();
    }
    /// 對每一列做一次滑動平均。邊界用端點值延伸（不是補 0），
    /// 否則四邊會暗一圈
    fn pass(input: &[f32], out: &mut [f32], w: usize, h: usize, r: usize) {
        let win = (2 * r + 1) as f32;
        for y in 0..h {
            let row = &input[y * w..(y + 1) * w];
            let mut acc: f32 = row[0] * (r + 1) as f32;
            for x in 1..=r {
                acc += row[x.min(w - 1)];
            }
            for x in 0..w {
                out[y * w + x] = acc / win;
                acc += row[(x + r + 1).min(w - 1)];
                acc -= row[x.saturating_sub(r)];
            }
        }
    }
    let transpose = |src: &[f32], w: usize, h: usize| {
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                out[x * h + y] = src[y * w + x];
            }
        }
        out
    };
    // 每一步用完就放掉：原尺寸的照片一張浮點平面動輒上百 MB，
    // 四份同時攤著與兩份差了一倍的峰值
    let mut tmp = vec![0.0f32; w * h];
    pass(src, &mut tmp, w, h, r);
    let t = transpose(&tmp, w, h);
    drop(tmp);
    let mut tmp2 = vec![0.0f32; w * h];
    pass(&t, &mut tmp2, h, w, r);
    drop(t);
    transpose(&tmp2, h, w)
}

/// 等比縮到長邊不超過 `long`（已經夠小就原樣複製）
fn shrink(img: &RgbImage, long: u32) -> RgbImage {
    let cur = img.width().max(img.height());
    if cur <= long {
        return img.clone();
    }
    let s = long as f32 / cur as f32;
    image::imageops::resize(
        img,
        ((img.width() as f32 * s).round() as u32).max(1),
        ((img.height() as f32 * s).round() as u32).max(1),
        image::imageops::FilterType::Triangle,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 產生一張純色圖
    fn solid(w: u32, h: u32, c: [u8; 3]) -> RgbImage {
        RgbImage::from_fn(w, h, |_, _| image::Rgb(c))
    }

    /// 遮罩的平均值＝它蓋住畫面的比例
    fn cover(m: &[f32]) -> f32 {
        m.iter().sum::<f32>() / m.len().max(1) as f32
    }

    #[test]
    fn 盒狀模糊大致保住平均值() {
        let src = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = box_blur(&src, 3, 3, 1);
        let a: f32 = src.iter().sum::<f32>() / 9.0;
        let b: f32 = out.iter().sum::<f32>() / 9.0;
        // 邊界用端點延伸，平均會有一點點偏移，但不該差太多
        assert!((a - b).abs() < 0.15, "{a} vs {b}");
    }

    #[test]
    fn 盒狀模糊把單點抹開() {
        let mut src = vec![0.0f32; 81];
        src[40] = 1.0;
        let out = box_blur(&src, 9, 9, 2);
        assert!(out[40] < 0.1, "中心該被抹淡：{}", out[40]);
        assert!(out[40 - 9] > 0.0, "上面的鄰居該分到一些");
    }

    #[test]
    fn 盒狀模糊不會被大於影像的半徑弄壞() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = box_blur(&src, 4, 3, 50);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn 偏藍的照片會被加暖() {
        let img = solid(64, 64, [90, 100, 150]);
        let a = analyze(&img, Preset::Landscape);
        assert!(a.grade.temp > 0, "偏藍應該加暖，得到 {}", a.grade.temp);
    }

    #[test]
    fn 偏橘的照片會被降溫() {
        let img = solid(64, 64, [170, 110, 70]);
        let a = analyze(&img, Preset::Landscape);
        assert!(a.grade.temp < 0, "偏暖應該降溫，得到 {}", a.grade.temp);
    }

    #[test]
    fn 灰濛濛的照片會被加去朦朧與對比() {
        // 對比極低、黑點被墊高：典型的「有霧」
        let img = RgbImage::from_fn(128, 128, |x, _| {
            let v = 110 + (x % 16) as u8;
            image::Rgb([v, v, v])
        });
        let a = analyze(&img, Preset::Landscape);
        assert!(a.grade.dehaze > 0, "該建議去朦朧，得到 {}", a.grade.dehaze);
        assert!(
            a.grade.contrast > 0,
            "該建議加對比，得到 {}",
            a.grade.contrast
        );
    }

    #[test]
    fn 偏暗的照片會被提亮() {
        let img = solid(96, 96, [40, 42, 45]);
        let a = analyze(&img, Preset::Landscape);
        assert!(a.grade.exposure > 0, "該建議提亮，得到 {}", a.grade.exposure);
    }

    /// 夜景是刻意的低調照片，不該被硬拉回「一般亮度」。
    /// 實測一張夜間煙火照（中位數 0.063、六成像素在 0.10 以下）
    /// 原本會被建議 +22 曝光、+30 陰影——那等於把整片夜空提成灰的
    #[test]
    fn 夜景不會被硬提亮() {
        // 大半是夜空，只有一小塊亮部（煙火）
        let img = RgbImage::from_fn(128, 128, |x, y| {
            if x < 20 && y < 20 {
                image::Rgb([230, 200, 170])
            } else {
                image::Rgb([10, 11, 16])
            }
        });
        for p in Preset::ALL {
            let a = analyze(&img, p);
            assert!(
                a.grade.exposure <= 3,
                "{p:?} 夜景不該提亮，得到 {}",
                a.grade.exposure
            );
            assert!(
                a.grade.shadows <= 3,
                "{p:?} 夜景不該抬陰影，得到 {}",
                a.grade.shadows
            );
        }
    }

    /// 一般亮度、只是拍暗了的照片仍要提亮——低調的判斷不能把它一起擋掉
    #[test]
    fn 只是拍暗了的照片照樣提亮() {
        let img = solid(96, 96, [58, 60, 63]);
        let a = analyze(&img, Preset::Landscape);
        assert!(a.grade.exposure > 5, "該提亮，得到 {}", a.grade.exposure);
    }

    #[test]
    fn 每一條建議都在合法範圍內() {
        let img = RgbImage::from_fn(96, 96, |x, y| {
            image::Rgb([(x * 2) as u8, (y * 2) as u8, ((x + y) % 256) as u8])
        });
        for p in Preset::ALL {
            let a = analyze(&img, p);
            for v in a.grade.values() {
                assert!((-100..=100).contains(&v), "{p:?} 超出範圍：{v}");
            }
        }
    }

    #[test]
    fn 人像的清晰度上限比風景低() {
        // 同一張照片，人像判出來的清晰度不該比風景還兇
        let img = RgbImage::from_fn(128, 128, |x, _| {
            let v = 90 + (x % 8) as u8;
            image::Rgb([v, v, v])
        });
        let land = analyze(&img, Preset::Landscape).grade.clarity;
        let port = analyze(&img, Preset::Portrait).grade.clarity;
        assert!(port <= land, "人像 {port} 該 ≦ 風景 {land}");
    }

    #[test]
    fn 膚色判得出來也不會把綠葉當成皮膚() {
        let (m, _, _) = skin_mask(&solid(64, 64, [222, 176, 148]));
        assert!(cover(&m) > 0.8, "膚色該幾乎全中，得到 {}", cover(&m));

        let (m, _, _) = skin_mask(&solid(64, 64, [70, 130, 60]));
        assert!(cover(&m) < 0.02, "綠葉不該被當成皮膚，得到 {}", cover(&m));

        let (m, _, _) = skin_mask(&solid(64, 64, [120, 130, 145]));
        assert!(cover(&m) < 0.02, "灰藍不該被當成皮膚，得到 {}", cover(&m));
    }

    /// 膚色範圍最容易誤收的三種東西：太濃的暖色（紅衣、煙火軌跡）、
    /// 太淡的暖白（被燈光打亮的煙與雲、米色牆面），以及太暗的暖色
    #[test]
    fn 濃到或淡到不像皮膚的暖色都擋得掉() {
        for (c, what) in [
            ([235u8, 60, 30], "濃橘紅"),
            ([245, 232, 224], "暖白"),
            ([46, 36, 30], "暗棕"),
        ] {
            let (m, _, _) = skin_mask(&solid(64, 64, c));
            assert!(cover(&m) < 0.05, "{what} 不該被當成皮膚，得到 {}", cover(&m));
        }
    }

    /// 散在各處的暖色小點（餘燼、遠處燈火）過得了顏色那一關，
    /// 但過不了「連成一片」那一關
    #[test]
    fn 零碎的膚色小點會被濾掉() {
        // 每隔 8 像素放一個 1px 的膚色點，其餘是深藍夜空
        let img = RgbImage::from_fn(256, 256, |x, y| {
            if x % 8 == 0 && y % 8 == 0 {
                image::Rgb([222, 176, 148])
            } else {
                image::Rgb([12, 14, 30])
            }
        });
        let (m, _, _) = skin_mask(&shrink(&img, MASK_LONG_EDGE));
        assert!(cover(&m) < 0.02, "零碎小點不該算膚色，得到 {}", cover(&m));
    }

    #[test]
    fn 主體遮罩會找到清楚的那一塊() {
        // 左半邊是平坦的天空、右半邊塞滿高頻細節（模擬對到焦的主體）
        let img = RgbImage::from_fn(200, 120, |x, y| {
            if x < 100 {
                image::Rgb([150, 175, 210])
            } else {
                let v = if (x + y) % 2 == 0 { 40 } else { 210 };
                image::Rgb([v, v, v])
            }
        });
        let small = shrink(&img, MASK_LONG_EDGE);
        let (skin, sw, sh) = skin_mask(&small);
        let (m, w, h) = subject_mask_of(&small, Preset::Bird, &skin, sw, sh);
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        for i in 0..w * h {
            if i % w < w / 2 {
                left += m[i];
            } else {
                right += m[i];
            }
        }
        assert!(right > left * 2.0, "細節那半邊該是主體：{left} vs {right}");
    }

    #[test]
    fn 沒開任何強化就完全不動照片() {
        let src = RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, 128]));
        let mut img = src.clone();
        apply_local(&mut img, Preset::Bird, Local { subject: 0, skin: 0 });
        assert_eq!(src, img);
    }

    #[test]
    fn 整張沒有人就不做柔膚() {
        let src = RgbImage::from_fn(64, 64, |x, y| image::Rgb([40, x as u8, y as u8]));
        let mut img = src.clone();
        apply_local(&mut img, Preset::Portrait, Local { subject: 0, skin: 100 });
        assert_eq!(src, img, "沒有膚色就不該動到任何像素");
    }

    #[test]
    fn 柔膚只動皮膚不動背景() {
        // 左半邊膚色 ＋ 細紋，右半邊綠色 ＋ 同樣的細紋
        let base = RgbImage::from_fn(128, 64, |x, y| {
            let n = if (x + y) % 2 == 0 { 12i32 } else { -12 };
            let c = if x < 64 { [222, 176, 148] } else { [70, 130, 60] };
            image::Rgb([
                (c[0] + n).clamp(0, 255) as u8,
                (c[1] + n).clamp(0, 255) as u8,
                (c[2] + n).clamp(0, 255) as u8,
            ])
        });
        let mut img = base.clone();
        apply_local(
            &mut img,
            Preset::Portrait,
            Local { subject: 0, skin: 100 },
        );
        let diff = |x0: u32, x1: u32| -> f32 {
            let mut d = 0.0;
            for x in x0..x1 {
                for y in 0..64 {
                    for c in 0..3 {
                        d += (img.get_pixel(x, y)[c] as f32 - base.get_pixel(x, y)[c] as f32).abs();
                    }
                }
            }
            d
        };
        // 邊界那幾行會被遮罩的羽化沾到，各留 8px 不算
        let skin_side = diff(0, 56);
        let other_side = diff(72, 128);
        assert!(skin_side > 0.0, "皮膚該被抹平");
        assert!(
            other_side < skin_side * 0.2,
            "背景幾乎不該動：皮膚 {skin_side} / 背景 {other_side}"
        );
    }

    #[test]
    fn 主體強化不會把像素算爆() {
        let img = RgbImage::from_fn(160, 100, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x * y) % 256) as u8])
        });
        for p in Preset::ALL {
            let mut out = img.clone();
            apply_local(
                &mut out,
                p,
                Local {
                    subject: 100,
                    skin: 100,
                },
            );
            assert_eq!(out.dimensions(), img.dimensions());
        }
    }

    #[test]
    fn 遮罩檢視不會改變尺寸() {
        let img = RgbImage::from_fn(120, 80, |x, y| {
            image::Rgb([(x * 2) as u8, (y * 3) as u8, 90])
        });
        let out = mask_overlay(&img, Preset::Bird);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn 類型的識別字串轉得回來() {
        for p in Preset::ALL {
            assert_eq!(Preset::from_id(p.id()), Some(p));
        }
        assert_eq!(Preset::from_id("nope"), None);
    }
}
