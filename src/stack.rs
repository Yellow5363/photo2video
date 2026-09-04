//! 煙火疊圖：把同一個機位連拍到的好幾張煙火，疊成一張「一次放完」的照片。
//!
//! 做法就是 Photoshop 那兩個混合模式，逐像素、逐通道算：
//!
//! * **加亮**（Lighten）＝兩張取比較亮的那一個。夜空本來就是黑的，煙火是
//!   亮的，取大值等於「把每一張的煙火都留下來、夜空維持原樣」——地景不會
//!   愈疊愈亮，是煙火疊圖最常用的一種。
//! * **濾色**（Screen）＝`1-(1-a)(1-b)`，兩張的光相加但永遠不會爆掉。
//!   煙火之間互相重疊的地方比加亮更亮、也更接近真的重複曝光；代價是地景與
//!   殘煙同樣被加亮，張數一多就會整片發灰。
//!
//! 兩者都直接在 sRGB 編碼值上算（不轉線性），與 Photoshop 預設的行為一致。
//!
//! 「地景」是那張當底圖的照片：它的地面、燈火、水面倒影原封不動留著，
//! 其餘每一張只把比它亮的部分疊上去。
//!
//! 遮色片沿用去煙霧那一套形狀（見 [`dehaze::shape_weights`]），
//! 但兩種照片上的意思不同，見 [`blend`]。

use image::RgbImage;

use crate::dehaze::{self, Shape};

/// 疊圖的混合方式
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendMode {
    /// 加亮：逐通道取兩張比較亮的那個
    Lighten,
    /// 濾色：兩張的光相加（不會超過純白）
    Screen,
}

impl BlendMode {
    /// 選單順序
    pub const ALL: [BlendMode; 2] = [BlendMode::Lighten, BlendMode::Screen];

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Lighten => "加亮",
            BlendMode::Screen => "濾色",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            BlendMode::Lighten => "只留比較亮的那一個：地景不會愈疊愈亮，最保險（預設）",
            BlendMode::Screen => "兩張的光相加：煙火重疊處更亮、更像重複曝光，但地景與殘煙也會一起變亮",
        }
    }

    /// 底層與上層的單一通道混合結果
    fn mix(self, base: u8, top: u8) -> u8 {
        match self {
            BlendMode::Lighten => base.max(top),
            // 255 - (255-a)(255-b)/255；+127 是四捨五入，整數算完全不必碰浮點
            BlendMode::Screen => {
                let inv = (255 - base) as u32 * (255 - top) as u32;
                (255 - ((inv + 127) / 255)) as u8
            }
        }
    }
}

/// 一層疊上去時要怎麼擺：平移、縮放、旋轉，全部以**畫面中心**為基準。
///
/// 三個量都是**相對的**（平移是佔畫面寬高的比例、縮放是倍率、旋轉是角度），
/// 所以同一組數字套在預覽縮圖與原尺寸照片上，擺出來的構圖一模一樣
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Xform {
    /// 平移，佔畫面寬／高的比例（正值往右、往下）
    pub dx: f32,
    pub dy: f32,
    /// 縮放倍率，1.0＝原大小。**只有使用者會動它**——自動對齊不縮放，
    /// 每一層都是原尺寸擺進畫布的（見 [`blend_layer`]）
    pub scale: f32,
    /// 旋轉角度（度，順時針）
    pub rot: f32,
}

impl Default for Xform {
    fn default() -> Self {
        Self { dx: 0.0, dy: 0.0, scale: 1.0, rot: 0.0 }
    }
}

/// 縮放倍率的上下限
pub const XFORM_MIN_SCALE: f32 = 0.2;
pub const XFORM_MAX_SCALE: f32 = 3.0;

impl Xform {
    /// 完全沒動過（原位、原大小、沒轉）
    pub fn is_identity(&self) -> bool {
        self.dx == 0.0 && self.dy == 0.0 && self.scale == 1.0 && self.rot == 0.0
    }

    /// 只有整數像素的平移——這種情形不必重新取樣（見 [`blend_layer`]）
    fn is_shift_only(&self) -> bool {
        self.rot.abs() < 1e-3 && (self.scale - 1.0).abs() < 1e-4
    }

    /// 夾回合法範圍。倍率不能是 0（要拿來當除數），角度收進 ±180
    pub fn clamped(self) -> Self {
        let f = |v: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
        Self {
            dx: f(self.dx, -2.0, 2.0),
            dy: f(self.dy, -2.0, 2.0),
            scale: if self.scale.is_finite() {
                self.scale.clamp(XFORM_MIN_SCALE, XFORM_MAX_SCALE)
            } else {
                1.0
            },
            rot: f(self.rot, -180.0, 180.0),
        }
    }
}

/// 地景保護：天際線以下那一片**不要被疊亮**。
///
/// 疊圖要的是天空的煙火與水面的倒影；天際線以下每一層拍到的是同一片城市，
/// 只是各張曝光差一點——用「加亮」疊就是取每一層的最大值，整片城市會隨著
/// 張數愈疊愈亮（實測疊一層就有六成以上的點被抬亮，平均 +18～21 階）。
///
/// 但也不能整片硬擋：真的落在城市上空的煙火還是要進得來。所以判準不是
/// 位置而是**亮多少**——只有「比地景亮不了多少」的才擋掉（那就是同一片
/// 城市的曝光差），亮得夠明顯的一律放行（那是新的光）。
///
/// 水面倒影不受影響：平靜的水面是暗而平的，從畫面上緣連得下來，
/// 天際線判定會把它算進「天空」那一側（實測天際線落在海岸線／橋那一帶）
pub struct Guard<'a> {
    /// 天空權重圖（1＝天際線以上的天空與水面，0＝地景），尺寸 `w`×`h`。
    /// **與畫布不同尺寸也沒關係**，會照比例取樣
    pub sky: &'a [f32],
    pub w: usize,
    pub h: usize,
    /// 地景區要比原本亮過這個階數才放行；低於它的視為「同一片城市的曝光差」
    pub threshold: u8,
}

/// 放行門檻之上再留這麼寬的過渡帶，剛好卡在門檻上下的點才不會出現硬邊
const GUARD_RAMP: f32 = 40.0;

impl Guard<'_> {
    /// 這一點（畫布座標）**有多少比例要保留地景**：0＝照疊、1＝完全不疊。
    /// `lift` 是這一層比目前結果亮了幾階
    fn keep(&self, x: usize, y: usize, gw: usize, gh: usize, lift: i32) -> f32 {
        if self.w == 0 || self.h == 0 {
            return 0.0;
        }
        let sx = (x * self.w / gw.max(1)).min(self.w - 1);
        let sy = (y * self.h / gh.max(1)).min(self.h - 1);
        // 天空權重越低＝越是地景，要保護的成分就越高
        let land = 1.0 - self.sky[sy * self.w + sx].clamp(0.0, 1.0);
        if land <= 0.0 {
            return 0.0;
        }
        let pass = ((lift as f32 - self.threshold as f32) / GUARD_RAMP).clamp(0.0, 1.0);
        land * (1.0 - pass)
    }
}

/// 算天際線用的縮圖長邊。天際線是大尺度的東西，算細節沒有意義，
/// 而原尺寸照片攤成 f32 一張就要好幾百 MB
const GUARD_LONG: u32 = 320;

/// 地景區要亮過地景這麼多階才放行（見 [`Guard`]）。
///
/// 實測同一場煙火各張之間，地景（城市）的曝光差大約在 ±20 階以內，
/// 而真的打在城市上空的煙火動輒亮上百階，30 在兩者中間
const GUARD_LIFT: u8 = 30;

/// 從地景那張算好的天空圖，拿來借出 [`Guard`]。
///
/// 分成兩個型別是因為疊圖是一層一層跑的，這張圖只要算一次
pub struct GuardMap {
    sky: Vec<f32>,
    w: usize,
    h: usize,
}

impl GuardMap {
    /// 從地景那張算出天際線。原尺寸或縮圖都可以，結果一樣（照比例取樣）
    pub fn new(ground: &RgbImage) -> Self {
        let (sky, w, h) = dehaze::sky_weights(ground, GUARD_LONG);
        Self { sky, w, h }
    }

    pub fn guard(&self) -> Guard<'_> {
        Guard { sky: &self.sky, w: self.w, h: self.h, threshold: GUARD_LIFT }
    }
}

/// 疊在地景上的一層
pub struct Layer<'a> {
    pub img: &'a RgbImage,
    /// 這一層的遮色片：蓋到的地方**不要**疊上去（例如它自己的地景、
    /// 或這張裡拍壞的那一朵煙火）
    pub mask: &'a [Shape],
    /// 遮色片反過來用：只有**畫到的地方**才疊進來，其餘一律擋掉
    pub invert: bool,
    /// 這一層要怎麼擺（對齊機位的位移、自己想搬的位置、縮放與旋轉）
    pub xform: Xform,
}

/// 把遮色片畫成權重圖；一個形狀都沒畫就回 None。
///
/// 「沒畫」與「畫了但全是 0」在後面差很多：回 None 的那條路連每個像素一次
/// 乘法都省掉，而整張 0 的權重圖光配置就要 `w*h*4` 位元組——原尺寸照片
/// 一層就是近百 MB，二十幾層疊下來不能每層都來一次
pub fn weights(
    shapes: &[Shape],
    feather: i32,
    density: i32,
    w: usize,
    h: usize,
    invert: bool,
) -> Option<Vec<f32>> {
    if !shapes.iter().any(|s| s.cleaned().is_some()) {
        return None;
    }
    let mut v = dehaze::shape_weights(shapes, feather, density, w, h);
    if !v.iter().any(|&x| x > 0.0) {
        return None;
    }
    // 反過來選：畫到的地方才算數，其餘一律擋掉。
    // **空白時不反轉**（上面已經回 None）——一個形狀都還沒畫就把整層擋光，
    // 使用者只會看到畫面突然少了一層，不知道發生什麼事
    if invert {
        for x in v.iter_mut() {
            *x = 1.0 - *x;
        }
    }
    Some(v)
}

/// 把一層就地疊到 `acc` 上。
///
/// * `acc`：目前疊到一半的結果，也就是地景那張。**它的尺寸就是成品的尺寸**
/// * `mask`：這一層的遮色片（蓋到的地方不疊進來），座標是對這一層自己算的
/// * `xform`：這一層要怎麼擺（見 [`Xform`]）
/// * `protect`：地景保護區的權重圖（見 [`weights`]），蓋到的地方誰都疊不上去。
///   逐層存檔時整批共用同一份，不必每層重畫一次
///
/// **每一層一律以原尺寸、一個像素對一個像素擺進來，預設置中**，不會為了
/// 塞進地景的畫面而縮放。同一場煙火各自在 Lightroom 裁過的輸出，尺寸與
/// 長寬比都不同，但它們是同一張感光元件影像的不同裁切——同一個景物點在
/// 兩張裡相距的像素數是一樣的，所以「原尺寸擺進來再平移」才對得準。
/// 先縮到同樣大小反而會把那個對應關係破壞掉（那正是之前對不齊的原因）。
/// 比地景大的部分超出畫布就切掉，小的部分四周維持地景。
///
/// 就地改而不是回傳新的一張：原尺寸的照片一張就要幾十 MB，
/// 每層複製一份等於白白多吃一倍記憶體、也多跑一趟複製
pub fn blend_layer(
    acc: &mut RgbImage,
    top: &RgbImage,
    mask: &[Shape],
    invert: bool,
    xform: Xform,
    protect: Option<&[f32]>,
    guard: Option<&Guard>,
    feather: i32,
    density: i32,
    mode: BlendMode,
) {
    let (gw, gh) = acc.dimensions();
    let (lw, lh) = top.dimensions();
    if gw == 0 || gh == 0 || lw == 0 || lh == 0 {
        return;
    }
    let (gwu, ghu) = (gw as usize, gh as usize);
    let (lwu, lhu) = (lw as usize, lh as usize);
    // 遮色片畫在**這一層自己**的尺寸上（形狀的相對座標是對它算的）
    let ex = weights(mask, feather, density, lwu, lhu, invert);
    let src = top.as_raw();
    let dst = acc.as_mut();
    let x = xform.clamped();

    // 把一點疊上去：`(px, y)` 是畫布座標、`si` 是這一層對應的取樣位置、
    // `t` 是那裡的顏色
    let mut put = |px: usize, y: usize, si: usize, t: [u8; 3]| {
        let i = y * gwu + px;
        // 這一點要疊進去多少：層自己被遮掉的、地景保護住的，兩邊都要扣。
        // 遮色片跟著這一層一起走，地景的保護區則固定在畫面上不動
        let mut k = match (&ex, protect) {
            (None, None) => 1.0,
            (Some(e), None) => 1.0 - e[si],
            (None, Some(p)) => 1.0 - p[i],
            (Some(e), Some(p)) => (1.0 - e[si]) * (1.0 - p[i]),
        };
        if k <= 0.002 {
            return;
        }
        // 天際線以下只放行明顯亮過地景的（真的煙火），其餘擋掉，
        // 免得整片城市被一層層抬亮（見 Guard）
        if let Some(g) = guard {
            let lift = (0..3)
                .map(|c| t[c] as i32 - dst[i * 3 + c] as i32)
                .max()
                .unwrap_or(0);
            k *= 1.0 - g.keep(px, y, gwu, ghu, lift);
            if k <= 0.002 {
                return;
            }
        }
        for c in 0..3 {
            let b = dst[i * 3 + c];
            let mixed = mode.mix(b, t[c]);
            // 羽化帶上照權重在「原樣」與「疊上去」之間過渡
            dst[i * 3 + c] = if k >= 0.998 {
                mixed
            } else {
                (b as f32 + (mixed as f32 - b as f32) * k).round() as u8
            };
        }
    };

    // 置中擺放的基準位移：這一層的中心對到畫布的中心
    let base_x = (lw as i64 - gw as i64) / 2;
    let base_y = (lh as i64 - gh as i64) / 2;

    if x.is_shift_only() {
        // 沒有縮放與旋轉就整數平移：不必重新取樣，煙火的細線一點都不會糊。
        // 對齊機位的位移正是這種情形，也是最常走到的一條
        let ox = (x.dx * gw as f32).round() as i64;
        let oy = (x.dy * gh as f32).round() as i64;
        for y in 0..ghu {
            let sy = y as i64 - oy + base_y;
            if sy < 0 || sy >= lhu as i64 {
                continue;
            }
            let srow = sy as usize * lwu;
            for px in 0..gwu {
                let sx = px as i64 - ox + base_x;
                if sx < 0 || sx >= lwu as i64 {
                    continue;
                }
                let si = srow + sx as usize;
                let t = [src[si * 3], src[si * 3 + 1], src[si * 3 + 2]];
                put(px, y, si, t);
            }
        }
        return;
    }

    // 使用者調過縮放或旋轉：反算回這一層的座標再雙線性取樣。
    // 反算（而不是正著把來源推到輸出）才不會在輸出上留下沒被寫到的洞
    let (gcx, gcy) = (gw as f32 / 2.0, gh as f32 / 2.0);
    let (lcx, lcy) = (lw as f32 / 2.0, lh as f32 / 2.0);
    let a = x.rot.to_radians();
    let (sin, cos) = a.sin_cos();
    let inv = 1.0 / x.scale;
    for y in 0..ghu {
        for px in 0..gwu {
            // 先移到「以畫布中心為原點」，去掉平移、旋轉與縮放，
            // 再換到這一層自己的像素座標（層中心對畫布中心）
            let u = px as f32 + 0.5 - gcx - x.dx * gw as f32;
            let v = y as f32 + 0.5 - gcy - x.dy * gh as f32;
            let su = (u * cos + v * sin) * inv + lcx - 0.5;
            let sv = (-u * sin + v * cos) * inv + lcy - 0.5;
            let Some(t) = sample_rgb(src, lwu, lhu, su, sv) else {
                continue;
            };
            // 遮色片取最近的一點就好：它的邊界本來就抹柔過了
            let mx = (su.round() as i64).clamp(0, lwu as i64 - 1) as usize;
            let my = (sv.round() as i64).clamp(0, lhu as i64 - 1) as usize;
            put(px, y, my * lwu + mx, t);
        }
    }
}

/// 在 `w`×`h` 的 RGB 緩衝上做雙線性取樣；落在影像外就回 None
fn sample_rgb(src: &[u8], w: usize, h: usize, x: f32, y: f32) -> Option<[u8; 3]> {
    if !(x > -0.5 && y > -0.5) || x > w as f32 - 0.5 || y > h as f32 - 0.5 {
        return None;
    }
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |ix: i64, iy: i64, c: usize| -> f32 {
        let ix = ix.clamp(0, w as i64 - 1) as usize;
        let iy = iy.clamp(0, h as i64 - 1) as usize;
        src[(iy * w + ix) * 3 + c] as f32
    };
    let mut out = [0u8; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let top = at(x0, y0, c) + (at(x0 + 1, y0, c) - at(x0, y0, c)) * fx;
        let bot = at(x0, y0 + 1, c) + (at(x0 + 1, y0 + 1, c) - at(x0, y0 + 1, c)) * fx;
        *o = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
    }
    Some(out)
}

/// 把 `layers` 依序疊到 `ground` 上，回傳新的一張。
///
/// * `ground`：當地景的那張，**輸出的尺寸就是它的尺寸**
/// * `protect`：畫在**地景**上的遮色片。蓋到的地方誰都疊不上去——想保住
///   地景的樹梢、招牌、水面倒影不被別張的煙火穿過去時用它
/// * `feather`：遮色片邊緣的羽化寬度 0~100（與去煙霧同一個尺規）
/// * `guard`：天際線以下不要被疊亮（見 [`Guard`]）；不要就給 None
///
/// 全部的層都已經在記憶體裡時用這個（預覽就是這樣）；原尺寸存檔是一層一層
/// 讀進來的，走 [`blend_layer`] 那條。一層都沒有時就是把地景原樣複製一份
pub fn blend(
    ground: &RgbImage,
    protect: &[Shape],
    protect_invert: bool,
    feather: i32,
    density: i32,
    layers: &[Layer],
    mode: BlendMode,
    guard: Option<&Guard>,
) -> RgbImage {
    let mut out = ground.clone();
    let (w, h) = (ground.width() as usize, ground.height() as usize);
    let prot = weights(protect, feather, density, w, h, protect_invert);
    for layer in layers {
        blend_layer(
            &mut out,
            layer.img,
            layer.mask,
            layer.invert,
            layer.xform,
            prot.as_deref(),
            guard,
            feather,
            density,
            mode,
        );
    }
    out
}

/// 把**地景自己的擺法**套到畫布上（就地改）。
///
/// 地景是底圖，畫布的尺寸就是它的尺寸；它自己要搬、要縮、要轉的時候，動的是
/// 「它的內容在畫布裡的位置」——挪開之後空出來的邊是黑的（那底下本來就沒有
/// 別的東西）。沒動過就完全不碰，一次取樣都不做。
///
/// 其他層的擺法是疊上去的當下才算的（見 [`blend_layer`]），只有地景要先擺好，
/// 因為它同時就是畫布本身
pub fn place(img: &mut RgbImage, xform: Xform) {
    let x = xform.clamped();
    if x.is_identity() {
        return;
    }
    let src = img.clone();
    img.as_mut().fill(0);
    // 疊到全黑的畫布上，加亮＝原樣照抄，等於「把它擺到新位置」
    // 沒有遮色片，濃度給多少都一樣
    blend_layer(img, &src, &[], false, x, None, None, 0, 100, BlendMode::Lighten);
}

/// 對齊時第一輪（大範圍粗找）用的長邊
const ALIGN_COARSE: u32 = 160;

/// 對齊時第二輪（小範圍細修）用的長邊
const ALIGN_FINE: u32 = 640;

/// 逐點差值算到這裡就封頂。
///
/// 兩張之間**差最多的正是煙火**（這一張有、那一張沒有），不封頂的話代價
/// 完全被煙火主導，對齊就變成「讓兩張的煙火重疊」，那是錯的目標。
/// 封頂之後煙火不管挪到哪都貢獻差不多的常數，真正決定勝負的是城市燈火、
/// 橋樑、天際線那些**兩張都有**的東西——也就是我們要對齊的東西
const ALIGN_CLIP: i32 = 40;

/// 平移的搜尋範圍，佔畫面的比例。各自裁過的照片，構圖中心可以差上一截
const ALIGN_SEARCH: f32 = 0.14;

/// 最好的那一組要比「典型的一組」好上這麼多，才算真的對上了。
///
/// 用相對值而不是絕對門檻：夜景大半是黑的，絕對代價本來就低，拿絕對值當
/// 門檻會把明明對不上的也放行。真的對上時最佳解會明顯比隨便挑的一組好；
/// 對不上時整片都差不多平
const ALIGN_MIN_GAIN: f32 = 0.90;

/// 估出「這一層要平移多少才會對準地景」；**對不上就回 None**
/// （呼叫端會把那一層整個略過，不要硬疊上去）。
///
/// 只找平移，不找縮放：疊圖時每一層都是**原尺寸**擺進畫布的
/// （見 [`blend_layer`]），同一個景物點在兩張裡相距的像素數本來就一樣，
/// 對齊要做的就只是把它推到對的位置。縮放留給使用者疊完之後自己調。
///
/// 比對的是**邊緣強度**而不是亮度（見 [`edge_pair`]）：兩張的曝光與白平衡
/// 常常不一樣（尤其是修過的版本），比亮度會被那個全域差值淹沒；邊緣只認
/// 結構，城市燈火、橋樑、天際線的輪廓在兩張裡都一樣。
///
/// 搜尋分兩輪，先在小圖上大範圍找、再在大圖上就近修
pub fn align_offset(ground: &RgbImage, layer: &RgbImage) -> Option<Xform> {
    let (gc, gs, lc, ls) = edge_pair(ground, layer, ALIGN_COARSE)?;
    let r = (gs.0.max(gs.1) as f32 * ALIGN_SEARCH).round().max(6.0) as i32;
    let coarse = search(&gc, gs, &lc, ls, (-r, r, 1), (-r, r, 1), 2);
    // 對不上就別再細修了，兩種情形都算對不上：最佳解沒有明顯比典型的一組好
    // （怎麼擺都一樣爛），或最佳解頂在搜尋範圍的邊界上（還沒找到谷底）
    if coarse.best >= coarse.typical * ALIGN_MIN_GAIN
        || coarse.dx.abs() >= r
        || coarse.dy.abs() >= r
    {
        return None;
    }
    let Some((gf, gsf, lf, lsf)) = edge_pair(ground, layer, ALIGN_FINE) else {
        return Some(Xform {
            dx: coarse.dx as f32 / gs.0 as f32,
            dy: coarse.dy as f32 / gs.1 as f32,
            ..Default::default()
        });
    };
    // 換到細的那一層：粗找的結果照比例放大當起點，再在一格的範圍內修
    let k = gsf.0 as f32 / gs.0 as f32;
    let (sx, sy) = (
        (coarse.dx as f32 * k).round() as i32,
        (coarse.dy as f32 * k).round() as i32,
    );
    let step = k.ceil() as i32 + 1;
    let fine = search(
        &gf,
        gsf,
        &lf,
        lsf,
        (sx - step, sx + step, 1),
        (sy - step, sy + step, 1),
        2,
    );
    let plain = Xform {
        dx: fine.dx as f32 / gsf.0 as f32,
        dy: fine.dy as f32 / gsf.1 as f32,
        ..Default::default()
    };
    // 再量一次水平（見 refine_rotation）：腳架多少會有一點滾動，
    // 每張又可能在 Lightroom 各自拉直過，差個零點幾度，原尺寸下兩邊
    // 就差了幾十個像素
    Some(refine_rotation(ground, layer, &fine, gsf).unwrap_or(plain))
}

/// 對齊時量水平（旋轉）用的長邊。
///
/// 角度非得在**大圖**上量不可：0.4° 在 160px 的小圖上只讓邊緣位移半個像素，
/// 根本量不出來；要到上千像素才有足夠的槓桿
const ALIGN_ROT: u32 = 1600;

/// 角度大於這個值就當作量錯了（同機位的照片不會差這麼多）
const ALIGN_MAX_ROT: f32 = 3.0;

/// 小於這個角度就當作沒轉——留在 0 才能走「整數平移、不重新取樣」那條路，
/// 煙火的細線一點都不會糊
const ALIGN_ROT_DEADZONE: f32 = 0.02;

/// 量出兩張之間的**水平差**，順便把平移一起修準。
///
/// 作法是**兩點測角**：在畫面左右兩側各挑一塊有結構的區域（城市燈火、
/// 橋樑那種），各自量它自己的局部位移，再從「兩塊的垂直位移差 ÷ 兩塊的
/// 水平距離」反推角度。基線拉得越長角度就越準，比在整張上多搜一個角度
/// 維度便宜太多，也準得多。
///
/// 局部位移用拋物線內插取到次像素，1600px 上兩塊相距約 800px 時，
/// 角度解析度可以到 0.03° 上下。
///
/// 挑不到夠有結構的兩塊、或兩塊量出來的結果彼此矛盾時回 None（維持不轉）
fn refine_rotation(
    ground: &RgbImage,
    layer: &RgbImage,
    fine: &Fit,
    fine_size: (usize, usize),
) -> Option<Xform> {
    let (g, gs, l, ls) = edge_pair(ground, layer, ALIGN_ROT)?;
    // 把細修階段的平移換算到這一層的解析度當起點
    let k = gs.0 as f32 / fine_size.0 as f32;
    let seed = (
        (fine.dx as f32 * k).round() as i32,
        (fine.dy as f32 * k).round() as i32,
    );
    // 一塊的半徑與搜尋範圍：塊要夠大才咬得住結構，範圍蓋得住換解析度的誤差
    let half = (gs.0.min(gs.1) / 6).clamp(60, 260) as i32;
    let range = (k.ceil() as i32 + 8).max(10);

    let (cx, cy) = (gs.0 as f32 / 2.0, gs.1 as f32 / 2.0);
    let base_x = (ls.0 as i64 - gs.0 as i64) / 2;
    let base_y = (ls.1 as i64 - gs.1 as i64) / 2;

    // 左右各挑一塊最有結構的：夜空是黑的，量不出東西，得挑城市那一帶
    let left = pick_patch(&g, gs, half, 0.05, 0.40)?;
    let right = pick_patch(&g, gs, half, 0.60, 0.95)?;
    // 基線太短角度就沒有解析度可言
    if (right.0 - left.0) < (gs.0 as i32) / 4 {
        return None;
    }

    let ml = patch_shift(&g, gs, &l, ls, left, half, seed, range, base_x, base_y)?;
    let mr = patch_shift(&g, gs, &l, ls, right, half, seed, range, base_x, base_y)?;

    // ay(x) = DY + (x - cx)·θ　→　θ 由兩塊的垂直位移差算出來
    let dxp = right.0 as f32 - left.0 as f32;
    let theta = (mr.1 - ml.1) / dxp;
    let deg = theta.to_degrees();
    if !deg.is_finite() || deg.abs() > ALIGN_MAX_ROT {
        return None;
    }
    let deg = if deg.abs() < ALIGN_ROT_DEADZONE { 0.0 } else { deg };
    let theta = deg.to_radians();

    // 反推整體平移：ax = DX - (y-cy)·θ、ay = DY + (x-cx)·θ，兩塊取平均
    let dxs = ((ml.0 + (left.1 as f32 - cy) * theta) + (mr.0 + (right.1 as f32 - cy) * theta)) / 2.0;
    let dys = ((ml.1 - (left.0 as f32 - cx) * theta) + (mr.1 - (right.0 as f32 - cx) * theta)) / 2.0;
    Some(Xform {
        dx: dxs / gs.0 as f32,
        dy: dys / gs.1 as f32,
        scale: 1.0,
        rot: deg,
    })
}

/// 在 `[lo, hi]`（佔寬度的比例）這一段裡挑一塊邊緣最多的區域當測量點，
/// 回傳它的中心。整段都平坦（例如全是夜空）就回 None
fn pick_patch(
    g: &[u8],
    gs: (usize, usize),
    half: i32,
    lo: f32,
    hi: f32,
) -> Option<(i32, i32)> {
    let x0 = (gs.0 as f32 * lo) as i32 + half;
    let x1 = (gs.0 as f32 * hi) as i32 - half;
    if x1 <= x0 {
        return None;
    }
    let (mut best, mut best_e) = ((0, 0), 0u64);
    let stride = (half / 2).max(8);
    let mut cy = half;
    while cy + half < gs.1 as i32 {
        let mut cx = x0;
        while cx <= x1 {
            let mut e = 0u64;
            let mut y = cy - half;
            while y < cy + half {
                let row = y as usize * gs.0;
                let mut x = cx - half;
                while x < cx + half {
                    e += g[row + x as usize] as u64;
                    x += 3;
                }
                y += 3;
            }
            if e > best_e {
                best_e = e;
                best = (cx, cy);
            }
            cx += stride;
        }
        cy += stride;
    }
    // 太平坦就別拿它測角，量出來的是雜訊
    let area = ((2 * half / 3) * (2 * half / 3)) as u64;
    (best_e > area * 12).then_some(best)
}

/// 量某一塊區域自己的局部位移（次像素）。回傳 (ax, ay)
#[allow(clippy::too_many_arguments)]
fn patch_shift(
    g: &[u8],
    gs: (usize, usize),
    l: &[u8],
    ls: (usize, usize),
    centre: (i32, i32),
    half: i32,
    seed: (i32, i32),
    range: i32,
    base_x: i64,
    base_y: i64,
) -> Option<(f32, f32)> {
    let n = (range * 2 + 1) as usize;
    let mut costs = vec![f32::MAX; n * n];
    let (mut best, mut best_i) = (f32::MAX, 0usize);
    for (iy, ay) in (seed.1 - range..=seed.1 + range).enumerate() {
        for (ix, ax) in (seed.0 - range..=seed.0 + range).enumerate() {
            let mut sum = 0u64;
            let mut cnt = 0u64;
            let mut y = centre.1 - half;
            while y < centre.1 + half {
                let sy = y as i64 - ay as i64 + base_y;
                if y >= 0 && (y as usize) < gs.1 && sy >= 0 && sy < ls.1 as i64 {
                    let row = y as usize * gs.0;
                    let srow = sy as usize * ls.0;
                    let mut x = centre.0 - half;
                    while x < centre.0 + half {
                        let sx = x as i64 - ax as i64 + base_x;
                        if x >= 0 && (x as usize) < gs.0 && sx >= 0 && sx < ls.0 as i64 {
                            let d = (g[row + x as usize] as i32 - l[srow + sx as usize] as i32).abs();
                            sum += d.min(ALIGN_CLIP) as u64;
                            cnt += 1;
                        }
                        x += 2;
                    }
                }
                y += 2;
            }
            if cnt == 0 {
                continue;
            }
            let avg = sum as f32 / cnt as f32;
            costs[iy * n + ix] = avg;
            if avg < best {
                best = avg;
                best_i = iy * n + ix;
            }
        }
    }
    if best == f32::MAX {
        return None;
    }
    let (bi, bj) = (best_i / n, best_i % n);
    // 頂在搜尋範圍邊界就不算數：真的谷底兩側都要有東西才內插得出來
    if bi == 0 || bj == 0 || bi + 1 >= n || bj + 1 >= n {
        return None;
    }
    // 拋物線內插取次像素：角度的解析度全靠這一步
    let sub = |a: f32, b: f32, c: f32| -> f32 {
        let d = a - 2.0 * b + c;
        if d.abs() < 1e-6 {
            0.0
        } else {
            ((a - c) / (2.0 * d)).clamp(-1.0, 1.0)
        }
    };
    let ox = sub(costs[bi * n + bj - 1], best, costs[bi * n + bj + 1]);
    let oy = sub(costs[(bi - 1) * n + bj], best, costs[(bi + 1) * n + bj]);
    Some((
        (seed.0 - range + bj as i32) as f32 + ox,
        (seed.1 - range + bi as i32) as f32 + oy,
    ))
}

/// 搜尋結果：最好的那個平移、它的代價，以及「典型」代價（所有候選的中位數）
struct Fit {
    dx: i32,
    dy: i32,
    best: f32,
    typical: f32,
}

/// 在給定範圍內找代價最小的平移。
///
/// 取樣方式與 [`blend_layer`] 疊圖時完全一致（原尺寸置中再平移），
/// 對齊算出來的那一組拿去疊才會真的對上。`step` 是取樣間隔
fn search(
    g: &[u8],
    gs: (usize, usize),
    l: &[u8],
    ls: (usize, usize),
    rx: (i32, i32, i32),
    ry: (i32, i32, i32),
    step: usize,
) -> Fit {
    let step = step.max(1);
    // 與 blend_layer 同一個置中基準
    let base_x = (ls.0 as i64 - gs.0 as i64) / 2;
    let base_y = (ls.1 as i64 - gs.1 as i64) / 2;
    // 重疊面積至少要有「比較小的那張」的一半，否則挪到只剩一角反而代價最低
    let sampled = |n: usize| (n + step - 1) / step;
    let need = (sampled(gs.0.min(ls.0)) * sampled(gs.1.min(ls.1))) as u64 / 2;
    let mut best = Fit { dx: 0, dy: 0, best: f32::MAX, typical: f32::MAX };
    let mut costs: Vec<f32> = Vec::new();
    let mut dy = ry.0;
    while dy <= ry.1 {
        let mut dx = rx.0;
        while dx <= rx.1 {
            let mut sum = 0u64;
            let mut n = 0u64;
            let mut y = 0usize;
            while y < gs.1 {
                let sy = y as i64 - dy as i64 + base_y;
                if sy >= 0 && sy < ls.1 as i64 {
                    let srow = sy as usize * ls.0;
                    let row = y * gs.0;
                    let mut x = 0usize;
                    while x < gs.0 {
                        let sx = x as i64 - dx as i64 + base_x;
                        if sx >= 0 && sx < ls.0 as i64 {
                            let d = (g[row + x] as i32 - l[srow + sx as usize] as i32).abs();
                            sum += d.min(ALIGN_CLIP) as u64;
                            n += 1;
                        }
                        x += step;
                    }
                }
                y += step;
            }
            if n >= need && n > 0 {
                let avg = sum as f32 / n as f32;
                costs.push(avg);
                if avg < best.best {
                    best = Fit { dx, dy, best: avg, typical: f32::MAX };
                }
            }
            dx += rx.2.max(1);
        }
        dy += ry.2.max(1);
    }
    // 「典型」取中位數：拿它跟最佳解比，才看得出這個最小值是不是真的谷底
    if !costs.is_empty() {
        costs.sort_by(f32::total_cmp);
        best.typical = costs[costs.len() / 2];
    }
    best
}

/// 把兩張**照同一個比例**縮小並轉成邊緣強度圖，回傳（地景圖, 地景尺寸,
/// 這一層的圖, 這一層的尺寸）。
///
/// 兩張各自保留原本的長寬比與相對大小（只是同時縮小），疊圖時就是這樣擺的；
/// 縮到同樣大小會把「一個像素對一個像素」的對應關係破壞掉。
///
/// 轉邊緣是刻意的：兩張的曝光與白平衡常常不一樣（修過的版本尤其明顯），
/// 直接比亮度會被那個全域差值淹沒；邊緣只認結構，對亮度偏移免疫。
/// 太小就對不出東西，回 None
#[allow(clippy::type_complexity)]
fn edge_pair(
    ground: &RgbImage,
    layer: &RgbImage,
    long: u32,
) -> Option<(Vec<u8>, (usize, usize), Vec<u8>, (usize, usize))> {
    let (gw, gh) = ground.dimensions();
    if gw == 0 || gh == 0 {
        return None;
    }
    // 同一個縮小比例套在兩張上，相對大小才留得住
    let k = long as f32 / gw.max(gh) as f32;
    let edges = |img: &RgbImage| -> Option<(Vec<u8>, (usize, usize))> {
        let w = ((img.width() as f32 * k).round() as u32).max(1);
        let h = ((img.height() as f32 * k).round() as u32).max(1);
        if w < 24 || h < 24 {
            return None;
        }
        let small = image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle);
        let (wu, hu) = (w as usize, h as usize);
        let gray: Vec<i32> = small
            .pixels()
            .map(|p| {
                // BT.601，與別處的亮度係數同一組
                (p.0[0] as i32 * 299 + p.0[1] as i32 * 587 + p.0[2] as i32 * 114) / 1000
            })
            .collect();
        let mut out = vec![0u8; wu * hu];
        for y in 1..hu.saturating_sub(1) {
            for x in 1..wu.saturating_sub(1) {
                let i = y * wu + x;
                // 中央差分：夠用又便宜，不必動用 Sobel
                let dx = gray[i + 1] - gray[i - 1];
                let dy = gray[i + wu] - gray[i - wu];
                out[i] = (dx.abs() + dy.abs()).min(255) as u8;
            }
        }
        Some((out, (wu, hu)))
    };
    let (g, gs) = edges(ground)?;
    let (l, ls) = edges(layer)?;
    Some((g, gs, l, ls))
}
#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, image::Rgb(c))
    }

    /// 預設是「選擇疊圖的區域」（反選），但**一個形狀都還沒畫**時整層仍要照疊——
    /// 地景的保護區也一樣。否則一切到專業模式，畫面就會莫名少掉所有層
    #[test]
    fn empty_mask_still_blends_even_when_inverted() {
        let g = solid(4, 4, [10, 10, 10]);
        let l = solid(4, 4, [200, 200, 200]);
        let layer = |inv: bool| Layer {
            img: &l,
            mask: &[],
            invert: inv,
            xform: Xform::default(),
        };
        // 層自己反選、地景的保護區也反選，兩邊都沒畫東西
        let out = blend(&g, &[], true, 25, 100, &[layer(true)], BlendMode::Lighten, None);
        assert_eq!(out.get_pixel(0, 0).0, [200, 200, 200], "反選＋空遮色片把整層擋光了");
        // 與不反選的結果要一模一樣
        let base = blend(&g, &[], false, 25, 100, &[layer(false)], BlendMode::Lighten, None);
        assert_eq!(out.get_pixel(2, 2).0, base.get_pixel(2, 2).0);
    }

    /// 地景的保護區是**全域**的：反選之後只有圈起來的地方讓別層疊得上來，
    /// 其餘一律擋掉。這正是為什麼地景的預設不能跟著一般的層一起反選
    /// （見 `StackTool::mask_inverted`）——地景上隨手畫一個形狀，
    /// 整張就會變成「沒有疊圖」
    #[test]
    fn inverted_ground_mask_blocks_everything_outside_it() {
        let g = solid(100, 100, [10, 10, 10]);
        let l = solid(100, 100, [200, 200, 200]);
        let spot = [Shape::Rect(dehaze::Region {
            x0: 0.4,
            y0: 0.4,
            x1: 0.6,
            y1: 0.6,
        })];
        let layer = Layer {
            img: &l,
            mask: &[],
            invert: true,
            xform: Xform::default(),
        };
        let out = blend(&g, &spot, true, 0, 100, &[layer], BlendMode::Lighten, None);
        assert_eq!(out.get_pixel(50, 50).0, [200, 200, 200], "圈起來的地方該疊上來");
        assert_eq!(out.get_pixel(5, 5).0, [10, 10, 10], "圈外該被擋住");
    }

    #[test]
    fn lighten_takes_the_brighter_channel() {
        let g = solid(4, 4, [10, 200, 30]);
        let l = solid(4, 4, [90, 40, 30]);
        let out = blend(&g, &[], false, 25, 100, &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }], BlendMode::Lighten, None);
        assert_eq!(out.get_pixel(0, 0).0, [90, 200, 30]);
    }

    #[test]
    fn screen_adds_light_without_clipping() {
        let g = solid(4, 4, [0, 128, 255]);
        let l = solid(4, 4, [0, 128, 255]);
        let out = blend(&g, &[], false, 25, 100, &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }], BlendMode::Screen, None);
        let p = out.get_pixel(0, 0).0;
        assert_eq!(p[0], 0, "全黑疊全黑仍是全黑");
        assert!((191..=193).contains(&p[1]), "128 疊 128 約 192，實得 {}", p[1]);
        assert_eq!(p[2], 255, "純白不會溢位");
    }

    #[test]
    /// 比地景小的層以**原尺寸置中**擺進來，不會被拉大填滿畫面；
    /// 四周沒有來源的地方維持地景
    fn smaller_layer_sits_centred_at_its_own_size() {
        let g = solid(8, 6, [0, 0, 0]);
        let l = solid(4, 2, [200, 200, 200]);
        let out = blend(
            &g,
            &[],
            false,
            25, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.dimensions(), (8, 6), "成品的尺寸就是地景的尺寸");
        // 4×2 置中在 8×6 上＝x 2..6、y 2..4
        assert_eq!(out.get_pixel(3, 2).0, [200, 200, 200], "中間那塊有東西");
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0], "四周維持地景");
        assert_eq!(out.get_pixel(7, 5).0, [0, 0, 0], "沒有被拉大填滿");
    }

    /// 比地景大的層一樣以原尺寸置中，超出畫布的部分切掉
    #[test]
    fn bigger_layer_is_cropped_by_the_canvas() {
        let g = solid(4, 4, [0, 0, 0]);
        let mut l = solid(8, 8, [0, 0, 0]);
        // 只把來源正中央那 4×4 塗白：置中之後應該剛好填滿畫布
        for y in 2..6 {
            for x in 2..6 {
                l.put_pixel(x, y, image::Rgb([200, 200, 200]));
            }
        }
        let out = blend(
            &g,
            &[],
            false,
            25, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.dimensions(), (4, 4));
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get_pixel(x, y).0, [200, 200, 200], "({x},{y})");
            }
        }
    }

    #[test]
    fn layer_mask_keeps_that_area_out() {
        let g = solid(100, 100, [0, 0, 0]);
        let l = solid(100, 100, [255, 255, 255]);
        // 左半邊遮掉（羽化 0，邊界才是硬的，好斷言）
        let mask = vec![Shape::Rect(dehaze::Region { x0: 0.0, y0: 0.0, x1: 0.5, y1: 1.0 })];
        let out = blend(&g, &[], false, 0, 100, &[Layer { img: &l, mask: &mask, invert: false, xform: Xform::default() }], BlendMode::Lighten, None);
        assert_eq!(out.get_pixel(10, 50).0, [0, 0, 0], "遮住的地方維持地景");
        assert_eq!(out.get_pixel(90, 50).0, [255, 255, 255], "沒遮的地方照疊");
    }

    /// 反過來選：畫到的地方才疊，其餘一律不疊（與上面那個剛好相反）
    #[test]
    fn an_inverted_layer_mask_keeps_only_that_area() {
        let g = solid(100, 100, [0, 0, 0]);
        let l = solid(100, 100, [255, 255, 255]);
        let mask = vec![Shape::Rect(dehaze::Region { x0: 0.0, y0: 0.0, x1: 0.5, y1: 1.0 })];
        let out = blend(
            &g,
            &[],
            false,
            0, 100,
            &[Layer { img: &l, mask: &mask, invert: true, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(10, 50).0, [255, 255, 255], "圈起來的才疊");
        assert_eq!(out.get_pixel(90, 50).0, [0, 0, 0], "圈外一律不疊");
    }

    /// 反選但一個形狀都沒畫＝**整張照疊**，不是整層消失。
    /// 反過來的話，勾了「選擇疊圖的區域」畫面就會莫名少一層
    #[test]
    fn an_inverted_but_empty_mask_changes_nothing() {
        let g = solid(20, 20, [0, 0, 0]);
        let l = solid(20, 20, [200, 200, 200]);
        let out = blend(
            &g,
            &[],
            false,
            0, 100,
            &[Layer { img: &l, mask: &[], invert: true, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(10, 10).0, [200, 200, 200]);
    }

    /// 地景的保護區也能反過來用：只有圈起來的地方讓別張疊上來
    #[test]
    fn an_inverted_ground_mask_only_lets_that_area_through() {
        let g = solid(100, 100, [0, 0, 0]);
        let l = solid(100, 100, [255, 255, 255]);
        let prot = vec![Shape::Rect(dehaze::Region { x0: 0.0, y0: 0.0, x1: 0.5, y1: 1.0 })];
        let out = blend(
            &g,
            &prot,
            true,
            0, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(10, 50).0, [255, 255, 255], "圈起來的才讓人疊上來");
        assert_eq!(out.get_pixel(90, 50).0, [0, 0, 0], "其餘的地景全保住");
    }

    #[test]
    fn ground_protection_blocks_every_layer() {
        let g = solid(100, 100, [0, 0, 0]);
        let a = solid(100, 100, [255, 255, 255]);
        let b = solid(100, 100, [200, 200, 200]);
        let prot = vec![Shape::Rect(dehaze::Region { x0: 0.0, y0: 0.5, x1: 1.0, y1: 1.0 })];
        let out = blend(
            &g,
            &prot,
            false,
            0, 100,
            &[Layer { img: &a, mask: &[], invert: false, xform: Xform::default() }, Layer { img: &b, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(50, 90).0, [0, 0, 0], "保護區兩層都疊不進來");
        assert_eq!(out.get_pixel(50, 10).0, [255, 255, 255]);
    }

    /// 上半天空、下半地景的權重圖。**故意手寫**而不去跑天際線判定：
    /// 這裡要驗的是「擋不擋」的規則，不是天際線找得準不準
    const HALF_SKY: [f32; 2] = [1.0, 0.0];

    fn half_guard() -> Guard<'static> {
        Guard { sky: &HALF_SKY, w: 1, h: 2, threshold: GUARD_LIFT }
    }

    /// 地景那一半只是曝光差一點（亮不到門檻）＝同一片地景，擋掉
    #[test]
    fn land_shrugs_off_a_mere_exposure_difference() {
        let g = solid(100, 100, [40, 40, 40]);
        let l = solid(100, 100, [55, 55, 55]);
        let out = blend(
            &g,
            &[],
            false,
            0, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            Some(&half_guard()),
        );
        assert_eq!(out.get_pixel(50, 90).0, [40, 40, 40], "天際線以下不疊");
        assert_eq!(out.get_pixel(50, 10).0, [55, 55, 55], "天空照疊");
    }

    /// 真的打在地景上空的煙火亮得夠明顯，照樣疊得進來
    #[test]
    fn a_burst_over_the_land_still_gets_through() {
        let g = solid(100, 100, [40, 40, 40]);
        let l = solid(100, 100, [240, 240, 240]);
        let out = blend(
            &g,
            &[],
            false,
            0, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            Some(&half_guard()),
        );
        assert_eq!(out.get_pixel(50, 90).0, [240, 240, 240], "地景上空的煙火要進得來");
        assert_eq!(out.get_pixel(50, 10).0, [240, 240, 240]);
    }

    /// 一層層疊下去，地景不會被抬亮——這正是這個功能要解決的事
    #[test]
    fn land_does_not_creep_up_layer_after_layer() {
        let g = solid(40, 40, [40, 40, 40]);
        let layers: Vec<RgbImage> = (0..8).map(|i| solid(40, 40, [50 + i, 50 + i, 50 + i])).collect();
        let refs: Vec<Layer> = layers
            .iter()
            .map(|img| Layer { img, mask: &[], invert: false, xform: Xform::default() })
            .collect();
        let out = blend(&g, &[], false, 0, 100, &refs, BlendMode::Screen, Some(&half_guard()));
        assert_eq!(out.get_pixel(20, 30).0, [40, 40, 40], "疊了八層地景還是原樣");
        assert!(out.get_pixel(20, 10).0[0] > 200, "天空該被八層濾色疊亮");
    }

    /// 地景也搬得動：搬的是它的內容在畫布裡的位置，讓開的地方是黑的
    #[test]
    fn the_ground_moves_inside_its_own_canvas() {
        let mut g = solid(100, 100, [90, 90, 90]);
        place(&mut g, Xform { dx: 0.5, ..Default::default() });
        assert_eq!(g.dimensions(), (100, 100), "畫布尺寸不變");
        assert_eq!(g.get_pixel(10, 50).0, [0, 0, 0], "讓開的地方是黑的");
        assert_eq!(g.get_pixel(80, 50).0, [90, 90, 90]);
    }

    /// 沒動過的地景一個位元都不碰（原尺寸存檔時這條路最常走到）
    #[test]
    fn an_unmoved_ground_is_left_alone() {
        let mut g = solid(16, 16, [7, 8, 9]);
        let before = g.clone();
        place(&mut g, Xform::default());
        assert_eq!(g, before);
    }

    /// 關掉保護就是原本的行為（拿來對照上面那個）
    #[test]
    fn without_the_guard_the_land_brightens() {
        let g = solid(100, 100, [40, 40, 40]);
        let l = solid(100, 100, [55, 55, 55]);
        let out = blend(
            &g,
            &[],
            false,
            0, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform::default() }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(50, 90).0, [55, 55, 55]);
    }

    /// 位移把整層挪過去；挪出畫面的那一塊沒有東西可疊，地景維持原樣
    #[test]
    fn layer_offset_shifts_what_gets_blended() {
        let g = solid(100, 100, [0, 0, 0]);
        let mut l = solid(100, 100, [0, 0, 0]);
        // 在左上角放一小塊白
        for y in 0..10 {
            for x in 0..10 {
                l.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        // 往右下各挪 0.5（＝50 px）
        let out = blend(
            &g,
            &[],
            false,
            25, 100,
            &[Layer { img: &l, mask: &[], invert: false, xform: Xform { dx: 0.5, dy: 0.5, ..Default::default() } }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(5, 5).0, [0, 0, 0], "原本那塊白已經挪走了");
        assert_eq!(out.get_pixel(55, 55).0, [255, 255, 255], "挪到新位置");
    }

    /// 縮放以畫面中心為基準：縮一半之後，原本在邊角的東西會往中心收
    #[test]
    fn layer_scale_shrinks_towards_the_centre() {
        let g = solid(100, 100, [0, 0, 0]);
        let mut l = solid(100, 100, [0, 0, 0]);
        // 整個右半邊塗白
        for y in 0..100 {
            for x in 50..100 {
                l.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        let out = blend(
            &g,
            &[],
            false,
            25, 100,
            &[Layer {
                img: &l,
                mask: &[],
                invert: false,
                xform: Xform { scale: 0.5, ..Default::default() },
            }],
            BlendMode::Lighten,
            None,
        );
        // 以中心（x=50）縮一半：原本 x=50~100 的白色收成 x=50~75
        assert_eq!(out.get_pixel(60, 50).0, [255, 255, 255], "白色仍在，只是收窄了");
        assert_eq!(out.get_pixel(74, 50).0, [255, 255, 255], "右邊界收到約 x=75");
        assert_eq!(out.get_pixel(78, 50).0, [0, 0, 0], "超過收窄後的右邊界就沒東西了");
        assert_eq!(out.get_pixel(30, 50).0, [0, 0, 0], "左半邊本來就是黑的");
        // 縮小之後四周沒有來源，維持地景
        assert_eq!(out.get_pixel(95, 5).0, [0, 0, 0], "縮小後的外圍沒東西可疊");
    }

    /// 轉 90° 之後，原本在右半邊的東西會跑到下半邊
    #[test]
    fn layer_rotation_turns_the_content() {
        let g = solid(100, 100, [0, 0, 0]);
        let mut l = solid(100, 100, [0, 0, 0]);
        for y in 0..100 {
            for x in 55..100 {
                l.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        let out = blend(
            &g,
            &[],
            false,
            25, 100,
            &[Layer {
                img: &l,
                mask: &[],
                invert: false,
                xform: Xform { rot: 90.0, ..Default::default() },
            }],
            BlendMode::Lighten,
            None,
        );
        assert_eq!(out.get_pixel(50, 85).0, [255, 255, 255], "右半邊轉到了下半邊");
        assert_eq!(out.get_pixel(85, 50).0, [0, 0, 0], "原本的右半邊已經空了");
    }

    /// 倍率不能是 0（會被拿來當除數），角度也要收在合法範圍內
    #[test]
    fn xform_clamps_degenerate_values() {
        let c = Xform { scale: 0.0, rot: 999.0, dx: 50.0, dy: -50.0 }.clamped();
        assert!(c.scale >= XFORM_MIN_SCALE, "倍率不可為 0：{}", c.scale);
        assert!((-180.0..=180.0).contains(&c.rot));
        assert!((-2.0..=2.0).contains(&c.dx) && (-2.0..=2.0).contains(&c.dy));
        assert!(Xform::default().is_identity());
        assert!(!Xform { scale: 1.5, ..Default::default() }.is_identity());
    }

    /// 同一個場景（只有煙火不同）要對得回來；完全不同的場景要回 None
    #[test]
    fn alignment_finds_the_shift_and_gives_up_on_a_different_scene() {
        // 造一張有「地景」的圖：下半部一排亮點。**間距要不規則**——等距的
        // 圖案挪一整個週期看起來一模一樣，那是真的對不出唯一解（測資的問題，
        // 不是演算法的），所以這裡讓每一顆的位置與高低都不一樣
        let dots: [(u32, u32); 12] = [
            (6, 108),
            (19, 114),
            (27, 104),
            (48, 112),
            (55, 118),
            (77, 106),
            (91, 115),
            (104, 109),
            (126, 119),
            (141, 105),
            (158, 113),
            (180, 110),
        ];
        let mut ground = solid(200, 150, [4, 4, 8]);
        for (x, y) in dots {
            for dy in 0..6 {
                for dx in 0..6 {
                    ground.put_pixel(x + dx, y + dy, image::Rgb([200, 180, 90]));
                }
            }
        }
        // 同一個場景整個挪 7,4 px，另外加一團「煙火」（只有這張有）
        let (sx, sy) = (7u32, 4u32);
        let mut shifted = solid(200, 150, [4, 4, 8]);
        for (x, y) in dots {
            for dy in 0..6 {
                for dx in 0..6 {
                    shifted.put_pixel(x + dx + sx, y + dy + sy, image::Rgb([200, 180, 90]));
                }
            }
        }
        for y in 20..60 {
            for x in 40..90 {
                shifted.put_pixel(x, y, image::Rgb([255, 240, 120]));
            }
        }
        let d = align_offset(&ground, &shifted).expect("同一個場景要對得上");
        let (px, py) = (d.dx * 200.0, d.dy * 150.0);
        // 對齊要把它挪回來，所以位移是負的
        assert!(
            (px + sx as f32).abs() <= 2.0 && (py + sy as f32).abs() <= 2.0,
            "應該約 ({}, {})，實得 ({px:.1}, {py:.1})",
            -(sx as f32),
            -(sy as f32)
        );

        // 完全不同的場景：怎麼挪都對不上，要老實回 None
        let mut other = solid(200, 150, [4, 4, 8]);
        for y in 0..150 {
            for x in 0..200 {
                if (x / 7 + y / 7) % 2 == 0 {
                    other.put_pixel(x, y, image::Rgb([230, 230, 230]));
                }
            }
        }
        assert!(align_offset(&ground, &other).is_none(), "不同場景不該硬對");
    }

    #[test]
    fn no_layers_is_a_copy_of_the_ground() {
        let g = solid(5, 5, [12, 34, 56]);
        let out = blend(&g, &[], false, 25, 100, &[], BlendMode::Screen, None);
        assert_eq!(out, g);
    }
}
