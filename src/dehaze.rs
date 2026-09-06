//! 煙霧移除：估出煙霧的散射光層後直接扣掉，保留煙火細節。
//!
//! 針對夜間煙火照片：火藥煙被煙火照亮後形成大面積的低頻輝光，
//! 蓋掉夜空也讓煙火線條發灰。
//!
//! 這裡不用經典的暗通道先驗去霧（I = J·t + A·(1−t)）——它假設霧是明亮的
//! 日景背景，而夜間煙霧的絕對亮度遠低於大氣光，暗通道估出的透射率幾乎恆為 1，
//! 等於什麼都沒去掉。夜空是黑的，煙霧是「疊加」在上面的散射光，
//! 所以改用加性模型 I = J + S，估出 S 再相減。
//!
//! 估 S 的關鍵在於煙霧是連續的「面」、煙火軌跡是離散的「細線」：
//! 最小值池化與形態學開運算會吃掉比結構元素細的亮物件，
//! 留下的就是煙霧層，煙火的線條因此不在 S 裡、扣完仍完整保留。
//!
//! 不過開運算取的是煙霧的「下包絡」，煙霧自己的紋理起伏會被削平而留在畫面上。
//! 所以要剝好幾輪：扣掉一層後把剩下的再估一次，加起來才是完整的 S。

use image::{Rgb, RgbImage};

/// 估計煙霧層時的工作解析度（長邊）。煙霧是低頻訊號，縮圖估計不影響品質。
///
/// 但也不能縮太狠：下採樣取的是區塊最小值，區塊越大、取到的最小值就離
/// 這一帶真正的煙霧濃度越遠（有紋理的煙、雜訊都會把最小值往下拉），
/// 煙霧層被低估，扣完就留下一層薄霧。四千萬畫素的照片縮到 640 時
/// 一個區塊有 13×13 像素，煙根本扣不乾淨；1600 讓區塊回到 5 像素左右。
///
/// 預覽與存檔的差別不在這個值：縮圖無論縮到 1600 還是 2560 都是 1:1 取樣，
/// 差的是「有沒有經過原尺寸的最小值池化」（見 [`pool_gain`]）。
/// 預覽要對得上成品，靠的是把強度折算回去，不是把預覽也放大到這個尺寸。
const WORK_LONG_EDGE: u32 = 2560;

/// 「速度優先」時改用的工作解析度（見 [`SmokeParams::fast`]）。
///
/// 估煙霧層是整段去煙裡最花時間的一塊（約八成），而它的成本是隨工作解析度
/// 的**平方**走的：砍半就少四分之三的工作量。
///
/// 訂在 1280 的理由是下採樣的區塊大小：4K 影片在這個值下一格是 3×3 像素、
/// 1080p 是 1.5×1.5——都還比四千萬畫素的照片在預設值下的 5×5 更細
/// （見 [`WORK_LONG_EDGE`] 的說明，那個尺度早就驗過夠用）。
/// 差別在於煙霧層的細節少一階，煙火線條邊緣附近的殘留會略有不同
const FAST_WORK_EDGE: u32 = 1280;

/// 煙霧層要剝幾輪（見 [`estimate_smoke`] 的第 4 步）。
/// 每多剝一輪，就多撈回一階被開運算削平的煙霧起伏；三輪之後殘留已在雜訊水準，
/// 再加只是多花時間。
const PEEL: usize = 3;

/// 線條佔掉的面積換算成「這一帶有多像煙火自己」的倍率與上限
/// （見 [`streak_gate`]）。只有整叢煙火的正中心會頂到上限
const STREAK_GAIN: f32 = 2.0;
const STREAK_MAX: f32 = 0.5;

/// 相當於 sRGB 18 的線性亮度：比這還暗就是乾淨的夜空，
/// 上面的亮暗差只是雜訊，不能拿來當判據的分母
const NIGHT_FLOOR: f32 = 0.004;

/// 相減係數 k 的上限（強度 100 時的倍率）。
/// 最小值池化取的是區塊下界，估出的煙霧層比實際低一截，放大到這個倍率
/// 才能在強度拉滿時把煙霧完全扣乾淨。[`auto_params`] 反推強度時也要用它
const DEHAZE_GAIN: f32 = 1.6;

/// 扣掉多少比例的亮度（k·ys/yi）就改走「等比例壓暗」保住色相：
/// 低於前一個數完全走逐通道相減、高於後一個數完全保色相，中間平滑過渡。
///
/// 相減的殘差是 I 與 k·S 兩個相近數字的差，煙霧層的色度誤差在裡頭被放大成
/// 1/(1−扣掉比例) 倍：扣掉一半時誤差翻倍還算得準，扣到剩五分之一就整整放大五倍，
/// 金黃的煙火線條與橘色的煙一起翻成橄欖綠。煙霧層又是逐通道估的
/// （各通道的最小值可能來自不同像素），色度本來就是三者裡最不可靠的一項。
/// 換路徑不影響扣掉多少光——兩條路的亮度同樣掉 k·ys，只差殘留的顏色
const FADE_KEEP: (f32, f32) = (0.35, 0.8);

/// 「亮到沒有細節」的起點（sRGB 0~255）：最亮的通道到這裡就開始不再壓暗，
/// 頂到 255 則完全維持原樣（但只給不是煙霧層本身的像素，見 [`CLIP_KEEP_ABOVE`]）。
/// 感光元件截斷後的亮度是假的，
/// 拿它去扣煙霧只會把煙火最亮的芯扣成灰色
const CLIP_KEEP: f32 = 246.0;

/// [`CLIP_KEEP`] 只在「這個像素比底下的煙霧層亮這麼多倍」（yi/ys）時才生效。
///
/// 被紅色煙火照亮的濃煙，紅色通道會頂到 255、綠藍卻只有一百上下——
/// 單看最亮的通道它就是「過曝」，整片煙於是原封不動留在畫面上，邊緣還是
/// 硬切的一塊（實照的煙火中央就是這樣，煙火線條之間留著一塊塊橘紅的煙）。
/// 它與過曝的煙火芯真正的差別在「是不是煙霧層本身」：煙是平的一面，
/// 煙霧層估出來就等於它自己（實測 yi/ys ≈ 1.1）；線條與小的芯比結構元素細，
/// 不在煙霧層裡，yi/ys 至少兩三倍。
/// 大而平的亮芯同樣 yi/ys ≈ 1，但那由 [`CORE_KEEP_Y`]（夠白）與
/// [`CORE_KEEP_AREA`]（整帶夠亮）保住，本來就不靠這條
const CLIP_KEEP_ABOVE: (f32, f32) = (1.3, 2.0);

/// [`CLIP_KEEP`] 的另一條路：過曝的像素顏色**不夠濃**（最暗通道 ÷ 最亮通道）
/// 就照舊保護，不看 yi/ys。
///
/// 只靠 yi/ys 把關會傷到兩種東西（實照 DSC00370 回報過）：
/// * 密集的金柳——線條擠成一片，煙霧層估在它們底下不遠，yi/ys 只有 1.5 上下，
///   過曝的金黃線條於是被壓暗到三分之一，看起來整叢發灰。
/// * 噴泉周圍被照亮的那團白煙——它就是煙霧層本身（yi/ys ≈ 1），保護一撤，
///   只剩「整帶夠亮」的那塊平台原樣留著，平台邊緣外一步就扣到近乎全黑，
///   畫面上是一圈硬邊的斷階。
///
/// 該扣掉的紅煙（255, 117, 74）比值只有 0.29，遠在門檻之下；金黃的線條
/// 0.5～0.7、白煙 0.8 以上都在門檻之上。橘紅的線條過不了這關，但它們細，
/// 由上面的 yi/ys 那條保住
const CLIP_KEEP_NEUTRAL: (f32, f32) = (0.35, 0.6);

/// 煙火亮芯的保護門檻：夠亮**又**夠接近純白就完全不壓暗。
///
/// [`CLIP_KEEP`] 只看亮度，門檻又訂在 246——芯沒有頂到 255（實照上多半落在
/// 230~250）就保不住，壓完整團變成灰色。可是亮度單看也不夠：被煙火照亮的
/// 濃煙一樣可以很亮（實測估出來的煙霧層有 0.6% 的點超過 246）。
///
/// 兩者真正的差別在顏色——芯是三個通道一起頂上去的白，煙則帶著明顯的暖色。
/// 所以用「最暗的通道 ÷ 最亮的通道」當第二道判準：越接近 1 越白。
/// 白得夠純又夠亮才算芯，暖色的濃煙再亮也過不了這一關
const CORE_KEEP_Y: (f32, f32) = (215.0, 235.0);
const CORE_KEEP_NEUTRAL: (f32, f32) = (0.75, 0.92);

/// 亮芯的另一條路：**整整一帶**都亮到這個程度（鄰域平均亮度，sRGB 0~255）。
///
/// 上面那條逐點看顏色，暖色的芯（金柳、橘紅的牡丹）過不了「夠白」那一關，
/// 一樣被當成煙壓暗。可是芯裡並不是每個點都頂到 255——亮點與縫隙交錯，
/// 逐點判斷於是同一團忽保忽扣，扣完整團斑駁發灰，就是實照上看到的樣子。
///
/// 煙火最亮的那團與被照亮的濃煙，真正的差別在「亮的範圍有多連續」：
/// 芯是一整片連續的亮，實測一帶的平均亮度在 220 以上；
/// 濃煙再亮也只到 160 上下（兩張實照量到的最大值是 169）。
/// 判準取鄰域平均，本身是低頻的，同一團就會被一致對待，不會再有斑駁
const CORE_KEEP_AREA: (f32, f32) = (190.0, 220.0);

/// 量「一帶有多亮」的半徑（佔影像長邊）。要比芯裡的縫隙寬（不然量到的還是
/// 單點的忽亮忽暗），又要比整叢煙火窄（否則芯周圍的煙也一起被算亮）。
/// 0.7% 在 8000px 的照片上約 58px
const CORE_AREA_RADIUS: f32 = 0.007;

/// 「整帶夠亮」的平台往外暈開多寬（佔影像長邊）：平台邊緣之外，保護權重
/// 平滑地降到 0，約在這個寬度的一倍半處收尾（暈法見 [`dehaze_to_linear`] 裡的說明）。
///
/// 少了這一圈，平台裡原樣留著、一步之外就扣到近乎全黑——噴泉周圍被照亮的
/// 濃煙是平的一面，煙霧層估出來就等於它自己——畫面上是一圈硬邊的斷階
/// （實照 DSC00370 回報過）。噴泉的光暈本來就是一路淡出去的，這一圈讓它
/// 照著淡出去，而不是被切一刀。1% 在 9984px 的照片上約 100px，斜坡 200px。
/// 只有「整帶夠亮」的平台會暈開；逐點判定的白芯與過曝像素不會，
/// 煙火線條之間的煙才不會跟著被保護
const CORE_SKIRT: f32 = 0.015;

/// 暈開的那一圈只保護**本身仍然亮**的地方（一帶的平均亮度，sRGB 0~255）：
/// 低於前一個數完全不保、高於後一個數完全照距離暈開的權重保。
///
/// 距離是一回事，亮不亮是另一回事：噴泉的光暈從平台邊緣一路淡出去，一帶的亮度
/// 是連續的，該跟著保；而噴泉火花團旁邊那片只是被照亮的暗煙，亮度掉了一大截，
/// 就算離平台很近也還是煙——只照距離暈開會在火花團周圍留下一圈褐色的煙
/// （實照 L1003436 回報過）。上限接在平台門檻（[`CORE_KEEP_AREA`]）的下緣，
/// 平台邊緣上兩邊才接得起來
const CORE_SKIRT_GLOW: (f32, f32) = (110.0, 190.0);

/// 「軌跡比周圍高出多少」要看多大一圈（佔影像長邊）。
/// 必須比軌跡本身粗（不然軌跡自己會被算進周圍的平均，高出來的量就沒了），
/// 又要比煙的起伏細（否則量到的是煙的紋理，等於把煙也一起救回來）。
/// 0.25% 在 6000px 的照片上約 15px，煙火軌跡大約 5~15px
const RESTORE_RADIUS: f32 = 0.0025;

/// 量「高出周圍多少」之前先抹平的尺度（佔影像長邊）：比這還細的起伏是感光
/// 雜訊，不是軌跡。0.05% 在 9528px 的照片上約 4px；1600 的預覽縮圖上不到一個
/// 像素，就不抹——縮圖的雜訊在縮的時候已經平均掉了。
///
/// 少了這一步，原尺寸的照片會整片煙都被撈回來：每個像素都帶著雜訊，
/// 「周圍最低點」量到的是雜訊的谷底，而不是煙的水準，每個像素都因此
/// 「高出周圍」一截，過了下面的相對門檻，整片煙就以斑駁的顆粒留下來
/// （實照六千萬畫素、強度 100：一團煙留 44%，抹掉雜訊後只留 7%）。
/// 預覽縮圖沒有這個問題——縮的時候雜訊早被平均掉了——所以預覽看起來
/// 乾淨、存出來的成品卻是髒的。抹平的尺度照影像長邊算，成品才跟預覽一致。
/// 軌跡本身比這粗得多（[`RESTORE_RADIUS`] 的說明），抹過仍撈得回來
const RESTORE_GRAIN: f32 = 0.0005;

/// 補回軌跡的作用門檻：這一點被扣掉了幾成（`k·ys/yi`）。
/// 只在真的扣掉很多的地方補——乾淨的夜空本來就沒被扣掉什麼，
/// 在那裡補等於把感光雜訊當成軌跡撈回來，畫面會浮出一層斑點
const RESTORE_ON: (f32, f32) = (0.25, 0.60);

/// 「高出周圍多少才算軌跡」的相對對比門檻（高出量 ÷ 周圍亮度）。
/// 煙自己的紋理起伏相對於它的亮度很小，軌跡則是又細又亮的一條。
/// 訂太低濃煙區會浮出一層斑駁的煙紋，訂太高則細的軌跡撈不回來
const RESTORE_REL: (f32, f32) = (0.10, 0.30);

/// 天空判定的亮度門檻（線性光）：`sky_range` 0 時就是這個值，
/// 往上則加 `srgb_to_linear(range/100)`——滑桿因此直接對應 0~255 的亮度，
/// 拉滿等於不管亮度（見 [`sky_mask`]）。
/// [`auto_params`] 要由煙霧的亮度反推範圍，兩邊必須是同一條公式
const SKY_Y_BASE: f32 = 0.004;

/// 線條要多密才算「煙火自己」而不是煙霧（相對於 [`STREAK_MAX`]）：
/// 低於前一個數完全算煙霧、高於後一個數完全是煙火，中間平滑過渡。
/// 判據本身是「一帶」的統計量，密集的煙火簇連同它自己的光暈都在上界那頭，
/// 飄開的煙與只有零星線條掃過的地方則落在下界，清雲碰得到
const SKY_BURST: (f32, f32) = (0.15, 0.60);

/// 算紋理對比時分母補的那一截亮度（線性光，約 sRGB 40）。
/// 純粹的相對對比在暗處會失控——夜空只有雜訊，除下去照樣是個大數字，
/// 整片乾淨的夜空反而被判成有紋理。補上這一截，暗處就回到看絕對落差，
/// 亮處才真的看對比
const SKY_CONTRAST_FLOOR: f32 = 0.02;

/// 紋理門檻（相對對比，見 [`sky_mask`]）：`sky_range` 0 時的下限，與可加上去的幅度。
/// 一帶的統計量是平均，所以門檻比逐點的低。
/// 煙霧面的對比大約 0.03、煙火線條 0.5 上下、城市燈火還要更高
const SKY_AREA_CONTRAST: (f32, f32) = (0.05, 0.20);
const SKY_PT_CONTRAST: (f32, f32) = (0.10, 0.40);

/// [`sky_region`] 判「這一欄的天空最低到哪」的門檻。
/// 傳播是取一路上最弱的一段，半透明的煙霧會讓值一路衰減，
/// 門檻訂太高天際線會被拉高到煙霧的上緣
const REGION_REACH_MIN: f32 = 0.35;

/// [`sky_region`] 往下傳播前，種子沿水平方向擴張多寬（佔畫面寬度）。
/// 比這窄的障礙（煙火簇、旗桿、燈柱）繞得過去，橫貫整個畫面的地景擋得住。
/// 太小則煙火會擋住它下方的天空，太大則地景上的縫隙會讓天空漏到水面
const REGION_GAP: f32 = 0.08;

/// [`sky_region`] 的「水平線」餘裕（佔畫面高度）：乾淨夜空最低出現的那一列
/// （見 [`SkyProbe::sea`]）再往下這麼多，就是天空最低能到的地方，比它更低的欄
/// 一律拉回來。
///
/// 岸邊地景沒有橫貫整張時（中間是海口、河口），天空會從缺口流進水面——
/// 水面同樣平坦，種子擋不住；夜裡被燈光照到的樹林紋理又淡，也擋不住——
/// 再沿著水面鋪開，整片水面都被當成天空扣煙，倒影上留下一塊塊暗斑
/// （實照 B0045103、A1202723 回報過）。水面是亮的倒影，不會被判成乾淨夜空，
/// 所以「乾淨夜空最低到哪」就是水平線。3% 留給遠岸那條線與水平線附近的煙
const REGION_SEA_MARGIN: f32 = 0.03;

/// 水平線（見 [`SkyProbe::sea`]）取各欄「乾淨夜空最低那一列」的哪個分位數。
/// 取偏低的那一頭：煙火簇擋住的欄位乾淨夜空只到簇的上緣，不能讓它們把線抬高；
/// 又不取最低的單一欄，免得一條縫裡漏進來的暗水面把線拉到畫面底
const SEA_LEVEL_Q: f32 = 0.9;

/// 夜空底色取在乾淨夜空裡煙霧層的哪個分位數（見 [`floor_from`]）。
///
/// 取偏暗的那一頭：「乾淨夜空」的判定只看暗不暗、平不平，整片天空罩著一層
/// 薄薄的霧時（實照 DSC00370，褐色的薄霧），薄霧一樣夠暗夠平、會被收進來，
/// 取中位數就把薄霧當成了夜空底色，煙變成沒有東西可扣（自動判強度掉到下限）。
/// 取最暗的一頭，底色就是夜空最乾淨的那一角，薄霧仍是高出底色的煙。
/// 煙霧層本身是平滑的（開運算加導引濾波），低分位數不會落到雜訊谷底
const SKY_FLOOR_Q: f32 = 0.15;

/// 乾淨夜空至少要佔畫面這個比例才拿它定底色；太少就當夜空是黑的（底色 0）
const SKY_FLOOR_MIN: f32 = 0.02;

/// 底色亮度的上限（線性光，約 sRGB 60）。比這還亮的「乾淨夜空」不是夜空——
/// 天還沒黑的照片沒有夜色可留，照舊當黑扣
const SKY_FLOOR_MAX: f32 = 0.045;

/// [`sky_region`] 判紋理時用的範圍值（相當於 `sky_range` 的 0~1）。
/// 取 [`sky_mask`] 預設範圍的那一組門檻——已經調到能把岸邊與水面擋在外面。
/// 這裡刻意不跟著使用者的「範圍」滑桿走：那條滑桿管的是清雲要壓多亮的雲，
/// 而「哪裡是地景」不該隨它變動
const REGION_RANGE: f32 = 0.4;

/// 由亮度反推 `sky_range`：[`sky_mask`] 那條公式的反函數
fn sky_range_for(y: f32) -> i32 {
    (linear_to_srgb((y - SKY_Y_BASE).max(0.0)) * 100.0).round() as i32
}

/// 只在畫面某個區塊去煙時的作用範圍。
/// 用相對座標（0~1）而非像素，預覽縮圖與原尺寸才會框到同一塊。
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Region {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Region {
    /// 夾回 0~1 並確保 x0 < x1、y0 < y1（框選可能從右下往左上拉）
    fn normalized(self) -> Self {
        let (x0, x1) = (self.x0.min(self.x1), self.x0.max(self.x1));
        let (y0, y1) = (self.y0.min(self.y1), self.y0.max(self.y1));
        Self {
            x0: x0.clamp(0.0, 1.0),
            y0: y0.clamp(0.0, 1.0),
            x1: x1.clamp(0.0, 1.0),
            y1: y1.clamp(0.0, 1.0),
        }
    }

    /// 框太細會退化成一條線，視為沒有框選
    fn is_usable(&self) -> bool {
        self.x1 - self.x0 > 0.01 && self.y1 - self.y0 > 0.01
    }
}

/// 線性漸層：拖曳的起點是全效果，沿著拖曳方向漸弱，到終點歸零；
/// 與拖曳方向垂直的兩側無限延伸（比照 Lightroom 的線性漸層）
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Linear {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Linear {
    /// 兩端太近就沒有方向可言，視為沒畫
    fn is_usable(&self) -> bool {
        let (dx, dy) = (self.x1 - self.x0, self.y1 - self.y0);
        dx * dx + dy * dy > 1e-4
    }
}

/// 放射性漸層：橢圓內全效果，往外在羽化帶裡漸弱到 0。
/// `invert` 打開就反過來——橢圓外才去煙、裡面保持原樣
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Radial {
    pub cx: f32,
    pub cy: f32,
    pub rx: f32,
    pub ry: f32,
    pub invert: bool,
}

impl Radial {
    fn is_usable(&self) -> bool {
        self.rx > 0.005 && self.ry > 0.005
    }
}

/// 筆刷塗出來的一筆：沿著 `pts` 連成的折線刷出固定半徑的軟邊筆跡。
///
/// 一筆之內自己交疊不會變濃（同一筆繞回來塗，那一塊還是同一個濃度），
/// **筆與筆之間才相加**——想加濃就再刷一遍，這是筆刷該有的手感
/// （見 [`ShapeMask`]）
#[derive(Clone, PartialEq, Debug)]
pub struct Brush {
    /// 筆跡經過的點（相對座標 0~1）
    pub pts: Vec<[f32; 2]>,
    /// 筆刷半徑，以影像長邊的比例表示（縮圖與原尺寸才刷得一樣粗）
    pub radius: f32,
}

impl Brush {
    fn is_usable(&self) -> bool {
        !self.pts.is_empty() && self.radius > 0.002
    }
}

/// 遮色片的一個形狀。同一張遮色片可以疊好幾個，怎麼合起來見 [`ShapeMask`]
#[derive(Clone, PartialEq, Debug)]
pub enum Shape {
    /// 矩形框：框內去煙、框外原樣
    Rect(Region),
    Linear(Linear),
    Radial(Radial),
    /// 只有 GUI 畫得出來，smoke_cli 用不到
    #[allow(dead_code)]
    Brush(Brush),
    /// 自動選取的物件：點一下照片，程式往外長出整團連在一起的東西
    /// （一朵煙火、一團煙），存成一張小張的權重圖（見 [`ObjectMask`]）
    #[allow(dead_code)]
    Object(Object),
}

/// 自動選取物件的結果：使用者框一塊，程式把框裡那個東西的輪廓找出來。
///
/// 前面幾種形狀都是幾何的（框、線、橢圓、筆跡），用幾個數字就描述得完；
/// 「物件」的邊界是照片內容決定的，只能存成點陣。存的是**框住那一塊的小圖**
/// 而不是整張照片：框選通常只圈畫面的一小塊，整張存等於把格子鋪在沒選到的
/// 地方，同樣的格數只鋪在框裡，邊界就細得多。座標一律用相對值，
/// 預覽縮圖與原尺寸才共用得了同一份——兩邊選到的必須是同一塊
#[derive(Clone, Debug)]
pub struct Object {
    /// 分割出來的原始權重圖，`w`×`h`，0~255，**還沒套羽化與邊緣**。
    ///
    /// 那兩條滑桿是選完之後會來回調的，每調一格就重跑一次分割不只慢，
    /// 形狀還會跟著跳；留著這份，調滑桿就只是從它重算 [`Object::mask`]
    raw: std::sync::Arc<Vec<u8>>,
    /// 套完羽化與邊緣、實際拿去算遮色片的權重圖（255＝完全在物件內）
    pub mask: std::sync::Arc<Vec<u8>>,
    pub w: usize,
    pub h: usize,
    /// 這張小圖鋪在整張照片的哪一塊（相對座標 0~1）。
    /// 比使用者拉的那個框大一圈：多出來的那一圈是演算法用來認背景的樣本，
    /// 羽化往外暈開時也得有地方可以暈
    pub area: Region,
    /// 羽化 0~100：邊界往外暈開多寬
    pub feather: i32,
    /// 邊緣 −100~100：邊界整圈往內收（負）或往外擴（正）
    pub edge: i32,
}

impl PartialEq for Object {
    /// 同一份（Arc 指向同一塊）就相等，不必逐位元組比。
    /// 這個比較每幀都要做好幾次（判斷要不要重算預覽），
    /// 逐位元組比一張十幾萬點的圖是白花力氣。
    /// 羽化與邊緣不必比：改了那兩個就會重算出另一份 mask
    fn eq(&self, other: &Self) -> bool {
        self.w == other.w
            && self.h == other.h
            && self.area == other.area
            && std::sync::Arc::ptr_eq(&self.mask, &other.mask)
    }
}

impl Object {
    /// 選到的面積佔這張小圖的比例；小到看不出來的就不算數
    fn coverage(&self) -> f32 {
        if self.mask.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.mask.iter().map(|&v| v as u64).sum();
        sum as f32 / (self.mask.len() as f32 * 255.0)
    }

    fn is_usable(&self) -> bool {
        self.w >= 4 && self.h >= 4 && self.coverage() > 0.0015
    }

    /// 取整張照片的相對座標 (0~1) 上的權重；框住的那一塊之外一律 0。
    ///
    /// 用雙線性而不是取最近的那一格：小圖套回原尺寸動輒放大十幾倍，
    /// 取最近的會把邊界切成一格一格的階梯
    fn at(&self, u: f32, v: f32) -> f32 {
        let (aw, ah) = (self.area.x1 - self.area.x0, self.area.y1 - self.area.y0);
        if self.w == 0 || self.h == 0 || aw <= 0.0 || ah <= 0.0 {
            return 0.0;
        }
        let fx = (u - self.area.x0) / aw * self.w as f32 - 0.5;
        let fy = (v - self.area.y0) / ah * self.h as f32 - 0.5;
        // 連最近的那一格都構不著＝在這張小圖之外
        if fx < -1.0 || fy < -1.0 || fx > self.w as f32 || fy > self.h as f32 {
            return 0.0;
        }
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let cx = |x: f32| x.clamp(0.0, (self.w - 1) as f32) as usize;
        let cy = |y: f32| y.clamp(0.0, (self.h - 1) as f32) as usize;
        let (x0i, x1i) = (cx(x0), cx(x0 + 1.0));
        let (y0i, y1i) = (cy(y0), cy(y0 + 1.0));
        let g = |xi: usize, yi: usize| self.mask[yi * self.w + xi] as f32;
        let top = g(x0i, y0i) + (g(x1i, y0i) - g(x0i, y0i)) * tx;
        let bot = g(x0i, y1i) + (g(x1i, y1i) - g(x0i, y1i)) * tx;
        (top + (bot - top) * ty) / 255.0
    }

    /// 換一組羽化／邊緣重算權重圖；分割本身不重跑（見 [`Object::raw`]）
    #[allow(dead_code)]
    pub fn refined(&self, feather: i32, edge: i32) -> Object {
        let (feather, edge) = (feather.clamp(0, 100), edge.clamp(-100, 100));
        if feather == self.feather && edge == self.edge {
            return self.clone();
        }
        Object {
            mask: std::sync::Arc::new(refine_object(&self.raw, self.w, self.h, feather, edge)),
            raw: self.raw.clone(),
            w: self.w,
            h: self.h,
            area: self.area,
            feather,
            edge,
        }
    }
}

impl Shape {
    /// 正規化並丟掉退化的形狀（拖一下就放開、細成一條線的框等）
    pub fn cleaned(&self) -> Option<Shape> {
        match self {
            Shape::Rect(r) => {
                let r = r.normalized();
                r.is_usable().then_some(Shape::Rect(r))
            }
            Shape::Linear(l) => l.is_usable().then_some(Shape::Linear(*l)),
            Shape::Radial(r) => r.is_usable().then_some(Shape::Radial(*r)),
            Shape::Brush(b) => b.is_usable().then(|| Shape::Brush(b.clone())),
            Shape::Object(o) => o.is_usable().then(|| Shape::Object(o.clone())),
        }
    }
}

/// 把分割出來的原始權重圖套上羽化與邊緣，變成實際要用的那一張
fn refine_object(raw: &[u8], w: usize, h: usize, feather: i32, edge: i32) -> Vec<u8> {
    let long = w.max(h) as f32;
    let mut p = Plane::new(w, h);
    for (d, &v) in p.d.iter_mut().zip(raw) {
        *d = v as f32 / 255.0;
    }
    // 邊緣：整圈往內收或往外擴。先把邊界糊成一條有寬度的帶子，再把
    // 「算是裡面」的那條等高線挪到帶子的另一側——等同於形態學的腐蝕與膨脹，
    // 但過渡仍然是連續的，不會把柔邊切成鋸齒
    if edge != 0 {
        let r =
            ((edge.unsigned_abs() as f32 / 100.0 * OBJECT_EDGE_R * long).round() as usize).max(1);
        p = box_mean(&p, r);
        let t = 0.5 - edge as f32 / 100.0 * OBJECT_EDGE_SHIFT;
        for v in p.d.iter_mut() {
            *v = smoothstep(t - OBJECT_EDGE_BAND, t + OBJECT_EDGE_BAND, *v);
        }
    }
    // 羽化：邊界往外暈開。兩趟方框平均近似高斯，
    // 一趟的直線斜坡會在兩端留下看得出來的折線
    let rf = (feather as f32 / 100.0 * OBJECT_FEATHER_R * long).round() as usize;
    if rf >= 1 {
        p = box_mean(&p, rf);
        p = box_mean(&p, (rf + 1) / 2);
    }
    p.d.iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// 在顏色格子空間裡抹一次（三個色軸各做一次 1-2-1）。
/// 格子開得細、樣本卻不多，不抹的話差一格的顏色就互相不認得
fn blur_bins(hist: &mut [f32]) {
    let mut tmp = vec![0f32; hist.len()];
    // 索引是 (r×B + g)×B + b，所以三個軸的間距分別是 1、B、B²
    for step in [1usize, OBJECT_BINS, OBJECT_BINS * OBJECT_BINS] {
        tmp.copy_from_slice(hist);
        for (i, v) in hist.iter_mut().enumerate() {
            let k = (i / step) % OBJECT_BINS;
            let lo = if k > 0 { tmp[i - step] } else { tmp[i] };
            let hi = if k + 1 < OBJECT_BINS { tmp[i + step] } else { tmp[i] };
            *v = (lo + 2.0 * tmp[i] + hi) * 0.25;
        }
    }
}

/// 自動選取物件：**框住**要選的東西，程式把框裡那一個從背景裡分出來
/// （比照 Lightroom 的「物件」選取）。
///
/// 框只是告訴程式「東西在這裡面」，真正的邊界由照片內容決定：把框外那一圈
/// 當成**確定是背景**的樣本，與框裡的顏色分佈互相比對，反覆修正每一格
/// 「比較像哪一邊」，最後再用照片本身的明暗把邊界吸附到真正的輪廓上。
///
/// 這比從一點往外洪泛穩得多：洪泛只看「有沒有比種子暗太多」，
/// 碰到背景與物件亮度接近的地方就整片洩出去；框選則永遠有背景樣本可比，
/// 而且使用者已經先講明了東西在哪一塊
///
/// * `img` 是預覽底圖（原圖等比縮小的那張就夠，邊界是柔的）
/// * `sel` 是拖出來的框（相對座標）
/// * `feather`、`edge` 見 [`Object`]
pub fn select_object(img: &RgbImage, sel: Region, feather: i32, edge: i32) -> Option<Object> {
    let sel = sel.normalized();
    if !sel.is_usable() {
        return None;
    }
    let (iw, ih) = (img.width() as usize, img.height() as usize);
    if iw < 8 || ih < 8 {
        return None;
    }
    // 框外再多留一圈：那一圈是「保證是背景」的樣本，羽化往外暈開也要有地方暈
    let (sw, sh) = (sel.x1 - sel.x0, sel.y1 - sel.y0);
    let pad = Region {
        x0: (sel.x0 - sw * OBJECT_PAD).max(0.0),
        y0: (sel.y0 - sh * OBJECT_PAD).max(0.0),
        x1: (sel.x1 + sw * OBJECT_PAD).min(1.0),
        y1: (sel.y1 + sh * OBJECT_PAD).min(1.0),
    };
    // 對齊到實際的像素邊界，小圖與照片才對得起來
    let x0 = ((pad.x0 * iw as f32).floor() as usize).min(iw - 1);
    let y0 = ((pad.y0 * ih as f32).floor() as usize).min(ih - 1);
    let x1 = ((pad.x1 * iw as f32).ceil() as usize).clamp(x0 + 1, iw);
    let y1 = ((pad.y1 * ih as f32).ceil() as usize).clamp(y0 + 1, ih);
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw < 8 || ch < 8 {
        return None;
    }
    let area = Region {
        x0: x0 as f32 / iw as f32,
        y0: y0 as f32 / ih as f32,
        x1: x1 as f32 / iw as f32,
        y1: y1 as f32 / ih as f32,
    };
    // 格子全鋪在框住的那一塊上：框得越小，邊界算得越細
    let scale = (OBJECT_WORK_EDGE as f32 / cw.max(ch) as f32).min(1.0);
    let w = ((cw as f32 * scale).round() as usize).max(8);
    let h = ((ch as f32 * scale).round() as usize).max(8);
    let n = w * h;

    // 降取樣：顏色取 sRGB 的區塊平均（顏色分佈在 sRGB 上分得比較開，
    // 線性光會把大半個夜景擠在最底下那幾格），亮度另存一份當導引圖
    let mut rgb = vec![[0f32; 3]; n];
    let mut lum = Plane::new(w, h);
    for yy in 0..h {
        let (by0, by1) = (yy * ch / h, ((yy + 1) * ch / h).max(yy * ch / h + 1).min(ch));
        for xx in 0..w {
            let (bx0, bx1) = (xx * cw / w, ((xx + 1) * cw / w).max(xx * cw / w + 1).min(cw));
            let mut s = [0f32; 3];
            let mut cnt = 0f32;
            for py in by0..by1 {
                for px in bx0..bx1 {
                    let p = img.get_pixel((x0 + px) as u32, (y0 + py) as u32).0;
                    for c in 0..3 {
                        s[c] += p[c] as f32;
                    }
                    cnt += 1.0;
                }
            }
            let cnt = cnt.max(1.0);
            let c = [s[0] / cnt, s[1] / cnt, s[2] / cnt];
            rgb[yy * w + xx] = c;
            lum.d[yy * w + xx] = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0;
        }
    }

    // 每一格落在哪個顏色格子裡（16×16×16）
    let bins: Vec<usize> = rgb
        .iter()
        .map(|c| {
            let q = |v: f32| ((v * OBJECT_BINS as f32 / 256.0) as usize).min(OBJECT_BINS - 1);
            (q(c[0]) * OBJECT_BINS + q(c[1])) * OBJECT_BINS + q(c[2])
        })
        .collect();
    // 哪些格子在使用者拉的框裡（框外一律是背景，不會被選進去）
    let inside: Vec<bool> = (0..n)
        .map(|i| {
            let (xx, yy) = (i % w, i / w);
            let u = area.x0 + (xx as f32 + 0.5) / w as f32 * (area.x1 - area.x0);
            let v = area.y0 + (yy as f32 + 0.5) / h as f32 * (area.y1 - area.y0);
            u >= sel.x0 && u <= sel.x1 && v >= sel.y0 && v <= sel.y1
        })
        .collect();
    let inside_n = inside.iter().filter(|&&b| b).count();
    if inside_n < 16 {
        return None;
    }

    // 反覆修正：先假設框裡全是物件，算出兩邊的顏色分佈，再回頭問每一格
    // 「你比較像哪一邊」，然後拿新的答案重算分佈。幾輪之後就收斂了
    const NBINS: usize = OBJECT_BINS * OBJECT_BINS * OBJECT_BINS;
    let mut fg = inside.clone();
    let mut prob = Plane::new(w, h);
    let rs = ((OBJECT_SMOOTH * w.max(h) as f32).round() as usize).max(1);
    for _ in 0..OBJECT_ROUNDS {
        let (mut hf, mut hb) = (vec![0f32; NBINS], vec![0f32; NBINS]);
        for i in 0..n {
            // 框外一律算背景樣本，而且加重——它是唯一「確定」的那一邊
            if !inside[i] {
                hb[bins[i]] += OBJECT_RING_W;
            } else if fg[i] {
                hf[bins[i]] += 1.0;
            } else {
                hb[bins[i]] += 1.0;
            }
        }
        blur_bins(&mut hf);
        blur_bins(&mut hb);
        let (nf, nb) = (hf.iter().sum::<f32>(), hb.iter().sum::<f32>());
        if nf <= 0.0 || nb <= 0.0 {
            return None;
        }
        for i in 0..n {
            let b = bins[i];
            let pf = (hf[b] + OBJECT_PRIOR) / (nf + OBJECT_PRIOR * NBINS as f32);
            let pb = (hb[b] + OBJECT_PRIOR) / (nb + OBJECT_PRIOR * NBINS as f32);
            prob.d[i] = pf / (pf + pb);
        }
        // 空間上抹一次：單獨一格跟旁邊唱反調多半是雜訊
        prob = box_mean(&prob, rs);
        for i in 0..n {
            fg[i] = inside[i] && prob.d[i] > 0.5;
        }
    }

    // 兩段門檻：先要有「很確定是物件」的核，再沿著「還算像」的格子長出去。
    // 框裡若根本沒有跟背景不一樣的東西（整片乾淨的夜空），機率會全部停在
    // 0.5 附近、長不出核來——那時寧可說一聲選不到，也不要交出半片隨機的東西。
    // 順手把碎塊丟掉：太小的一團多半是雜訊，不是使用者想選的東西
    let speck = ((inside_n as f32 * OBJECT_SPECK).round() as usize).max(4);
    let mut keep = vec![false; n];
    let mut seen = vec![false; n];
    let mut comp: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if seen[start] || !fg[start] {
            continue;
        }
        comp.clear();
        stack.clear();
        stack.push(start);
        seen[start] = true;
        let mut core = false;
        while let Some(i) = stack.pop() {
            comp.push(i);
            core |= prob.d[i] >= OBJECT_SEED;
            let (x, y) = (i % w, i / w);
            let mut push = |j: usize, seen: &mut Vec<bool>, st: &mut Vec<usize>| {
                if !seen[j] && fg[j] {
                    seen[j] = true;
                    st.push(j);
                }
            };
            if x > 0 {
                push(i - 1, &mut seen, &mut stack);
            }
            if x + 1 < w {
                push(i + 1, &mut seen, &mut stack);
            }
            if y > 0 {
                push(i - w, &mut seen, &mut stack);
            }
            if y + 1 < h {
                push(i + w, &mut seen, &mut stack);
            }
        }
        if core && comp.len() >= speck {
            for &i in &comp {
                keep[i] = true;
            }
        }
    }
    if !keep.iter().any(|&b| b) {
        return None;
    }

    // 這裡先問一次「這真的是一個東西嗎」：把選到的地方平均起來看有多少把握。
    //
    // 光有核還不夠。乾淨的夜空其實不是死黑一片，遠處的燈光會讓它有一點很淡的
    // 漸層——兩邊的顏色分佈於是照樣分得出高低，長出一團跨過門檻卻毫無意義的
    // 東西（實測那種情況平均把握 0.57，真的框到煙火或橋塔則有 0.73 以上）。
    // 框到的是一片沒有東西的地方時，寧可讓 UI 說一聲重框。
    //
    // **一定要在下面補洞之前算**：補進來的是暗處，它們本來就不像前景，
    // 算進平均只會把把握度拉低——一棟暗面佔一半的大樓會因此被誤判成沒東西
    let (sum, cnt) = keep.iter().enumerate().fold((0f32, 0f32), |(s, c), (i, &k)| {
        if k {
            (s + prob.d[i], c + 1.0)
        } else {
            (s, c)
        }
    });
    if cnt <= 0.0 || sum / cnt < OBJECT_CONFIDENT {
        return None;
    }

    // 選到的那一團裡面常常缺一塊塊的暗處：大樓沒打燈的那一面、煙火線條之間的
    // 夜空。使用者要的是**這個東西整個**，不是它亮的那部分——所以先做一次
    // 閉運算（先膨脹再腐蝕，整體大小不變）把邊上的缺口與凹進去的地方接起來
    let rc = ((OBJECT_CLOSE * w.max(h) as f32).round() as usize).max(1);
    let mut m = Plane::new(w, h);
    for i in 0..n {
        m.d[i] = if keep[i] { 1.0 } else { 0.0 };
    }
    // 膨脹：視窗裡只要有一格選到就算（方框平均 > 0）
    let grown = box_mean(&m, rc);
    for i in 0..n {
        m.d[i] = if grown.d[i] > 1e-6 { 1.0 } else { 0.0 };
    }
    // 腐蝕：視窗裡全是選到的才留下，把剛才多長出來的那一圈收回去。
    // 框外一律不算——那一圈是背景樣本，膨脹不該把它吃進來
    let shrunk = box_mean(&m, rc);
    for i in 0..n {
        keep[i] = inside[i] && shrunk.d[i] > 1.0 - 1e-4;
    }

    // 再把包在裡面的空缺補滿，**不管多大**：從小圖的外框往內灌水，灌得到的
    // 才是真正的背景，灌不到的就是被這個東西整個圍住的洞——就算它全黑也是
    // 這個東西的一部分（一整面沒打燈的牆正是這樣）
    let mut seen = vec![false; n];
    stack.clear();
    for i in 0..n {
        let (x, y) = (i % w, i / w);
        if (x == 0 || y == 0 || x + 1 == w || y + 1 == h) && !keep[i] && !seen[i] {
            seen[i] = true;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize, seen: &mut Vec<bool>, st: &mut Vec<usize>| {
            if !seen[j] && !keep[j] {
                seen[j] = true;
                st.push(j);
            }
        };
        if x > 0 {
            push(i - 1, &mut seen, &mut stack);
        }
        if x + 1 < w {
            push(i + 1, &mut seen, &mut stack);
        }
        if y > 0 {
            push(i - w, &mut seen, &mut stack);
        }
        if y + 1 < h {
            push(i + w, &mut seen, &mut stack);
        }
    }
    for i in 0..n {
        keep[i] |= !seen[i];
    }

    // 用照片本身的明暗把邊界吸到真正的輪廓上：導引濾波讓權重在亮度相近的
    // 地方保持一致、在有邊的地方跟著跳，等於沿著物件的邊切下去。
    // 到這裡為止的邊界是「顏色像不像」決定的，只準到格子那一級
    let mut soft = Plane::new(w, h);
    for i in 0..n {
        soft.d[i] = if keep[i] { 1.0 } else { 0.0 };
    }
    let rg = ((OBJECT_GUIDE * w.max(h) as f32).round() as usize).max(2);
    let soft = Guide::new(&lum, rg).filter(&soft, OBJECT_GUIDE_EPS);

    let raw: Vec<u8> = (0..n)
        .map(|i| {
            // 框外一律 0：使用者已經講明東西在框裡，導引濾波溢出去的那一點
            // 不該把框外的東西也算進來
            let v = if inside[i] {
                soft.d[i].clamp(0.0, 1.0)
            } else {
                0.0
            };
            (v * 255.0).round() as u8
        })
        .collect();
    let raw = std::sync::Arc::new(raw);
    let (feather, edge) = (feather.clamp(0, 100), edge.clamp(-100, 100));
    let obj = Object {
        mask: std::sync::Arc::new(refine_object(&raw, w, h, feather, edge)),
        raw,
        w,
        h,
        area,
        feather,
        edge,
    };
    obj.is_usable().then_some(obj)
}

/// 一張遮色片最多疊幾個形狀。逐像素要走過每個形狀，
/// 疊太多不只算得慢，畫面上也看不出誰是誰了
pub const MAX_SHAPES: usize = 32;

/// 自動選取物件的工作解析度（長邊）。格子只鋪在框住的那一塊上，
/// 所以框得越小、邊界算得越細；算太細只是慢，雜訊還會讓分割碎掉
const OBJECT_WORK_EDGE: usize = 384;

/// 框外再多留幾成當背景樣本（佔框本身的邊長）
const OBJECT_PAD: f32 = 0.18;

/// 顏色分佈的格數（每個色軸）
const OBJECT_BINS: usize = 16;

/// 「這一格像前景還是背景」要反覆修正幾輪
const OBJECT_ROUNDS: usize = 4;

/// 框外那一圈當背景樣本時的權重。它是唯一「確定」的那一邊，算重一點
const OBJECT_RING_W: f32 = 2.0;

/// 顏色分佈的先驗（每格先墊這麼多），免得沒出現過的顏色機率直接變成 0。
/// 墊太多會蓋過樣本本身：四千多個格子每格墊 0.5 就是兩千多的假樣本，
/// 比框裡真正的格子數還多，兩邊只剩「誰的樣本少誰佔便宜」
const OBJECT_PRIOR: f32 = 0.02;

/// 機率圖的空間平滑半徑（佔工作解析度長邊）
const OBJECT_SMOOTH: f32 = 0.012;

/// 認定為「核」的機率門檻。往外長的門檻是 0.5——先要有核，才沿著還算像的
/// 格子長出去；框裡沒有跟背景不一樣的東西時就長不出核（見 [`select_object`]）
const OBJECT_SEED: f32 = 0.62;

/// 選到的地方平均要有這麼高的把握，才算真的框到一個東西
/// （見 [`select_object`] 最後那一關）
const OBJECT_CONFIDENT: f32 = 0.65;

/// 比框內面積這個比例還小的碎塊丟掉
const OBJECT_SPECK: f32 = 0.003;

/// 閉運算的半徑（佔工作解析度長邊）：邊上比這個窄的缺口會被接起來，
/// 接起來之後就成了被包在裡面的洞，跟著一起補滿（見 [`select_object`]）
const OBJECT_CLOSE: f32 = 0.02;

/// 邊界吸附用的導引濾波半徑（佔工作解析度長邊）
const OBJECT_GUIDE: f32 = 0.015;

/// 邊界吸附的容許差異：比這個小的亮度起伏當成同一塊東西
const OBJECT_GUIDE_EPS: f32 = 1e-3;

/// 邊緣拉到底時，邊界最多挪多寬（佔小圖長邊）
const OBJECT_EDGE_R: f32 = 0.02;

/// 邊緣拉到底時，「算是裡面」的那條等高線挪多少
const OBJECT_EDGE_SHIFT: f32 = 0.30;

/// 等高線挪完之後，它兩側的過渡寬度
const OBJECT_EDGE_BAND: f32 = 0.12;

/// 羽化拉到底時，邊界往外暈開多寬（佔小圖長邊）
const OBJECT_FEATHER_R: f32 = 0.05;

/// 「物件」選取的羽化預設值。小圖套回原尺寸是放大十幾倍，
/// 0 會讓邊界看得出格子的階梯，先給一點過渡
pub const OBJECT_FEATHER: i32 = 15;

/// 去煙參數
#[derive(Clone, PartialEq, Debug)]
pub struct SmokeParams {
    /// 去除強度 0~100：煙霧層要扣掉多少，100 時幾乎移除全部散射光
    pub strength: i32,
    /// 細節保留 0~100：數值越高，煙霧層越貼合原圖邊緣（煙火線條越不被削）
    pub detail: i32,
    /// 補回煙裡的軌跡：煙散掉之後，把被連同煙一起扣掉的煙火軌跡救回來。
    ///
    /// 煙是**平順**的一層，軌跡是高出來的那一點。濃煙很亮時估出來的煙霧層
    /// 會比軌跡本身還高，`i − k·s` 於是把兩者一起扣成 0——畫面上就是煙火被
    /// 咬掉一塊。軌跡的訊號其實還在檔案裡（把那塊拉高對比就看得到），
    /// 所以這裡不是無中生有，是把它從煙底下撈回來（見 [`RESTORE_RADIUS`]）
    pub restore_trails: bool,
    /// 只處理天空：作用範圍自動限在天際線以上（見 [`sky_region`]）。
    /// 煙只飄在天空，可是岸邊燈火與水面倒影又亮又連續，估起來也像一層煙，
    /// 扣下去整片會被壓暗。關掉則整張一視同仁——煙真的飄到地面、
    /// 或畫面裡根本沒有地景時才需要
    pub sky_only: bool,
    /// 遮色片：只在這些形狀蓋到的地方去煙；空的＝整張照片
    pub shapes: Vec<Shape>,
    /// 形狀邊緣的羽化寬度 0~100（相對於形狀本身的尺度）。
    /// 邊界硬切會在畫面上留下一條看得出來的接縫
    pub feather: i32,
    /// 筆刷濃度 0~100：筆刷**一筆**上多少（見 [`ShapeMask`]）。
    /// 只作用在筆刷上——框選、漸層與物件一律 100%；沒畫任何形狀時也不作用
    pub mask_density: i32,
    /// 保護色（sRGB）：與其中任一色相近的像素都不去煙，用來留住不想被扣掉的顏色。
    /// 固定長度是為了不必為了幾個色票配置一個 Vec；全為 None＝不做顏色保護
    pub protect: [Option<[u8; 3]>; MAX_PROTECT],
    /// 保護色的容許範圍 0~100：越大則越多相近的顏色一起被保護
    pub tolerance: i32,
    /// 雲朵清除 0~100：把天空裡沒有紋理的暗面（雲、殘餘輝光）壓回乾淨的夜色
    pub sky_clean: i32,
    /// 天空判定的亮度範圍 0~100：越大則越亮的雲也會被認定成天空
    pub sky_range: i32,
    /// 夜空顏色；None＝維持原本的色調
    pub sky_color: Option<[u8; 3]>,
    /// 夜空上色強度 0~100
    pub sky_tint: i32,
    /// 雲色（sRGB）：用吸管從照片上吸下來的雲朵顏色。
    /// 亮到不會被亮度／紋理判定成天空的雲，靠這個直接指名，一樣壓回夜色
    pub cloud: [Option<[u8; 3]>; MAX_CLOUD],
    /// 雲色的容許範圍 0~100：越大則越多相近的顏色一起被當成雲
    pub cloud_range: i32,
    /// 手上這張是某張原圖的**縮圖**時，填原圖的長邊；None＝這張就是原圖。
    ///
    /// 預覽走這條路。天空範圍的統計半徑是固定的像素尺度，縮圖上要照原圖
    /// 換算過去，預覽看到的範圍才與成品相同（見 [`region_radii`]）。
    /// 強度那邊也有一份同樣用意的折算（見 [`preview_strength`]）
    pub preview_of: Option<u32>,
    /// 速度優先：煙霧層改用比較小的工作解析度估（見 [`FAST_WORK_EDGE`]），
    /// 快一倍上下，代價是煙霧層的細節少一階。
    ///
    /// 影片模組專用的取捨——一支片子有上萬格，照片一張只要一秒，
    /// 沒有必要為了那一秒換掉品質。強度會照 [`pool_gain_at`] 折算，
    /// 所以同一個滑桿數字在兩種模式下仍代表同一種效果
    pub fast: bool,
}

impl SmokeParams {
    /// 手上這張影像該照哪個長邊換算尺度：縮圖照原圖、原圖照自己
    /// （見 [`SmokeParams::preview_of`]）
    fn source_long(&self, img_long: u32) -> f32 {
        self.preview_of.unwrap_or(img_long) as f32
    }

    /// 這組參數要用多大的工作解析度估煙霧層
    fn work_edge(&self) -> u32 {
        if self.fast {
            FAST_WORK_EDGE
        } else {
            WORK_LONG_EDGE
        }
    }
}

/// 最多可以指定幾個保護色
pub const MAX_PROTECT: usize = 6;

/// 最多可以吸幾個雲色
pub const MAX_CLOUD: usize = 6;

impl Default for SmokeParams {
    fn default() -> Self {
        Self {
            strength: 80,
            detail: 60,
            restore_trails: true,
            sky_only: true,
            shapes: Vec::new(),
            feather: 25,
            mask_density: 80,
            protect: [None; MAX_PROTECT],
            tolerance: 30,
            sky_clean: 0,
            sky_range: 40,
            sky_color: None,
            sky_tint: 60,
            cloud: [None; MAX_CLOUD],
            cloud_range: 35,
            preview_of: None,
            fast: false,
        }
    }
}

impl SmokeParams {
    pub fn clamped(&self) -> Self {
        let mut s = self.clone();
        s.strength = s.strength.clamp(0, 100);
        s.detail = s.detail.clamp(0, 100);
        s.feather = s.feather.clamp(0, 100);
        s.tolerance = s.tolerance.clamp(0, 100);
        s.sky_clean = s.sky_clean.clamp(0, 100);
        s.sky_range = s.sky_range.clamp(0, 100);
        s.sky_tint = s.sky_tint.clamp(0, 100);
        s.cloud_range = s.cloud_range.clamp(0, 100);
        s.shapes = s
            .shapes
            .iter()
            .filter_map(Shape::cleaned)
            .take(MAX_SHAPES)
            .collect();
        s
    }

    /// 有沒有畫過遮色片（全部退化的形狀不算）
    #[allow(dead_code)]
    pub fn has_shapes(&self) -> bool {
        self.shapes.iter().any(|s| s.cleaned().is_some())
    }

    /// 疊一個形狀上去；已經滿了就回傳 false
    #[allow(dead_code)]
    pub fn add_shape(&mut self, s: Shape) -> bool {
        if self.shapes.len() >= MAX_SHAPES {
            return false;
        }
        self.shapes.push(s);
        true
    }

    /// 還原最後畫上去的那個形狀
    #[allow(dead_code)]
    pub fn undo_shape(&mut self) {
        self.shapes.pop();
    }

    #[allow(dead_code)]
    pub fn clear_shapes(&mut self) {
        self.shapes.clear();
    }

    /// 最後畫上去的形狀（GUI 用來就地改剛畫好的那一個，例如放射漸層的反轉）
    #[allow(dead_code)]
    pub fn last_shape_mut(&mut self) -> Option<&mut Shape> {
        self.shapes.last_mut()
    }

    /// 完全沒有要動到影像（去煙、清雲、夜空上色都關著）
    pub fn is_neutral(&self) -> bool {
        self.strength <= 0 && !self.touches_sky()
    }

    /// 有沒有要動天空（清雲或上色）
    fn touches_sky(&self) -> bool {
        self.sky_clean > 0 || (self.sky_color.is_some() && self.sky_tint > 0)
    }

    // 以下幾個是給 GUI 用的，smoke_cli 只會用到 add_protect
    /// 目前設定的保護色（依加入順序）
    #[allow(dead_code)]
    pub fn protect_colors(&self) -> impl Iterator<Item = (usize, [u8; 3])> + '_ {
        self.protect
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
    }

    #[allow(dead_code)]
    pub fn has_protect(&self) -> bool {
        self.protect.iter().any(Option::is_some)
    }

    /// 加一個保護色；已經滿了或已存在同色則回傳 false
    pub fn add_protect(&mut self, c: [u8; 3]) -> bool {
        if self.protect.contains(&Some(c)) {
            return false;
        }
        match self.protect.iter_mut().find(|s| s.is_none()) {
            Some(slot) => {
                *slot = Some(c);
                true
            }
            None => false,
        }
    }

    #[allow(dead_code)]
    pub fn remove_protect(&mut self, i: usize) {
        if let Some(slot) = self.protect.get_mut(i) {
            *slot = None;
        }
    }

    #[allow(dead_code)]
    pub fn clear_protect(&mut self) {
        self.protect = [None; MAX_PROTECT];
    }

    /// 目前吸下來的雲色（依加入順序）
    #[allow(dead_code)]
    pub fn cloud_colors(&self) -> impl Iterator<Item = (usize, [u8; 3])> + '_ {
        self.cloud
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, c)))
    }

    #[allow(dead_code)]
    pub fn has_cloud(&self) -> bool {
        self.cloud.iter().any(Option::is_some)
    }

    /// 加一個雲色；已經滿了或已存在同色則回傳 false
    pub fn add_cloud(&mut self, c: [u8; 3]) -> bool {
        if self.cloud.contains(&Some(c)) {
            return false;
        }
        match self.cloud.iter_mut().find(|s| s.is_none()) {
            Some(slot) => {
                *slot = Some(c);
                true
            }
            None => false,
        }
    }

    #[allow(dead_code)]
    pub fn remove_cloud(&mut self, i: usize) {
        if let Some(slot) = self.cloud.get_mut(i) {
            *slot = None;
        }
    }

    #[allow(dead_code)]
    pub fn clear_cloud(&mut self) {
        self.cloud = [None; MAX_CLOUD];
    }
}

/// 單通道浮點影像平面
#[derive(Clone)]
struct Plane {
    w: usize,
    h: usize,
    d: Vec<f32>,
}

impl Plane {
    fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            d: vec![0.0; w * h],
        }
    }

    /// 以像素中心為基準的雙線性取樣（座標可落在格子之間）
    fn sample(&self, fx: f32, fy: f32) -> f32 {
        let fx = fx.max(0.0);
        let fy = fy.max(0.0);
        let x0 = (fx as usize).min(self.w - 1);
        let y0 = (fy as usize).min(self.h - 1);
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let (wx, wy) = (fx - x0 as f32, fy - y0 as f32);
        let top =
            self.d[y0 * self.w + x0] + (self.d[y0 * self.w + x1] - self.d[y0 * self.w + x0]) * wx;
        let bot =
            self.d[y1 * self.w + x0] + (self.d[y1 * self.w + x1] - self.d[y1 * self.w + x0]) * wx;
        top + (bot - top) * wy
    }
}

/// sRGB → 線性光。散射光在線性空間才是加性的，去霧公式必須在這裡算。
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// 建 8-bit sRGB → 線性的查表，避免每像素做 powf
fn srgb_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = srgb_to_linear(i as f32 / 255.0);
    }
    lut
}

/// 診斷用：`SMOKE_TIMING=1` 時把去煙各階段花的時間印到 stderr
/// （拿 smoke_cli 跑一張 1080p 看瓶頸在哪；影片是逐格跑這一套，
/// 這裡省下一成整支就快一成）
fn timing_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SMOKE_TIMING").is_some())
}

/// 印出上一個 `tick` 到現在花的時間並重新計時（見 [`timing_on`]）
fn tick(label: &str, t: &mut std::time::Instant) {
    if timing_on() {
        eprintln!("  [去煙] {label}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        *t = std::time::Instant::now();
    }
}

/// 半徑 r 的方框均值（積分圖，O(n)）。邊界以實際覆蓋面積正規化，
/// 不做 padding，避免邊緣被拉暗。
fn box_mean(p: &Plane, r: usize) -> Plane {
    let (w, h) = (p.w, p.h);
    // 可分離：先橫再直，各用一條滑動和。
    //
    // 原本是積分圖（多一行一列的 f64 前綴和），數學上一樣，但它要先寫出一塊
    // 兩倍大的 f64 陣列——1080p 就是 16MB，而去煙一格會呼叫這裡五十幾次。
    // 影片模組幾十條執行緒同時跑時，記憶體頻寬先被這塊積分圖吃光，加執行緒
    // 完全不會變快（實測 32 條的效率只剩 12%）。滑動和只用一塊 f32 暫存，
    // 且兩趟都是循序存取；邊界一樣是「除以實際涵蓋到的格數」，
    // 所以縱向那趟的每一列橫向視窗相同，兩趟的平均合起來仍等於真正的方框平均。
    //
    // 精度也比積分圖好：積分圖是兩個大數相減，滑動和則只累積視窗內那幾格

    // 橫向的結果**不整張存下來**：縱向那趟同時只看得到 2r+1 列，
    // 就只留這麼多列的環形緩衝（半徑上限 60、工作解析度寬 2560 時約 1.2MB，
    // 待得住在快取裡）。原圖因此只讀一遍、成品只寫一遍，
    // 中間那張跟畫面一樣大的暫存完全不必進記憶體——一格 1080p 的流量
    // 從 41MB 降到 17MB，而這裡是整段去煙搬最多資料的地方
    let mut out = Plane::new(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    let ring_h = (2 * r + 1).min(h);
    let mut ring = vec![0.0f32; ring_h * w];
    // 每一欄的累加值（一整列的 f64）
    let mut col = vec![0.0f64; w];
    // 每個位置的視窗涵蓋幾格，先換算成倒數。除法一個要十幾個週期又不能管線化，
    // 而這裡每個輸出像素都要除一次——一格 1080p 的去煙會做上億次除法。
    // 橫向的分母只跟 x 有關，整張圖共用一份；縱向的只跟 y 有關，一列算一次
    let inv_x: Vec<f64> = (0..w)
        .map(|x| 1.0 / ((x + r + 1).min(w) - x.saturating_sub(r)) as f64)
        .collect();
    // 已經做過橫向濾波的列數 [0, loaded)
    let mut loaded = 0usize;
    for y in 0..h {
        // 先減掉滑出視窗的那一列：它和即將載入的那一列**共用同一個槽**
        // （兩者相距正好 2r+1），順序反過來就會先被蓋掉
        if y > r {
            let s = ((y - r - 1) % ring_h) * w;
            for (c, v) in col.iter_mut().zip(&ring[s..s + w]) {
                *c -= *v as f64;
            }
        }
        let y1 = (y + r + 1).min(h);
        while loaded < y1 {
            let s = (loaded % ring_h) * w;
            {
                // 這一列的橫向滑動和
                let row = &p.d[loaded * w..loaded * w + w];
                let dst = &mut ring[s..s + w];
                let mut sum = 0.0f64;
                // hi＝已經加進來的格數（視窗右界）
                let mut hi = 0usize;
                for x in 0..w {
                    let x1 = (x + r + 1).min(w);
                    while hi < x1 {
                        sum += row[hi] as f64;
                        hi += 1;
                    }
                    if x > r {
                        sum -= row[x - r - 1] as f64;
                    }
                    dst[x] = (sum * inv_x[x]) as f32;
                }
            }
            for (c, v) in col.iter_mut().zip(&ring[s..s + w]) {
                *c += *v as f64;
            }
            loaded += 1;
        }
        let inv = 1.0 / (y1 - y.saturating_sub(r)) as f64;
        let o = &mut out.d[y * w..y * w + w];
        for (v, c) in o.iter_mut().zip(&col) {
            *v = (*c * inv) as f32;
        }
    }
    out
}

/// 半徑 r 的可分離最小值濾波（形態學腐蝕）。
/// 先橫後直，各用單調佇列做 O(n) 滑動極值。
fn min_filter(p: &Plane, r: usize) -> Plane {
    let tmp = extreme_1d_rows(p, r, false);
    let t = transpose(&tmp);
    let t = extreme_1d_rows(&t, r, false);
    transpose(&t)
}

/// 半徑 r 的可分離最大值濾波（形態學膨脹）
fn max_filter(p: &Plane, r: usize) -> Plane {
    let tmp = extreme_1d_rows(p, r, true);
    let t = transpose(&tmp);
    let t = extreme_1d_rows(&t, r, true);
    transpose(&t)
}

/// 半徑 r 的形態學開運算（先腐蝕再膨脹），等同 `max_filter(&min_filter(p, r), r)`。
///
/// 腐蝕與膨脹各自可分離，膨脹的兩個方向又可以對調，於是「腐蝕做完轉回來、
/// 膨脹再轉過去」那兩次轉置可以省掉：橫向腐蝕 → 轉置 → 橫向腐蝕（就是原本的
/// 縱向）→ 橫向膨脹（仍在縱向）→ 轉置 → 橫向膨脹。四次轉置變兩次，
/// 結果逐位元相同（min/max 沒有捨入誤差）。轉置是整塊亂序搬運，
/// 在這一串裡是最傷快取的一步
fn open_filter(p: &Plane, r: usize) -> Plane {
    let a = extreme_1d_rows(p, r, false);
    let t = transpose(&a);
    let t = extreme_1d_rows(&t, r, false);
    let t = extreme_1d_rows(&t, r, true);
    let b = transpose(&t);
    extreme_1d_rows(&b, r, true)
}

/// 轉置。分成小方塊搬：整列讀、跨列寫的話，每寫一個值就踩到一條新的快取線，
/// 一千萬像素的平面等於把整塊記憶體來回刷好幾遍。一次搬 32×32（4KB，
/// 兩邊都塞得進 L1）則讀寫都落在同幾條快取線上
fn transpose(p: &Plane) -> Plane {
    const B: usize = 32;
    let (w, h) = (p.w, p.h);
    let mut out = Plane::new(h, w);
    for y0 in (0..h).step_by(B) {
        let y1 = (y0 + B).min(h);
        for x0 in (0..w).step_by(B) {
            let x1 = (x0 + B).min(w);
            for y in y0..y1 {
                for x in x0..x1 {
                    out.d[x * h + y] = p.d[y * w + x];
                }
            }
        }
    }
    out
}

/// 對每一列做視窗 2r+1 的滑動極值。max=true 取最大值，否則取最小值。
///
/// 用 van Herk–Gil-Werman：把每一列切成長度 2r+1 的區塊，各算一次「由左累積」
/// 與「由右累積」的極值，任何一個視窗都恰好橫跨相鄰兩塊，於是答案就是
/// 「左邊那塊的右累積」與「右邊那塊的左累積」取一次極值。每個元素固定三次比較，
/// 沒有分支、沒有佇列，暫存只有一列的長度（幾 KB，整趟待在 L1）。
///
/// 原本用單調佇列，複雜度一樣是 O(n)，但每個元素都要動到一個堆積上的
/// VecDeque、還帶著不可預測的分支；這一段是去煙裡最花時間的一塊
/// （開運算佔了估煙霧層的四成），常數差很多。
///
/// 邊界的處理與原本相同——視窗被畫面切掉時只看留在裡面的部分：
/// 兩側各補 r 格「極值單位元」（取小補 +∞、取大補 −∞），補出來的永遠選不上，
/// 等同把視窗夾在畫面內
fn extreme_1d_rows(p: &Plane, r: usize, max: bool) -> Plane {
    let (w, h) = (p.w, p.h);
    let mut out = Plane::new(w, h);
    if r == 0 {
        out.d.copy_from_slice(&p.d);
        return out;
    }
    // 比較寫成 if 而不是 f32::min/max：後者要先處理 NaN，這裡的資料不會有
    let pick = |a: f32, b: f32| {
        if max {
            if a >= b {
                a
            } else {
                b
            }
        } else if a <= b {
            a
        } else {
            b
        }
    };
    let id = if max { f32::NEG_INFINITY } else { f32::INFINITY };
    let k = 2 * r + 1;
    // 兩側各補 r 格，再補到區塊的整數倍
    let blocks = (w + 2 * r).div_ceil(k);
    let m = blocks * k;
    let mut ext = vec![id; m];
    // 由左累積／由右累積，各一份
    let mut pre = vec![id; m];
    let mut suf = vec![id; m];
    for y in 0..h {
        // 補的那些格永遠是單位元，不必每列重設，只換中間這一段
        ext[r..r + w].copy_from_slice(&p.d[y * w..y * w + w]);
        for b in 0..blocks {
            let s = b * k;
            let mut acc = ext[s];
            pre[s] = acc;
            for i in 1..k {
                acc = pick(acc, ext[s + i]);
                pre[s + i] = acc;
            }
            let mut acc = ext[s + k - 1];
            suf[s + k - 1] = acc;
            for i in (0..k - 1).rev() {
                acc = pick(acc, ext[s + i]);
                suf[s + i] = acc;
            }
        }
        let o = &mut out.d[y * w..y * w + w];
        for (x, v) in o.iter_mut().enumerate() {
            // 原座標 x 的視窗，在 ext 上就是 [x, x+2r]
            *v = pick(suf[x], pre[x + 2 * r]);
        }
    }
    out
}

/// 導引濾波：以 guide 的邊緣結構重建 src，讓透射率貼齊煙火線條，
/// 消除單純模糊會產生的光暈。
///
/// 這個型別存的是「只跟 guide 有關」的那幾項。
///
/// 估煙霧層時同一輪的三個通道**共用同一張導引圖**（見 [`estimate_smoke`] 第 4 步），
/// 可是 `mean_i`、`mean_ii` 與 guide 的平方都只跟 guide 有關——原本每個通道各算
/// 一次，等於同樣的東西算了三遍。整輪算一次就好：一輪的方框平均因此從 18 次
/// 降到 14 次，省下的還都是最貴的那種（半徑動輒 60）
struct Guide<'a> {
    g: &'a Plane,
    r: usize,
    mean_i: Plane,
    mean_ii: Plane,
}

impl<'a> Guide<'a> {
    fn new(g: &'a Plane, r: usize) -> Self {
        let mut t = std::time::Instant::now();
        let mut ii = Plane::new(g.w, g.h);
        for (v, s) in ii.d.iter_mut().zip(&g.d) {
            *v = s * s;
        }
        let mean_i = box_mean(g, r);
        let mean_ii = box_mean(&ii, r);
        tick("    Guide::new（2 次 box_mean）", &mut t);
        Self {
            g,
            r,
            mean_i,
            mean_ii,
        }
    }

    fn filter(&self, src: &Plane, eps: f32) -> Plane {
        let (w, h) = (self.g.w, self.g.h);
        let n = self.g.d.len();
        let mut ip = Plane::new(w, h);
        for i in 0..n {
            ip.d[i] = self.g.d[i] * src.d[i];
        }
        let mean_p = box_mean(src, self.r);
        let mean_ip = box_mean(&ip, self.r);

        let mut a = Plane::new(w, h);
        let mut b = Plane::new(w, h);
        for i in 0..n {
            let var = self.mean_ii.d[i] - self.mean_i.d[i] * self.mean_i.d[i];
            let cov = mean_ip.d[i] - self.mean_i.d[i] * mean_p.d[i];
            a.d[i] = cov / (var + eps);
            b.d[i] = mean_p.d[i] - a.d[i] * self.mean_i.d[i];
        }
        let mean_a = box_mean(&a, self.r);
        let mean_b = box_mean(&b, self.r);
        let mut out = Plane::new(w, h);
        for i in 0..n {
            out.d[i] = mean_a.d[i] * self.g.d[i] + mean_b.d[i];
        }
        out
    }
}

/// 雙線性放大單通道平面到指定尺寸
fn upscale(p: &Plane, w: usize, h: usize) -> Plane {
    if p.w == w && p.h == h {
        return p.clone();
    }
    let mut out = Plane::new(w, h);
    let sx = p.w as f32 / w as f32;
    let sy = p.h as f32 / h as f32;
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy as usize).min(p.h - 1);
        let y1 = (y0 + 1).min(p.h - 1);
        let wy = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx as usize).min(p.w - 1);
            let x1 = (x0 + 1).min(p.w - 1);
            let wx = fx - x0 as f32;
            let v00 = p.d[y0 * p.w + x0];
            let v01 = p.d[y0 * p.w + x1];
            let v10 = p.d[y1 * p.w + x0];
            let v11 = p.d[y1 * p.w + x1];
            let top = v00 + (v01 - v00) * wx;
            let bot = v10 + (v11 - v10) * wx;
            out.d[y * w + x] = top + (bot - top) * wy;
        }
    }
    out
}

/// 區塊統計（線性光）：估煙霧層要的是各通道最小值，
/// 判斷「這一格裡有沒有煙火線條、佔了多少面積」則要亮度的平均與最大值
struct Blocks {
    /// 各通道最小值
    min: Vec<[f32; 3]>,
    /// 亮度的平均
    mean: Plane,
    /// 亮度的最大值
    max: Plane,
}

/// 逐列吐出 (2r+1)×(2r+1) 方框平均後的像素（邊界複製最外圈）。
///
/// 只給 [`downsample`] 取區塊最小值用：整張抹好存下來要 180MB（六千萬畫素），
/// 逐列算只要留 2r+1 列的橫向窗和。列必須從 0 依序取
struct RowBlur<'a> {
    img: &'a RgbImage,
    r: usize,
    /// 縱向窗裡那 2r+1 列的橫向窗和，最舊的在最前面
    ring: std::collections::VecDeque<Vec<[u32; 3]>>,
    /// 縱向窗裡各欄的總和
    vsum: Vec<[u32; 3]>,
    out: Vec<[u8; 3]>,
    next: usize,
}

impl<'a> RowBlur<'a> {
    fn new(img: &'a RgbImage, r: usize) -> Self {
        let fw = img.width() as usize;
        let mut s = Self {
            img,
            r,
            ring: std::collections::VecDeque::with_capacity(2 * r + 1),
            vsum: vec![[0; 3]; fw],
            out: vec![[0; 3]; fw],
            next: 0,
        };
        // 縱向窗一開始蓋著 -r..=r（負的列複製第 0 列）
        for yy in 0..=2 * r {
            let mut row = vec![[0u32; 3]; fw];
            s.hsum(yy.saturating_sub(r), &mut row);
            for (v, h) in s.vsum.iter_mut().zip(&row) {
                for c in 0..3 {
                    v[c] += h[c];
                }
            }
            s.ring.push_back(row);
        }
        s
    }

    /// 第 `y` 列各欄的橫向窗和（欄超出邊界就複製最外一欄）
    fn hsum(&self, y: usize, out: &mut [[u32; 3]]) {
        let (fw, fh) = (self.img.width() as isize, self.img.height() as usize);
        let y = y.min(fh - 1) as u32;
        let px = |x: isize| self.img.get_pixel(x.clamp(0, fw - 1) as u32, y).0;
        let r = self.r as isize;
        let mut acc = [0u32; 3];
        for d in -r..=r {
            let p = px(d);
            for c in 0..3 {
                acc[c] += p[c] as u32;
            }
        }
        out[0] = acc;
        for x in 1..fw {
            let (drop, add) = (px(x - r - 1), px(x + r));
            for c in 0..3 {
                acc[c] = acc[c] + add[c] as u32 - drop[c] as u32;
            }
            out[x as usize] = acc;
        }
    }

    /// 抹好的第 `y` 列（必須從 0 依序呼叫）
    fn row(&mut self, y: usize) -> &[[u8; 3]] {
        debug_assert_eq!(y, self.next, "RowBlur 的列要依序取");
        self.next = y + 1;
        if y > 0 {
            // 窗往下移一列：丟掉 y-1-r 那列、補進 y+r 那列（超出底邊就複製最後一列）
            let mut row = self.ring.pop_front().expect("窗裡永遠有 2r+1 列");
            for (v, h) in self.vsum.iter_mut().zip(&row) {
                for c in 0..3 {
                    v[c] -= h[c];
                }
            }
            self.hsum(y + self.r, &mut row);
            for (v, h) in self.vsum.iter_mut().zip(&row) {
                for c in 0..3 {
                    v[c] += h[c];
                }
            }
            self.ring.push_back(row);
        }
        let area = ((2 * self.r + 1) * (2 * self.r + 1)) as u32;
        for (o, v) in self.out.iter_mut().zip(&self.vsum) {
            for c in 0..3 {
                o[c] = ((v[c] + area / 2) / area) as u8;
            }
        }
        &self.out
    }
}

/// 下採樣：把原圖切成 ww×wh 塊，每塊算出 [`Blocks`] 要的三組統計。
///
/// 區塊最小值在取之前先把逐像素的感光雜訊抹掉（方框平均，半徑約區塊的一半）。
/// 原尺寸的照片每個像素都帶著雜訊，區塊裡十幾個像素的最小值落在雜訊的谷底，
/// 煙霧層因此被低估兩成上下（六千萬畫素的實照量到 S/I ≈ 0.8），而且雜訊越大
/// 低估越多——不同相機、不同 ISO 都不一樣，靠 [`pool_gain`] 用尺寸推估補不準：
/// 補不夠的那截扣完就留在畫面上，原本煙最亮的地方（噴泉周圍）看得最清楚，
/// 邊上還有一道殘留多寡不同的分界（實照 L1003436 自動判 69 時回報過）。
/// 抹掉雜訊之後最小值取到的才是煙本身的水準，原尺寸估出來的煙霧層與預覽縮圖
/// （縮的時候雜訊早被平均掉）一致，滑桿上的數字兩邊才是同一回事。
///
/// 平均的半徑跟著區塊走：區塊不到一個半像素（縮圖、預覽）就不抹——
/// 那時像素本身已是原圖好幾個像素的平均。半徑只有區塊的一半，煙火線條
/// （原尺寸上寬 8~24px）只被抹寬一點，最小值仍取得到線條旁邊的煙；
/// 平均、最大值兩組統計照原像素算，線條判據要看的是原本的亮點
fn downsample(img: &RgbImage, lut: &[f32; 256], ww: usize, wh: usize) -> Blocks {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let mut b = Blocks {
        min: vec![[1.0f32; 3]; ww * wh],
        mean: Plane::new(ww, wh),
        max: Plane::new(ww, wh),
    };
    let mut n = vec![0u32; ww * wh];
    let block = fw as f32 / ww as f32;
    // 縮圖（預覽、自動判參數用的 768）也至少抹 3×3：預覽與成品估出來的才是同一層煙，
    // 預覽折算強度的指數（AUTO_POOL_EXP）就是照這樣量的。實測 1600 的預覽不抹時
    // 煙霧層反而比原尺寸估得低（強度 60 留 30%，原尺寸留 22%），折算公式補不了
    // 反向的差
    let r_blur = ((block / 2.0).round() as usize).max(1);
    let mut blur = Some(RowBlur::new(img, r_blur));
    for y in 0..fh {
        // 區塊索引直接由座標比例算，可容忍 fw/ww 非整數倍
        let by = (y * wh / fh).min(wh - 1);
        let smooth = blur.as_mut().map(|s| s.row(y));
        for x in 0..fw {
            let bx = (x * ww / fw).min(ww - 1);
            let px = img.get_pixel(x as u32, y as u32);
            let i = by * ww + bx;
            let o = &mut b.min[i];
            let low = smooth.map_or(px.0, |s| s[x]);
            let mut lum = 0.0;
            for (c, w) in [0.2126f32, 0.7152, 0.0722].iter().enumerate() {
                let v = lut[low[c] as usize];
                if v < o[c] {
                    o[c] = v;
                }
                lum += w * lut[px[c] as usize];
            }
            b.mean.d[i] += lum;
            if lum > b.max.d[i] {
                b.max.d[i] = lum;
            }
            n[i] += 1;
        }
    }
    for (v, n) in b.mean.d.iter_mut().zip(n) {
        *v /= n.max(1) as f32;
    }
    b
}

/// 估計煙霧輝光層（三通道線性光，已放大回原尺寸）
fn estimate_smoke(img: &RgbImage, p: &SmokeParams, lut: &[f32; 256]) -> Vec<Plane> {
    let (fw, fh) = (img.width() as usize, img.height() as usize);

    // --- 1. 縮到工作尺寸估煙霧層 ---
    let (ww, wh) = work_size(fw, fh, p.work_edge());
    // 用「最小值池化」而非平均縮圖：平均會把煙火線條的亮度抹進背景，
    // 讓煙霧層被高估、煙火簇整團被當成煙霧削掉。取區塊最小值則等同先做一次
    // 腐蝕，細線在這一步就消失，只有連續的煙霧面留下來。
    let mut t = std::time::Instant::now();
    let blocks = downsample(img, lut, ww, wh);
    tick(&format!("downsample → {ww}×{wh}"), &mut t);
    let lin = &blocks.min;

    // --- 2. 形態學開運算的結構元素 ---
    // 開運算（先腐蝕再膨脹）會移除比結構元素細的亮物件（煙火線條、星點），
    // 只留下大尺度的連續亮面 —— 正是煙霧。單用腐蝕會把整層壓低，
    // 膨脹再把煙霧的原始高度還原回來。
    // 最小值池化已清掉細線，這裡只需小半徑掃掉殘餘的線條交叉點與亮星點。
    // 半徑放大反而會把煙火簇整團當成煙霧，連線條一起削暗。
    let r_open = ((ww.max(wh) as f32 * 0.006).round() as usize).clamp(2, 8);
    let mut guide = Plane::new(ww, wh);
    for (i, c) in lin.iter().enumerate() {
        guide.d[i] = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }

    // --- 3. 導引濾波的參數：把開運算的方塊邊緣抹平，並貼回原圖結構 ---
    // detail 越高 → 半徑越小、eps 越小 → 煙霧層越貼合原圖，煙火線條保留越完整
    let dt = p.detail as f32 / 100.0;
    let r_guide = ((ww.max(wh) as f32 * (0.06 - 0.045 * dt)).round() as usize).clamp(3, 60);
    let eps = 10f32.powf(-3.0 - 2.0 * dt);

    // --- 4. 逐層剝掉 ---
    // 開運算取的是「下包絡」：比結構元素窄的起伏會被削平，煙霧自己的紋理與
    // 煙柱的峰頂因此不在第一層裡，扣完會以斑駁的薄霧留在畫面上——就是看得到的殘留。
    // 把這一層扣掉後剩下的東西再估一次：殘留的起伏這時自己成了連續的亮面，
    // 下一輪的開運算就抓得到；幾層加起來才是完整的煙霧。
    // 煙火線條每一輪都被開運算掃掉，所以再怎麼剝也不會被算進煙霧層裡。
    let mut resid = Vec::with_capacity(3);
    for c in 0..3 {
        let mut ch = Plane::new(ww, wh);
        for (i, px) in lin.iter().enumerate() {
            ch.d[i] = px[c];
        }
        resid.push(ch);
    }
    let mut layer = vec![Plane::new(ww, wh); 3];
    for round in 0..PEEL {
        // 這一輪三個通道共用同一張導引圖，只跟它有關的那幾項先算好（見 [`Guide`]）
        let g = Guide::new(&guide, r_guide);
        for c in 0..3 {
            let s = envelope(&resid[c], &g, r_open, eps);
            for i in 0..ww * wh {
                layer[c].d[i] += s.d[i];
                // 下一輪的輸入＝這一層扣完之後的樣子
                resid[c].d[i] = (resid[c].d[i] - s.d[i]).max(0.0);
            }
        }
        // 導引也換成殘留的亮度，才不會被已經扣掉的結構牽著走
        drop(g);
        for i in 0..ww * wh {
            guide.d[i] = 0.2126 * resid[0].d[i] + 0.7152 * resid[1].d[i] + 0.0722 * resid[2].d[i];
        }
        tick(&format!("peel 第 {} 輪合計", round + 1), &mut t);
    }

    // --- 5. 把沒收斂完的那截補上 ---
    // 剝離是一輪一輪逼近的，濃而孤立的煙團（飄在乾淨夜空上的那種）收斂得慢，
    // 剝完仍剩一截，扣完就成了一朵灰粉色的雲。把剩下的那截補進煙霧層，
    // 濃煙才會真的被壓回夜空的底色。
    //
    // 剩下的那截仍要先過一次開運算：`resid` 裡除了沒收斂完的煙霧，
    // 還有開運算每一輪剔掉的煙火線條本身，整個補回去等於把煙火當成煙霧扣掉。
    // 這一次的結構元素比第 4 步小得多：要掃掉的只剩線條本身（寬不過幾個像素），
    // 用大半徑會連煙霧殘留的紋理一起削平，補不回東西。
    //
    // 這裡不看「附近有沒有煙火」：試過依線條密度少補少扣，實照上行不通——
    // 一朵煙火的線條散得很開，整個煙火周圍連同該扣的煙都會被判成密集，
    // 於是煙火旁邊的煙一律留下來，強度拉到 100 也扣不掉。
    // 該保住的是煙火本身，那由 [`CLIP_KEEP`]（亮到沒細節的芯不壓暗）
    // 與保色相那條路徑（自發光的線條等比例壓暗）各自處理，不必再靠位置猜。
    let r_rest = (r_open / 4).max(2);
    for (c, ch) in layer.iter_mut().enumerate() {
        let rest = open_filter(&resid[c], r_rest);
        for (v, r) in ch.d.iter_mut().zip(&rest.d) {
            *v += r;
        }
    }
    tick("rest（3 次 open_filter）", &mut t);

    let out: Vec<Plane> = layer.iter().map(|s| upscale(s, fw, fh)).collect();
    tick("upscale ×3", &mut t);
    out
}

/// 估煙霧層的工作尺寸：長邊縮到 [`WORK_LONG_EDGE`]，比這還小的照片就原尺寸做
fn work_size(fw: usize, fh: usize, edge: u32) -> (usize, usize) {
    let long = fw.max(fh) as u32;
    let scale = if long > edge {
        edge as f32 / long as f32
    } else {
        1.0
    };
    (
        ((fw as f32 * scale).round() as usize).max(1),
        ((fh as f32 * scale).round() as usize).max(1),
    )
}

/// 煙霧層在每個工作解析度像素上要「少扣多少」0~1：
/// 線條越密就越接近 [`STREAK_MAX`]，乾淨的煙霧面則是 0。
/// 判據自己在夠粗的網格上算（見 [`GATE_BLOCK`]），算完再放大回工作解析度
fn streak_gate(b: &Blocks, fw: usize, ww: usize, wh: usize) -> Plane {
    let f = (GATE_BLOCK * ww).div_ceil(fw.max(1)).max(1);
    let coarse = coarsen(b, f);
    let mut g = streak_fill(&coarse, streak_radius(coarse.mean.w, coarse.mean.h));
    for v in g.d.iter_mut() {
        *v = (*v * STREAK_GAIN).min(STREAK_MAX);
    }
    upscale(&g, ww, wh)
}

/// 整張照片的線條判據（工作解析度）：哪裡是煙火自己、哪裡是沒有紋路的煙霧。
/// 去煙時用它決定哪裡要手下留情，清雲則反過來——它讀到 0 的地方才是「雲朵」
fn streak_plane(img: &RgbImage, edge: u32) -> Plane {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let (ww, wh) = work_size(fw, fh, edge);
    streak_gate(&downsample(img, &srgb_lut(), ww, wh), fw, ww, wh)
}

/// 判斷線條密不密集時要看多大一帶（判據網格的格數）。
/// 大約是長邊的 3%：整叢煙火的尺度，比線條之間的縫隙大得多；
/// 取得夠寬，濃淡的變化才會慢慢過渡而不是一道邊
fn streak_radius(cw: usize, ch: usize) -> usize {
    ((cw.max(ch) as f32 * 0.03).round() as usize).clamp(3, 32)
}

/// 判據網格的一格至少要涵蓋原圖這麼多像素見方。
/// 「線條佔掉多少面積」是格子內的統計量，一格只有一兩個像素就恆等於 0；
/// 工作解析度會隨照片大小變動（小圖甚至不縮），所以判據要自己的網格
const GATE_BLOCK: usize = 8;

/// 把工作解析度的區塊統計併成夠粗的判據網格：每邊合併 `f` 格。
/// min/mean/max 都可以直接由子格的同名統計併出來，不必再讀一次原圖
fn coarsen(b: &Blocks, f: usize) -> Blocks {
    let (ww, wh) = (b.mean.w, b.mean.h);
    let (cw, ch) = (ww.div_ceil(f), wh.div_ceil(f));
    let mut out = Blocks {
        min: vec![[1.0f32; 3]; cw * ch],
        mean: Plane::new(cw, ch),
        max: Plane::new(cw, ch),
    };
    let mut n = vec![0u32; cw * ch];
    for y in 0..wh {
        for x in 0..ww {
            let (i, o) = (y * ww + x, (y / f) * cw + x / f);
            for c in 0..3 {
                if b.min[i][c] < out.min[o][c] {
                    out.min[o][c] = b.min[i][c];
                }
            }
            out.mean.d[o] += b.mean.d[i];
            if b.max.d[i] > out.max.d[o] {
                out.max.d[o] = b.max.d[i];
            }
            n[o] += 1;
        }
    }
    for (v, n) in out.mean.d.iter_mut().zip(n) {
        *v /= n.max(1) as f32;
    }
    out
}

/// 「這一帶被自發光的線條佔掉多少面積」0~1，見 [`estimate_smoke`] 的第 5 步。
///
/// 不能只看亮結構比煙霧底亮多少——疏疏落落的長線條也一樣亮，
/// 但它們之間全是該扣掉的煙。要看的是**佔了多少面積**：
/// 每一格先問「這裡有沒有比煙霧底亮很多的東西」，有的話再問「它佔了這一格幾成」，
/// 兩者相乘後在整叢煙火的尺度上取平均。
/// 一格裡只穿過一條線 → 佔比很低；整叢密集的煙火 → 格格都被線條填滿。
fn streak_fill(b: &Blocks, r: usize) -> Plane {
    let (ww, wh) = (b.mean.w, b.mean.h);
    let mut v = Plane::new(ww, wh);
    for i in 0..ww * wh {
        let lo = 0.2126 * b.min[i][0] + 0.7152 * b.min[i][1] + 0.0722 * b.min[i][2];
        let span = b.max.d[i] - lo;
        // 有沒有：最亮處比煙霧底高出這麼多倍，才算是自發光而不是煙霧自己的起伏。
        // 分母補一個底，乾淨夜空上的雜訊才不會被除成很大的倍率
        let hot = smoothstep(0.5, 2.0, span / (lo + NIGHT_FLOOR));
        // 佔多少：平均落在最小值與最大值之間的位置，就是亮的那些像素的面積比例
        let fill = if span > 1e-6 {
            (b.mean.d[i] - lo) / span
        } else {
            0.0
        };
        v.d[i] = hot * fill;
    }
    // 線條之間的縫隙本身沒有線條，但它與線條同屬一叢煙火，要跟著一起被認出來
    box_mean(&v, r)
}

/// 取一個通道的「平滑下包絡」：開運算掃掉細亮結構，導引濾波把方塊邊緣抹平
/// 並貼回原圖的邊緣，最後夾在原值以下
fn envelope(ch: &Plane, guide: &Guide, r_open: usize, eps: f32) -> Plane {
    let mut t = std::time::Instant::now();
    let opened = open_filter(ch, r_open);
    tick("    envelope/open_filter", &mut t);
    let mut s = guide.filter(&opened, eps);
    tick("    envelope/guided（4 次 box_mean）", &mut t);
    for i in 0..s.d.len() {
        s.d[i] = s.d[i].clamp(0.0, ch.d[i]);
    }
    s
}

/// 診斷用（smoke_cli）：把估出的煙霧層輸出成影像，並回傳其最濃處的顏色
#[allow(dead_code)]
pub fn debug_smoke_layer(img: &RgbImage, params: &SmokeParams) -> (RgbImage, [f32; 3]) {
    let lut = srgb_lut();
    let smoke = estimate_smoke(img, &params.clamped(), &lut);
    let mut a = [0.0f32; 3];
    for (c, s) in smoke.iter().enumerate() {
        a[c] = s.d.iter().copied().fold(0.0f32, f32::max);
    }
    let mut out = RgbImage::new(img.width(), img.height());
    for (i, px) in out.pixels_mut().enumerate() {
        *px = Rgb([
            (linear_to_srgb(smoke[0].d[i].clamp(0.0, 1.0)) * 255.0) as u8,
            (linear_to_srgb(smoke[1].d[i].clamp(0.0, 1.0)) * 255.0) as u8,
            (linear_to_srgb(smoke[2].d[i].clamp(0.0, 1.0)) * 255.0) as u8,
        ]);
    }
    (out, a)
}

/// 筆跡預先刷好的解析度（長邊）。筆跡是軟邊的低頻圖案，
/// 存這個尺寸再取樣就夠細；逐像素去比對折線的每一段則會慢上好幾個數量級
const BRUSH_WORK_EDGE: usize = 1024;

/// 遮色片裡一個「算得出權重」的形狀，座標都已換算成影像的像素座標
enum Field {
    /// 框線 [x0, y0, x1, y1] 與羽化寬度（像素）
    Rect([f32; 4], f32),
    /// 起點、單位方向與長度：投影量 0＝全效果、1＝歸零
    Linear {
        ox: f32,
        oy: f32,
        ux: f32,
        uy: f32,
        len: f32,
    },
    /// 中心與半徑，`inner` 是羽化帶起點的相對半徑（0~1）
    Radial {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        inner: f32,
        invert: bool,
    },
    /// 自動選取的物件：權重直接查它自己那張小圖。
    /// `sx`、`sy` 把像素座標換回 0~1（物件圖存的是相對座標）
    Object { o: Object, sx: f32, sy: f32 },
}

impl Field {
    /// 單軸的過渡權重：[lo, hi] 內為 1，往外 f 個像素平滑降到 0
    fn axis(v: f32, lo: f32, hi: f32, f: f32) -> f32 {
        let d = if v < lo {
            (v - (lo - f)) / f
        } else if v > hi {
            ((hi + f) - v) / f
        } else {
            return 1.0;
        };
        let t = d.clamp(0.0, 1.0);
        // smoothstep：線性過渡在羽化帶兩端會留下看得見的折線
        t * t * (3.0 - 2.0 * t)
    }

    fn at(&self, px: f32, py: f32) -> f32 {
        match *self {
            Field::Rect(b, f) => Self::axis(px, b[0], b[2], f) * Self::axis(py, b[1], b[3], f),
            Field::Linear {
                ox,
                oy,
                ux,
                uy,
                len,
            } => {
                // 投影到拖曳方向上：起點之前整片都是全效果，終點之後全都不動
                let t = ((px - ox) * ux + (py - oy) * uy) / len;
                1.0 - smoothstep(0.0, 1.0, t)
            }
            Field::Radial {
                cx,
                cy,
                rx,
                ry,
                inner,
                invert,
            } => {
                let (dx, dy) = ((px - cx) / rx, (py - cy) / ry);
                let d = (dx * dx + dy * dy).sqrt();
                let w = 1.0 - smoothstep(inner, 1.0, d);
                if invert {
                    1.0 - w
                } else {
                    w
                }
            }
            Field::Object { ref o, sx, sy } => o.at(px * sx, py * sy),
        }
    }
}

/// 預先刷好的筆跡權重圖，連同「影像像素座標→這張圖的座標」的縮放
struct Stamp {
    p: Plane,
    sx: f32,
    sy: f32,
}

impl Stamp {
    fn at(&self, px: f32, py: f32) -> f32 {
        self.p.sample(px * self.sx - 0.5, py * self.sy - 0.5)
    }
}

/// 遮色片：形狀蓋到的地方為 1、沒蓋到為 0，邊界以羽化寬度平滑過渡。
/// 一個形狀都沒有時整張都是 1（＝整張照片都去煙），這時濃度也不作用——
/// 沒畫遮色片是「全部都要」，不是「全部都打八折」。
///
/// 每一個形狀都會**相加**，畫兩個就疊兩次；差別只在「一次加多少」：
///
/// * **筆刷**加的是**濃度**那個值。它是塗上去的東西，同一塊想更濃就再刷一遍
///   （濃度 80% 時塗一筆八成、再塗一筆就滿），這是 Photoshop／Lightroom
///   一貫的手感。同一筆之內自己交疊不算兩次（見 [`stamp`]）。
/// * **框選、線性漸層、放射性漸層、物件**一律加 **100%**，**不吃濃度**。
///   它們是「圈出一塊範圍」，圈到就是要，沒有濃淡可言。
///
/// 相加的結果夾回 0~1，所以幾個範圍疊在一起時，交集處早就滿了、
/// 只有整片的**外緣**還留著羽化的過渡——羽化因此是套在「疊完的範圍」上
struct ShapeMask {
    /// 框選、兩種漸層與物件：一律 100%。數量少，逐像素直接算比預刷成圖準也快
    full: Vec<Field>,
    /// 筆刷：已經在 [`stamp`] 裡把每一筆加起來了，這裡再乘上濃度
    brush: Option<Stamp>,
    /// 筆刷的濃度 0~1（只作用在筆刷上）
    density: f32,
    /// 有沒有任何形狀；都沒有時整張視為 1
    any_add: bool,
}

impl ShapeMask {
    /// `shapes` 必須是 [`SmokeParams::clamped`] 整理過的（矩形已正規化、退化的已剔除）。
    /// `density` 為 0~100
    fn new(shapes: &[Shape], feather: i32, density: i32, fw: usize, fh: usize) -> Self {
        let (fwf, fhf) = (fw as f32, fh as f32);
        let f = feather as f32 / 100.0;
        let mut full = Vec::new();
        let mut brushes: Vec<&Brush> = Vec::new();
        for s in shapes {
            match s {
                Shape::Rect(r) => {
                    let px = [r.x0 * fwf, r.y0 * fhf, r.x1 * fwf, r.y1 * fhf];
                    // 羽化寬度以框的短邊為基準，框拉得再小也不會被過渡帶整個吃掉
                    let short = (px[2] - px[0]).min(px[3] - px[1]).max(1.0);
                    full.push(Field::Rect(px, (short * f * 0.5).max(0.5)));
                }
                Shape::Linear(l) => {
                    let (ox, oy) = (l.x0 * fwf, l.y0 * fhf);
                    let (dx, dy) = (l.x1 * fwf - ox, l.y1 * fhf - oy);
                    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
                    full.push(Field::Linear {
                        ox,
                        oy,
                        ux: dx / len,
                        uy: dy / len,
                        len,
                    });
                }
                Shape::Radial(r) => {
                    let (rx, ry) = ((r.rx * fwf).max(0.5), (r.ry * fhf).max(0.5));
                    // 羽化帶至少要有半個像素寬，否則邊緣會是鋸齒
                    let inner = (1.0 - f).clamp(0.0, 1.0 - 0.5 / rx.min(ry));
                    full.push(Field::Radial {
                        cx: r.cx * fwf,
                        cy: r.cy * fhf,
                        rx,
                        ry,
                        inner,
                        invert: r.invert,
                    });
                }
                Shape::Brush(b) => brushes.push(b),
                // 物件的權重直接查表，不像其他形狀要逐點算幾何。
                // 邊界在選取時就抹柔過了，這裡不再套羽化
                Shape::Object(o) => full.push(Field::Object {
                    o: o.clone(),
                    sx: 1.0 / fwf,
                    sy: 1.0 / fhf,
                }),
            }
        }
        let any_add = !full.is_empty() || !brushes.is_empty();
        Self {
            full,
            brush: stamp(&brushes, f, fw, fh),
            density: (density.clamp(0, 100) as f32) / 100.0,
            any_add,
        }
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        // 一個形狀都沒畫＝整張都要處理，濃度不作用（見型別說明）
        if !self.any_add {
            return 1.0;
        }
        // 取像素中心，框線落在像素邊界時兩側才對稱
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        // 框、漸層與物件一律 100% 相加；筆刷才乘上濃度。
        // 中途不提早收工：筆刷那一份要先乘完濃度才知道有沒有真的滿
        let mut w = 0.0f32;
        for s in &self.full {
            w += s.at(px, py);
        }
        if let Some(b) = &self.brush {
            w += b.at(px, py) * self.density;
        }
        w.clamp(0.0, 1.0)
    }
}

/// 把遮色片形狀畫成一張 `w`×`h` 的權重圖（0~1，逐列排列）。
///
/// 幾何與羽化與去煙霧的遮色片同一套（[`ShapeMask`]），差別只在「一個形狀
/// 都沒畫」的意思：去煙那邊代表整張都要處理（回 1），這裡代表整張都沒被
/// 蓋到（回 0）——疊圖是拿它標「哪裡不要合成」，沒畫就是整張都要合成
/// （見 [`crate::stack::blend`]）
pub fn shape_weights(
    shapes: &[Shape],
    feather: i32,
    density: i32,
    w: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    let cleaned: Vec<Shape> = shapes
        .iter()
        .filter_map(Shape::cleaned)
        .take(MAX_SHAPES)
        .collect();
    let m = ShapeMask::new(&cleaned, feather.clamp(0, 100), density, w, h);
    // 一個形狀都沒畫＝沒有東西被蓋到，不是整張都被蓋到
    if !m.any_add {
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = m.at(x, y);
        }
    }
    out
}

/// 把幾筆筆跡刷成一張權重圖（見 [`BRUSH_WORK_EDGE`]）。`f` 為羽化比例 0~1。
///
/// **一筆之內取最大、筆與筆之間相加**：同一筆繞回來塗到自己不會變濃
/// （否則手抖畫個圈就出現一塊深色），放開滑鼠再塗一筆才會疊上去。
/// 回傳的值因此可能大於 1，由 [`ShapeMask::at`] 乘上濃度後才夾回 0~1
fn stamp(brushes: &[&Brush], f: f32, fw: usize, fh: usize) -> Option<Stamp> {
    if brushes.is_empty() {
        return None;
    }
    let long = fw.max(fh);
    let scale = if long > BRUSH_WORK_EDGE {
        BRUSH_WORK_EDGE as f32 / long as f32
    } else {
        1.0
    };
    let ww = ((fw as f32 * scale).round() as usize).max(1);
    let wh = ((fh as f32 * scale).round() as usize).max(1);
    let mut p = Plane::new(ww, wh);
    // 這一筆自己的權重，刷完再加進 p。每一筆重用同一塊，不必為每筆配一張
    let mut one = Plane::new(ww, wh);
    let long_w = ww.max(wh) as f32;
    for b in brushes {
        one.d.fill(0.0);
        // 半徑以長邊為基準：同一筆設定套在預覽縮圖與原尺寸上才一樣粗
        let r = (b.radius * long_w).max(0.75);
        // 內圈之外開始遞減；至少留 0.75 像素的過渡帶，硬邊會是鋸齒
        let inner = (r * (1.0 - f)).min(r - 0.75).max(0.0);
        let pts: Vec<[f32; 2]> = b
            .pts
            .iter()
            .map(|q| [q[0] * ww as f32, q[1] * wh as f32])
            .collect();
        // 折線的每一段各刷一次，只走它外接矩形內的像素；
        // 單點的一筆（點一下就放開）自己與自己成段，刷出一個圓點
        for i in 0..pts.len() {
            let a = pts[i];
            let c = pts.get(i + 1).copied().unwrap_or(a);
            let x0 = ((a[0].min(c[0]) - r).floor().max(0.0) as usize).min(ww.saturating_sub(1));
            let x1 = ((a[0].max(c[0]) + r).ceil().max(0.0) as usize).min(ww - 1);
            let y0 = ((a[1].min(c[1]) - r).floor().max(0.0) as usize).min(wh.saturating_sub(1));
            let y1 = ((a[1].max(c[1]) + r).ceil().max(0.0) as usize).min(wh - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d = dist_to_segment(x as f32 + 0.5, y as f32 + 0.5, a, c);
                    let w = if d <= inner {
                        1.0
                    } else if d >= r {
                        continue;
                    } else {
                        let t = (r - d) / (r - inner);
                        t * t * (3.0 - 2.0 * t)
                    };
                    let o = &mut one.d[y * ww + x];
                    if w > *o {
                        *o = w;
                    }
                }
            }
        }
        // 這一筆完成，加進累計（筆與筆之間才相加）
        for (acc, v) in p.d.iter_mut().zip(&one.d) {
            *acc += *v;
        }
    }
    Some(Stamp {
        p,
        sx: ww as f32 / fw as f32,
        sy: wh as f32 / fh as f32,
    })
}

/// 點 (px, py) 到線段 a–b 的距離
fn dist_to_segment(px: f32, py: f32, a: [f32; 2], b: [f32; 2]) -> f32 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = vx * vx + vy * vy;
    let t = if l2 > 1e-9 {
        (((px - a[0]) * vx + (py - a[1]) * vy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (dx, dy) = (px - (a[0] + t * vx), py - (a[1] + t * vy));
    (dx * dx + dy * dy).sqrt()
}

/// 顏色比對：算一個像素和「指定的那幾個顏色」有多接近，
/// 相當於 Photoshop 依「顏色範圍」建出來的遮色片。
/// 保護色（命中就不去煙）與雲色（命中就當成天空）都用它。
struct ColorMatch {
    /// 指定的顏色（正規化 sRGB）；空的代表沒有指定任何顏色
    keys: Vec<[f32; 3]>,
    /// 容許距離的內外界
    t0: f32,
    t1: f32,
}

impl ColorMatch {
    fn new(colors: impl Iterator<Item = [u8; 3]>, tolerance: i32) -> Self {
        let keys = colors
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();
        // 距離在 sRGB 空間量（與眼睛看到的「顏色像不像」較接近，
        // 也和 Photoshop 的顏色範圍一致）；最大距離為 √3，正規化到 0~1
        let t1 = tolerance as f32 / 100.0 * 0.9;
        Self {
            keys,
            t0: t1 * 0.6,
            t1,
        }
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// 這個像素「沒命中」的程度：完全命中任一指定色為 0，離每個都夠遠為 1。
    /// 保護色直接拿它當「要去煙的比例」用
    fn at(&self, src: &Rgb<u8>) -> f32 {
        if self.keys.is_empty() {
            return 1.0;
        }
        // 取離最近的那個指定色的距離：命中任一色就算命中
        let mut d = f32::INFINITY;
        for key in &self.keys {
            let dist = (0..3)
                .map(|c| {
                    let v = src[c] as f32 / 255.0 - key[c];
                    v * v
                })
                .sum::<f32>()
                .sqrt()
                / 3f32.sqrt();
            if dist < d {
                d = dist;
            }
        }
        if d <= self.t0 {
            return 0.0;
        }
        if d >= self.t1 || self.t1 <= self.t0 {
            return 1.0;
        }
        // 邊界平滑，避免保護區外圍出現鋸齒狀的硬邊
        let t = (d - self.t0) / (self.t1 - self.t0);
        t * t * (3.0 - 2.0 * t)
    }

    /// 命中程度：1＝就是這個顏色，0＝差得夠遠（沒指定顏色時一律 0）
    fn hit(&self, src: &Rgb<u8>) -> f32 {
        1.0 - self.at(src)
    }
}

/// 雲色遮罩：與吸管吸下來的任一個雲色相近的像素為 1。
/// 比對用原圖的 sRGB——吸管吸的就是原圖上的顏色，
/// 這樣判定不會被前面的去煙結果牽著走。沒吸過雲色則回傳 None
fn cloud_mask(img: &RgbImage, p: &SmokeParams) -> Option<Plane> {
    let keys = ColorMatch::new(p.cloud.iter().flatten().copied(), p.cloud_range);
    if keys.is_empty() {
        return None;
    }
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let mut m = Plane::new(fw, fh);
    for (i, px) in img.pixels().enumerate() {
        m.d[i] = keys.hit(px);
    }
    Some(m)
}

/// 天空遮罩：圈出「雲朵」要處理的範圍。
///
/// 這裡的雲朵指的是**天空裡沒有煙火紋路的東西**——被煙火照亮的煙霧面、
/// 殘餘的輝光，以及夜空上真正的雲，都算。判準因此是「有沒有煙火線條」
/// （見 [`streak_gate`]）而不是「亮不亮」：煙霧被照亮後比夜空亮得多，
/// 用暗面當判準只會把最該處理的那一片整個排除掉。
///
/// 三個條件相乘：
/// 1. 從畫面上緣連得過來——水面同樣平坦，但被岸邊的燈火隔開，傳不過去
/// 2. 這一帶沒有煙火線條（`streaks`）——煙火簇自己與地景的燈火都在這裡出局
/// 3. 亮度在 `range` 之內——多亮的煙霧算雲朵由使用者定，[`auto_params`] 會照這張照片先判一個
///
/// 最後再逐像素把關一次高頻細節，星點與細線就是這樣保住的。
///
/// `streaks` 是工作解析度的線條判據（見 [`streak_plane`]），沒給時退回舊的
/// 「暗而平坦」判法；`cloud` 是吸管指名的雲色遮罩（見 [`cloud_mask`]）：
/// 亮到連放寬範圍都蓋不到的雲，由使用者直接指認，命中的像素亮度不合格也算天空，
/// 但一樣要從畫面上緣連得過來，免得地面上同色的東西被一起壓掉。
/// 去煙的作用範圍：**從畫面上緣連得過來的天空**，1＝是天空、0＝地景。
///
/// 煙只飄在天空，可是「估煙霧層」在地面一樣估得出東西——岸邊燈火與水面倒影
/// 又亮又連續，扣下去整片會被壓暗一層（實測水面的改動量是天空的十倍）。
/// 所以去煙前先框出天空，地景與水面一個像素都不動。
///
/// 和清雲用的 [`sky_mask`] 是兩回事，差在兩點：
///
/// * **煙火算天空**。清雲要避開煙火（那圈光是煙火的、不是煙），但去煙正是要
///   清掉煙火周圍那團煙；把煙火判成非天空，煙火下方的天空還會連不過來。
/// * **不看亮度**。要清的煙本來就被照得很亮，拿亮度當門檻會把它排除掉。
///   分辨天空與地景靠的是「大尺度平不平」與連通性，不是亮不亮。
/// 一維滑動平均（給天際線抹平用）：視窗是 [i-r, i+r]，兩端縮短不補值
fn smooth_1d(v: &[f32], r: usize) -> Vec<f32> {
    let n = v.len();
    let mut pre = vec![0f64; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + v[i] as f64;
    }
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(r);
            let hi = (i + r + 1).min(n);
            ((pre[hi] - pre[lo]) / (hi - lo) as f64) as f32
        })
        .collect()
}

/// 一維滑動最大值（給 [`sky_region`] 的傳播用）：視窗是 [i-r, i+r]。
/// 單調遞減佇列，整列只掃一遍
fn row_max(row: &[f32], r: usize) -> Vec<f32> {
    let w = row.len();
    let mut out = vec![0f32; w];
    let mut dq: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    // 先把右界推到 i，能定案的中心就是 i-r
    for i in 0..w {
        while dq.back().is_some_and(|&b| row[b] <= row[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i >= r {
            let c = i - r;
            while dq.front().is_some_and(|&f| f + r < c) {
                dq.pop_front();
            }
            out[c] = row[dq[0]];
        }
    }
    // 尾端這幾個中心的右界已經到底，只要繼續退左界
    for c in w.saturating_sub(r)..w {
        while dq.front().is_some_and(|&f| f + r < c) {
            dq.pop_front();
        }
        out[c] = row[dq[0]];
    }
    out
}

/// [`sky_region`] 的兩個統計半徑：小的量細節、大的看「這一帶」。
///
/// 半徑照**原圖**的長邊夾好，再等比換算到手上這張的像素上。夾成固定像素是
/// 刻意的——城市的窗、水面的漣漪在感光元件上就是那麼大，量它們要用固定的
/// 像素尺度，不是佔畫面的比例（照比例算，七千萬畫素的照片會把整片市區
/// 判成天空）。但預覽是原圖縮到 1600 的縮圖，同一片紋理只佔幾分之一的
/// 像素：照縮圖自己的尺寸去夾，量到的就不是同一件事——煙火簇那一帶的
/// 紋理統計被拉高，種子判定過不了關，那幾欄從水面一路擋到畫面上緣，中央
/// 長出一根「柱子」，柱子裡完全不去煙，存出來的成品卻是好的（實際回報過
/// 的狀況）。換算過去，預覽與成品才是同一張範圍圖
fn region_radii(long: f32, source_long: f32) -> (usize, usize) {
    let k = long / source_long.max(1.0);
    let r_hi = (((source_long * 0.0015).clamp(1.0, 6.0) * k).round() as usize).max(1);
    let r_area = (((source_long * 0.01).clamp(4.0, 40.0) * k).round() as usize).max(2);
    (r_hi, r_area)
}

/// `source_long`＝原圖的長邊。手上這張就是原圖時傳它自己的長邊
/// `sea`＝水平線（佔畫面高度 0~1，見 [`SkyProbe::sea`]），天空不會比它再低多少；
/// 給 1.0 就不設限
/// [`sky_region`] 的種子：這一帶夠平就是天空（1），有紋理就是地景（0）。
/// 回傳種子與「一帶」的半徑（呼叫端最後抹平用同一個尺度）
fn region_seed(lin: &[[f32; 3]], fw: usize, fh: usize, source_long: f32) -> (Plane, usize) {
    let long = fw.max(fh) as f32;
    let mut y = Plane::new(fw, fh);
    for (i, c) in lin.iter().enumerate() {
        y.d[i] = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    let (r_hi, r_area) = region_radii(long, source_long);
    let blur_hi = box_mean(&y, r_hi);
    let mut detail = Plane::new(fw, fh);
    for i in 0..fw * fh {
        detail.d[i] = (y.d[i] - blur_hi.d[i]).abs();
    }
    let detail_area = box_mean(&detail, r_area);
    let blur_lo = box_mean(&y, r_area);
    // 門檻沿用 [`sky_mask`] 在預設範圍下的那一組——那組已經調到能把岸邊地景與
    // 水面倒影擋在外面（水面同樣暗，卻因為佈滿倒影而紋理偏高）。
    // 放寬過頭會讓整張都算天空，等於沒擋
    let da1 = SKY_AREA_CONTRAST.0 + REGION_RANGE * SKY_AREA_CONTRAST.1;
    let da0 = da1 * 0.25;
    let mut area = Plane::new(fw, fh);
    for i in 0..fw * fh {
        let contrast = detail_area.d[i] / (blur_lo.d[i] + SKY_CONTRAST_FLOOR);
        area.d[i] = 1.0 - smoothstep(da0, da1, contrast);
    }
    // 地景在夜裡多半是「暗底上稀疏的亮點」：城市的燈、樹林間的路燈、遠岸的一排燈。
    // 逐點看紋理只有亮點那幾塊是黑的，其餘一片白，種子圖上一目瞭然（DSC00370 的
    // 城市、B0045103 的樹林）。試過把黑塊撐大成牆：不分亮暗撐，相鄰的煙火簇連成
    // 橫貫整張的牆；只撐暗處、再看黑塊密度，被照亮的煙一樣又暗又有紋理，
    // 牆立在天空裡。所以種子不再加工，地景漏過去的部分交給水平線去擋
    // （見 [`SkyProbe::sea`]）
    (area, r_area)
}

/// 診斷用（smoke_cli）：天空範圍的種子（白＝平、算天空；黑＝有紋理、算地景）
#[allow(dead_code)]
pub fn debug_sky_seed(img: &RgbImage, params: &SmokeParams) -> RgbImage {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let lut = srgb_lut();
    let lin: Vec<[f32; 3]> = img
        .pixels()
        .map(|px| [lut[px[0] as usize], lut[px[1] as usize], lut[px[2] as usize]])
        .collect();
    let (seed, _) = region_seed(&lin, fw, fh, params.source_long(fw.max(fh) as u32));
    let mut out = RgbImage::new(fw as u32, fh as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        let v = (seed.d[i].clamp(0.0, 1.0) * 255.0) as u8;
        *px = Rgb([v, v, v]);
    }
    out
}

/// 天空範圍（[`sky_region`]）在多大的縮圖上算（長邊）。
///
/// 要與 GUI 的預覽底圖同一個尺寸、同一種縮圖濾波（Triangle，見 main.rs 的
/// `shrink_long`）：範圍的種子是紋理判定，同一片岸邊在原尺寸與縮圖上量到的紋理
/// 不是同一回事——實照 A1202725 的暗色漁港在 1600 縮圖上擋得住、原尺寸上擋不住，
/// 原尺寸整張漏到底、被水平線切在 47%，預覽把煙扣乾淨、存檔卻整片留著。
/// 原尺寸與預覽都縮到這個尺寸再算，兩邊算的就是同一張圖，結果自然一致；
/// 範圍是大尺度的東西，算完放大回去就夠。
/// （[`region_radii`] 把半徑照原圖換算，只能讓兩邊「量的尺度」一樣，
/// 量的對象仍是不同解析度的影像，雜訊與銳利度不同，判定就會不同）
const REGION_LONG_EDGE: u32 = 1600;

/// 一張影像的天空範圍（原尺寸的權重平面）：縮到 [`REGION_LONG_EDGE`] 算、再放大回來。
/// `source_long`＝原圖長邊、`sea`＝水平線，同 [`sky_region`]
fn sky_region_of(img: &RgbImage, source_long: f32, sea: f32) -> Plane {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let small = shrink(img, REGION_LONG_EDGE);
    let (sw, sh) = (small.width() as usize, small.height() as usize);
    let lut = srgb_lut();
    let lin: Vec<[f32; 3]> = small
        .pixels()
        .map(|px| [lut[px[0] as usize], lut[px[1] as usize], lut[px[2] as usize]])
        .collect();
    let r = sky_region(&lin, sw, sh, source_long, sea);
    if sw == fw && sh == fh {
        r
    } else {
        upscale(&r, fw, fh)
    }
}

/// `sea`＝水平線（佔畫面高度 0~1，見 [`SkyProbe::sea`]），天空不會比它再低多少；
/// 給 1.0 就不設限
fn sky_region(lin: &[[f32; 3]], fw: usize, fh: usize, source_long: f32, sea: f32) -> Plane {
    let (area, r_area) = region_seed(lin, fw, fh, source_long);
    // 從上緣往下傳播，但「上一列」看的是左右一整段而不是緊鄰的三格。
    //
    // 這是分開「窄的障礙」與「整條橫貫的地景」的關鍵：煙火簇只擋住中間幾欄，
    // 天空從兩側繞過去就流到它下方；岸邊地景橫貫整個畫面，每一欄都被擋住，
    // 左右沒有天空可借，照樣擋得死。
    // 這樣不必再用線條判據去猜哪裡是煙火——那個判據會被水面的倒影騙過去
    let r_gap = ((fw as f32 * REGION_GAP).round() as usize).max(1);
    let mut reach = area.clone();
    for yy in 1..fh {
        let prev = row_max(&reach.d[(yy - 1) * fw..yy * fw], r_gap);
        for x in 0..fw {
            let i = yy * fw + x;
            reach.d[i] = reach.d[i].min(prev[x]);
        }
    }
    // 逐欄取天際線：這一欄最低的天空在哪，它以上就全是天空。
    //
    // 「地景以下不處理」就是這一步。少了它，煙火簇本身（種子判成有紋理）
    // 會在天空中間留一塊不去煙的洞，煙火周圍那團最該清的煙反而清不到
    let cut = fh as f32;
    let mut horizon: Vec<f32> = (0..fw)
        .map(|x| {
            (0..fh)
                .rev()
                .find(|&yy| reach.d[yy * fw + x] > REGION_REACH_MIN)
                .map_or(0.0, |yy| yy as f32 + 1.0)
        })
        .collect();
    // 海口、河口擋不住：地景沒有橫貫整張時，天空從缺口流進水面，再沿著平坦的
    // 水面向左右鋪開，連地景正下方的水面都連得到，每一欄「最低的天空」於是都在
    // 畫面底。水平線由呼叫端量（見 [`SkyProbe::sea`]），比它低太多的欄一律拉
    // 回來——要在向左右借之前做，缺口才不會把「一路到底」借給鄰欄。
    //
    // 每一欄都套：試過只拉「漏到畫面底」的欄、靠紋理擋住的欄不動（想把整片下半
    // 天空都是煙的照片岸邊那截煙也涵蓋進來），被倒影擋在半途的水面欄位就漏掉了，
    // 使用者裁定退回。水平線切太高的照片交給使用者自己用遮色片補
    let sea = (sea.clamp(0.0, 1.0) + REGION_SEA_MARGIN) * fh as f32;
    for h in horizon.iter_mut() {
        *h = h.min(sea);
    }
    // 每一欄再向左右借「最低的那條天際線」。
    //
    // 傳播時每個像素仍受自己的種子把關，所以煙火那幾欄永遠過不了關——
    // 只看自己的話天際線會被煙火頂到它的上緣，煙火周圍那團最該清的煙
    // 反而落在界線下面。向兩側借之後，煙火欄跟著鄰欄的天空一起沉下去；
    // 岸邊地景橫貫整排、左右都沒有更低的天空可借，界線就守得住
    horizon = row_max(&horizon, r_gap);
    // 天際線本身抹平一下，免得一欄高一欄低在畫面上切出鋸齒
    horizon = smooth_1d(&horizon, ((fw as f32 * 0.02).round() as usize).max(1));
    // 依天際線填回整張：界線上下用一小段高度平滑過渡，接縫才不會是一條硬邊
    let feather = (fh as f32 * 0.01).max(2.0);
    let mut out = Plane::new(fw, fh);
    for yy in 0..fh {
        for x in 0..fw {
            let h = horizon[x].min(cut);
            out.d[yy * fw + x] = 1.0 - smoothstep(h - feather, h + feather, yy as f32);
        }
    }
    box_mean(&out, r_area)
}

fn sky_mask(
    lin: &[[f32; 3]],
    fw: usize,
    fh: usize,
    range: i32,
    cloud: Option<&Plane>,
    streaks: Option<&Plane>,
) -> Plane {
    let long = fw.max(fh) as f32;
    let mut y = Plane::new(fw, fh);
    for (i, c) in lin.iter().enumerate() {
        y.d[i] = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    // 高頻細節＝與「很小範圍」的平均的落差。半徑只取長邊的 0.15%，
    // 大半徑會把雲本身的起伏也算成細節，雲就永遠不會被判定成天空
    let r_hi = ((long * 0.0015).round() as usize).clamp(1, 6);
    let blur_hi = box_mean(&y, r_hi);
    let mut detail = Plane::new(fw, fh);
    for i in 0..fw * fh {
        detail.d[i] = (y.d[i] - blur_hi.d[i]).abs();
    }
    // 用平均而非最大值——最大值會讓單顆雜訊點汙染整個鄰域，
    // 並在遮罩上留下方形結構元素的塊狀痕跡
    let r_area = ((long * 0.01).round() as usize).clamp(4, 40);
    // 這一帶整體的紋理與亮度（煙火簇、岸邊地景、水面倒影都偏高）
    let detail_area = box_mean(&detail, r_area);
    let blur_lo = box_mean(&y, r_area);
    // 這個點本身的細節
    let detail_pt = box_mean(&detail, r_hi);
    // 紋理一律換算成「相對於自己有多亮」的對比：絕對落差在亮的地方天生就大，
    // 被煙火照亮的煙霧光是自己的起伏就會被當成有紋理，而那正是要處理的東西。
    // 分母補一個夜空的底，純黑處的雜訊才不會除出一個很大的對比
    let contrast = |d: f32, y: f32| d / (y + SKY_CONTRAST_FLOOR);
    // 雲色遮罩也要兩種尺度：區域的用來當「這一帶是雲」的證據，
    // 逐點的只平滑掉雜訊，雲裡的煙火線條才不會跟著被指認成雲
    let cloud_area = cloud.map(|c| box_mean(c, r_area));
    let cloud_pt = cloud.map(|c| box_mean(c, r_hi));
    // 「這一帶是不是煙火自己」：線條佔比 0～[`STREAK_MAX`]，煙霧面與雲讀到 0。
    // 判據算在自己的粗網格上（見 [`streak_gate`]），這裡直接就近取樣，
    // 不必為了對齊而放大成整張平面——它本來就是慢慢過渡的大尺度統計量
    let burst = |i: usize| match streaks {
        None => 0.0,
        Some(s) => {
            let (x, y) = (i % fw, i / fw);
            let sx = (x * s.w / fw.max(1)).min(s.w - 1);
            let sy = (y * s.h / fh.max(1)).min(s.h - 1);
            (s.d[sy * s.w + sx] / STREAK_MAX).clamp(0.0, 1.0)
        }
    };

    // range 同時放寬亮度與紋理門檻：調高則更亮、更有紋理的煙霧也算雲朵
    let rr = range as f32 / 100.0;
    // 亮度門檻直接對應 sRGB 的 0~255：範圍 50 就是「亮到一半的煙霧也算」，
    // 拉滿則不再管亮度，全交給線條判據與連通性把關。
    // 被煙火照亮的煙霧比夜空亮上兩個數量級，門檻若照線性光等分，
    // 滑桿走到底也還蓋不到它——那正是原本清雲碰不到煙的原因
    let y1 = SKY_Y_BASE + srgb_to_linear(rr);
    let y0 = y1 * 0.3;
    let da1 = SKY_AREA_CONTRAST.0 + rr * SKY_AREA_CONTRAST.1;
    let da0 = da1 * 0.25;
    let dp1 = SKY_PT_CONTRAST.0 + rr * SKY_PT_CONTRAST.1;
    let dp0 = dp1 * 0.25;

    // 第一層：這「一帶」是不是天空（＝沒有煙火紋路的煙霧、雲與夜空）。
    // 煙火簇被線條判據排除，岸邊地景與水面倒影則是紋理或線條兩關過不了
    let mut area = Plane::new(fw, fh);
    for i in 0..fw * fh {
        let dark = 1.0 - smoothstep(y0, y1, blur_lo.d[i]);
        let flat = 1.0 - smoothstep(da0, da1, contrast(detail_area.d[i], blur_lo.d[i]));
        // 這一帶有沒有煙火線條。密集的煙火簇連同它自己的光暈都要留著——
        // 那圈光是煙火的，不是煙；飄開的煙只有零星線條掃過，就交給清雲
        let quiet = 1.0 - smoothstep(SKY_BURST.0, SKY_BURST.1, burst(i));
        area.d[i] = (dark * flat * quiet).max(cloud_area.as_ref().map_or(0.0, |c| c.d[i]));
    }
    // 從畫面上緣往下傳播：只有「從天空一路連過來」的區域才算天空。
    // 水面同樣暗而平坦，但被岸邊地景與船隻的燈火隔開，傳不過去，
    // 否則清雲會把照片下半部的水面一起壓黑。
    //
    // 連通度取「一路上最弱的那一段」而不是連乘：連乘每往下一列就再乘一次，
    // 半透明一點的天空（例如亮度剛好落在範圍邊上的煙霧）幾列之內就被乘成 0，
    // 看起來像是同一片煙霧越往下越不算天空
    for y in 1..fh {
        for x in 0..fw {
            let lo = x.saturating_sub(1);
            let hi = (x + 1).min(fw - 1);
            let above = (lo..=hi).fold(0.0f32, |a, xx| a.max(area.d[(y - 1) * fw + xx]));
            let i = y * fw + x;
            area.d[i] = area.d[i].min(above);
        }
    }
    // 往外擴張再平滑：區域統計會讓煙火簇的影響範圍比簇本身大一圈，
    // 不補回來的話簇周圍會留下一道沒被處理的黑邊，看起來像貼上去的。
    // 擴張進來的部分安不安全，由下面的逐像素條件把關
    let area = box_mean(&max_filter(&area, r_area), r_area);

    // 第二層：逐像素把關。即使身處天空，也只動亮度在範圍內、本身又沒有紋路的點——
    // 煙火的線條與星點就是這樣保住的，而線條之間的空隙仍然算天空
    let mut m = Plane::new(fw, fh);
    for i in 0..fw * fh {
        let dark_pt = 1.0 - smoothstep(y0, y1, blur_hi.d[i]);
        let flat_pt = 1.0 - smoothstep(dp0, dp1, contrast(detail_pt.d[i], blur_hi.d[i]));
        // 同樣取大的：這個點本身就是雲色，亮一點、有點紋理也還是雲
        let sky_pt = (dark_pt * flat_pt).max(cloud_pt.as_ref().map_or(0.0, |c| c.d[i]));
        m.d[i] = area.d[i] * sky_pt;
    }
    // 遮罩本身再平滑一次，邊界才不會出現硬切的塊狀
    box_mean(&m, r_hi * 2)
}

fn smoothstep(e0: f32, e1: f32, v: f32) -> f32 {
    if e1 <= e0 {
        return if v >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 遮色片預覽：把「不會去煙」的區域疊上紅色，
/// 讓使用者看得到框選範圍與顏色保護實際蓋住哪裡（比照 Photoshop 的快速遮色片）
pub fn mask_overlay(img: &RgbImage, params: &SmokeParams) -> RgbImage {
    let p = params.clamped();
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let mask = ShapeMask::new(&p.shapes, p.feather, p.mask_density, fw, fh);
    let protect = ColorMatch::new(p.protect.iter().flatten().copied(), p.tolerance);
    let mut out = img.clone();
    for y in 0..fh {
        for x in 0..fw {
            let px = out.get_pixel_mut(x as u32, y as u32);
            let covered = 1.0 - mask.at(x, y) * protect.at(px);
            if covered <= 0.0 {
                continue;
            }
            let a = covered * 0.55;
            const RED: [f32; 3] = [220.0, 40.0, 60.0];
            for c in 0..3 {
                px[c] = (px[c] as f32 * (1.0 - a) + RED[c] * a)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// 只看形狀的遮色片預覽：把形狀**沒蓋到**的地方疊上紅色（`invert` 為 true 時
/// 反過來，蓋到的地方紅），影片去煙霧的兩份遮色片都用它。
/// 與 [`mask_overlay`] 同一種畫法，差別只在不看保護色。
/// 一個形狀都沒畫時整張都算蓋到（去煙那邊的意思），所以不會塗紅
#[allow(dead_code)]
pub fn shape_overlay(
    img: &RgbImage,
    shapes: &[Shape],
    feather: i32,
    density: i32,
    invert: bool,
) -> RgbImage {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let cleaned: Vec<Shape> = shapes
        .iter()
        .filter_map(Shape::cleaned)
        .take(MAX_SHAPES)
        .collect();
    let mask = ShapeMask::new(&cleaned, feather.clamp(0, 100), density, fw, fh);
    let mut out = img.clone();
    for y in 0..fh {
        for x in 0..fw {
            let mut inside = mask.at(x, y);
            // 反選：只有真的畫了形狀才反過來（沒畫時整張都算蓋到，反過來會整張紅）
            if invert && mask.any_add {
                inside = 1.0 - inside;
            }
            let covered = 1.0 - inside;
            if covered <= 0.0 {
                continue;
            }
            let px = out.get_pixel_mut(x as u32, y as u32);
            let a = covered * 0.55;
            const RED: [f32; 3] = [220.0, 40.0, 60.0];
            for c in 0..3 {
                px[c] = (px[c] as f32 * (1.0 - a) + RED[c] * a)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// 診斷用（smoke_cli）：把天空遮罩輸出成灰階影像（白＝判定為天空）
#[allow(dead_code)]
/// 去煙作用範圍的灰階圖（白＝會去煙的天空、黑＝地景一個像素都不動）。
/// 調 [`sky_region`] 的門檻時最快的檢查方式
/// 找出「天際線以上」的權重圖：1＝天空（含平靜的水面），0＝地景。
///
/// 回傳（權重, 寬, 高）。**刻意在縮小過的圖上算**：天際線是大尺度的東西，
/// 而原尺寸照片攤成 f32 要好幾百 MB，划不來。呼叫端照比例取樣即可。
///
/// 疊圖模組拿它來保護地景（見 [`crate::stack::Guard`]）：天際線以下每一層
/// 拍到的是同一片城市，只是曝光各差一點，用「加亮」疊下去會愈疊愈亮
pub fn sky_weights(img: &RgbImage, long: u32) -> (Vec<f32>, usize, usize) {
    let (w, h) = img.dimensions();
    let s = (long as f32 / w.max(h).max(1) as f32).min(1.0);
    let (sw, sh) = (
        ((w as f32 * s).round() as u32).max(1),
        ((h as f32 * s).round() as u32).max(1),
    );
    let small = if sw == w && sh == h {
        std::borrow::Cow::Borrowed(img)
    } else {
        std::borrow::Cow::Owned(image::imageops::resize(
            img,
            sw,
            sh,
            image::imageops::FilterType::Triangle,
        ))
    };
    let lut = srgb_lut();
    let lin: Vec<[f32; 3]> = small
        .pixels()
        .map(|px| [lut[px[0] as usize], lut[px[1] as usize], lut[px[2] as usize]])
        .collect();
    // 這裡刻意照**縮小後**的尺寸換算半徑（不像預覽那樣折回原圖）：疊圖拿它
    // 保護地景，界線寧可守得寬一點，也不要漏一條縫讓城市越疊越亮
    let r = sky_region(&lin, sw as usize, sh as usize, sw.max(sh) as f32, 1.0);
    (r.d, sw as usize, sh as usize)
}

/// 「天空」遮色片的工作解析度（長邊）。天際線本身是大尺度的東西，幾百像素
/// 就描得夠貼；但這張遮色片還要**避開煙火的線條**，線條是細的，縮太小就糊成
/// 一片認不出來，所以取與預覽天空範圍相同的尺寸（見 [`REGION_LONG_EDGE`]）
const SKY_SELECT_EDGE: u32 = REGION_LONG_EDGE;

/// 逐點判「這裡是煙火紋路」的相對對比門檻（見 [`sky_points`]）：
/// 與 [`sky_mask`] 拉滿範圍時那組相同——那組已經調到「線條與星點擋掉、
/// 被煙火照亮的平順煙霧留著」，亮度則不看（煙火照亮的煙比夜空亮上兩個數量級）
const SKY_POINT_CONTRAST: (f32, f32) = (
    SKY_PT_CONTRAST.0 + SKY_PT_CONTRAST.1,
    (SKY_PT_CONTRAST.0 + SKY_PT_CONTRAST.1) * 0.25,
);

/// 逐點的「不是煙火紋路」權重：1＝平順（夜空、煙、雲），0＝線條或星點。
///
/// 只看紋理不看亮度：細節＝與很小範圍平均的落差，換算成相對於自己亮度的對比，
/// 煙火線條與星點相對它周圍暗得多的天空對比極高，被煙火照亮的煙霧再亮也是平的。
/// 線條往外撐一圈再抹平：遮色片的邊界要離線條一點距離，羽化時才不會又暈回去
fn sky_points(lin: &[[f32; 3]], fw: usize, fh: usize) -> Plane {
    let long = fw.max(fh) as f32;
    let mut y = Plane::new(fw, fh);
    for (i, c) in lin.iter().enumerate() {
        y.d[i] = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    // 半徑與 sky_mask 的 r_hi 同一個算法，兩邊認的「細節」才是同一種東西
    let r_hi = ((long * 0.0015).round() as usize).clamp(1, 6);
    let blur_hi = box_mean(&y, r_hi);
    let mut detail = Plane::new(fw, fh);
    for i in 0..fw * fh {
        detail.d[i] = (y.d[i] - blur_hi.d[i]).abs();
    }
    let detail_pt = box_mean(&detail, r_hi);
    let (dp1, dp0) = SKY_POINT_CONTRAST;
    let mut streak = Plane::new(fw, fh);
    for i in 0..fw * fh {
        let contrast = detail_pt.d[i] / (blur_hi.d[i] + SKY_CONTRAST_FLOOR);
        streak.d[i] = smoothstep(dp0, dp1, contrast);
    }
    let streak = box_mean(&max_filter(&streak, r_hi), r_hi);
    let mut out = Plane::new(fw, fh);
    for i in 0..fw * fh {
        out.d[i] = 1.0 - streak.d[i];
    }
    out
}

/// 自動選取天空：一鍵把「天際線以上」整片圈成一個遮色片形狀
/// （比照 Lightroom 的「選取天空」）。
///
/// 界線用的是「只處理天空」那一套判定（[`sky_region`] 加上水平線
/// [`SkyProbe::sea`]）：從畫面上緣一路連得下來的平順區域算天空，橫貫整排、
/// 佈滿細節的岸邊地景與水面倒影擋在外面。
///
/// **煙火的紋路要避開**（使用者裁定）：天際線以上再乘一次逐點的判定
/// （[`sky_points`]）——煙火的線條與星點擋掉，線條之間與周圍的煙霧、
/// 被煙火照亮的天空都照樣選進來。只避開線條本身、不避開整團煙火與它的光暈：
/// 煙火周圍那團煙正是最該處理的地方。
/// 結果存成一個鋪滿整張的 [`Object`]，羽化與邊緣兩條滑桿照物件那套調
///
/// * `img` 是預覽底圖（原圖等比縮小的那張就夠）
/// * `source_long` 是原圖的長邊：判定天空的統計半徑要照原圖換算，預覽與
///   成品才會圈到同一條線。手上就是原圖（或不知道）時傳 0，照這張自己算
/// * `feather`、`edge` 見 [`Object`]
///
/// 整張都是地景（或整張都是煙、分不出天際線）時回 None
pub fn select_sky(img: &RgbImage, source_long: u32, feather: i32, edge: i32) -> Option<Object> {
    if img.width() < 32 || img.height() < 32 {
        return None;
    }
    let small = shrink(img, SKY_SELECT_EDGE);
    let (w, h) = (small.width() as usize, small.height() as usize);
    let lut = srgb_lut();
    let lin: Vec<[f32; 3]> = small
        .pixels()
        .map(|px| [lut[px[0] as usize], lut[px[1] as usize], lut[px[2] as usize]])
        .collect();
    let source_long = if source_long == 0 {
        img.width().max(img.height())
    } else {
        source_long.max(img.width().max(img.height()))
    };
    // 水平線在整張上量（與去煙時同一個值），天空才不會沿著水面鋪到畫面底
    let sea = sky_probe(img).sea;
    let region = sky_region(&lin, w, h, source_long as f32, sea);
    // 避開煙火紋路：逐點擋掉線條與星點，煙與夜空留著
    let points = sky_points(&lin, w, h);
    let raw: Vec<u8> = region
        .d
        .iter()
        .zip(points.d.iter())
        .map(|(&a, &b)| ((a * b).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let (feather, edge) = (feather.clamp(0, 100), edge.clamp(-100, 100));
    let o = Object {
        mask: std::sync::Arc::new(refine_object(&raw, w, h, feather, edge)),
        raw: std::sync::Arc::new(raw),
        w,
        h,
        area: Region {
            x0: 0.0,
            y0: 0.0,
            x1: 1.0,
            y1: 1.0,
        },
        feather,
        edge,
    };
    o.is_usable().then_some(o)
}

pub fn debug_sky_region(img: &RgbImage, params: &SmokeParams) -> RgbImage {
    let r = sky_region_of(
        img,
        params.source_long(img.width().max(img.height())),
        sky_probe(img).sea,
    );
    let mut out = RgbImage::new(img.width(), img.height());
    for (i, px) in out.pixels_mut().enumerate() {
        let v = (r.d[i].clamp(0.0, 1.0) * 255.0) as u8;
        *px = Rgb([v, v, v]);
    }
    out
}

pub fn debug_sky_mask(img: &RgbImage, params: &SmokeParams) -> RgbImage {
    let p = params.clamped();
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    // 天空判定是在去煙後的影像上做的，診斷也要走同一條路才看得準
    let field = field_of(img, &p);
    let (buf, _, _) = dehaze_to_linear(img, &p, &field);
    let sky = sky_mask(
        &buf,
        fw,
        fh,
        p.sky_range,
        cloud_mask(img, &p).as_ref(),
        Some(&streak_plane(img, p.work_edge())),
    );
    let mut out = RgbImage::new(fw as u32, fh as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        let v = (sky.d[i].clamp(0.0, 1.0) * 255.0) as u8;
        *px = Rgb([v, v, v]);
    }
    out
}

/// 診斷用（smoke_cli）：把「線條佔掉多少面積」的判據輸出成灰階圖。
/// 白＝這一帶被煙火線條填滿、煙霧層會少扣一些；黑＝乾淨的煙霧面，照扣
#[allow(dead_code)]
pub fn debug_streak_map(img: &RgbImage) -> RgbImage {
    let streaks = streak_plane(img, WORK_LONG_EDGE);
    let (ww, wh) = (streaks.w, streaks.h);
    let mut out = RgbImage::new(ww as u32, wh as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        // 直接看「少扣多少」：白＝完全不扣，黑＝照扣
        let v = (streaks.d[i].clamp(0.0, 1.0) * 255.0) as u8;
        *px = Rgb([v, v, v]);
    }
    out
}

// ---------- 自動判參數 ----------
//
// 開檔就替每張照片量一次，滑桿一開始就停在這張自己該有的位置。
// 量的都是這套演算法自己用得到的東西：估出的煙霧層有多濃、天空在哪、
// 煙火線條佔掉多少面積——所以回推的參數與實際處理的結果是同一套判準，
// 而不是另外湊一組經驗公式。
//
// 判準只有一句話：**把天空裡的煙壓回這張照片自己的乾淨夜空**。
// 乾淨夜空有多暗、煙有多濃都是量出來的，所以濃煙的與清朗的會得到不同的
// 強度，而不是一律套同一個數字。

/// 自動判參數的工作解析度（長邊）。要量的都是大尺度的統計量，縮到這裡就夠準；
/// 開一整批照片時每張才不必等上一秒。
/// 改用 1600 量同一張照片，四個值的差距在 1 以內
const AUTO_LONG_EDGE: u32 = 768;

/// 圈「天空連同飄在上面的煙」用的寬鬆判定：量煙要連煙一起圈進來。
/// 不能開到 100——那等於完全不看亮度，白天的照片會整張被當成天空
const AUTO_SKY_WIDE: i32 = 70;
/// 圈「真正乾淨的夜空」用的嚴格判定：扣完該有多暗，以這一帶為準
const AUTO_SKY_CLEAN: i32 = 20;

/// 天空佔畫面不到這個比例就不自動判（近拍、白天的照片都落在這裡）：
/// 統計量沒有立足點，硬給一組數字不如維持預設值
const AUTO_SKY_MIN: f32 = 0.10;

/// 地平線取在「還有這麼多天空」的最後一列（相對於最空的那一列）。
/// 天空遮罩認得的是沒有紋理的暗面，濃煙與煙火簇太亮又太花，一概不在裡面——
/// 可是要量的正是那些煙。所以只拿遮罩定出天空到哪一列為止，這一列以上就整片都算
const AUTO_HORIZON_KEEP: f32 = 0.25;

/// 線條佔比超過這個數的點不拿來回推強度：那裡是煙火自己，
/// 演算法本來就會少扣（見 [`estimate_smoke`] 的第 5 步），量了也不作數
const AUTO_STREAK_MAX: f32 = 0.15;

/// 算「這一帶有多密」時，線條佔比低於這個數的點不列入：
/// 空曠的夜空沒有東西要保護，讓它參與平均只會把密度稀釋成構圖的函數
const AUTO_LINE_MIN: f32 = 0.10;

/// 「濃煙」取在煙霧層亮度的哪個分位數。取得高才問得到真正該扣的那一帶，
/// 太高則只剩幾顆離群點說了算
const AUTO_THICK_Q: f32 = 0.9;

/// 濃煙那一頭回推出來的係數取哪個分位數：中位數。
/// 試過取偏高的 0.75 想把有紋理的煙（實照 A1202720，被風吹成一絲絲的白煙）
/// 也扣乾淨——沒用：縮圖上量不到那種紋理，那張只多了 1，別張卻衝到上限
/// （DSC00370 從 80 跳到 95）。那種煙要靠使用者自己把強度拉高
const AUTO_K_Q: f32 = 0.5;

/// 煙至少要蓋掉地平線以上這麼多面積，才算「這張照片有煙要扣」。
/// 算的是「高出夜空底色一個夜色的量」的像素：煙只飄在煙火那一帶、
/// 天空大半乾淨的照片（實照 DSC00370 只有 1.4%）也要算有煙，門檻不能訂高
const AUTO_SMOKE_COVER: f32 = 0.01;

/// 去除強度的上下限。量出來的值再怎麼極端也夾在這個帶裡：
/// 下限之下等於沒去煙，上限之上會連煙火自己的光暈一起扣掉
const AUTO_STRENGTH: (i32, i32) = (45, 95);

/// 細節的下限與可加上去的幅度：天空裡完全沒有線條就取下限，
/// 線條密到頂（[`STREAK_MAX`]）則加滿
const AUTO_DETAIL: (i32, i32) = (55, 40);

/// 縮圖上量不到原尺寸的雜訊底（見 [`pool_gain`]）。
///
/// 指數是拿實照量的：原尺寸與 1600 縮圖各跑一排強度，在同一團煙上找「殘留一樣」
/// 的配對。最早（取最小值前不抹雜訊）一張四千萬畫素的照片量到 0.15
/// （長邊 8251、3.2 倍：縮圖 46 ↔ 原尺寸 55，3.2^0.15 ≈ 1.19）。
/// [`downsample`] 改成先抹掉雜訊再取最小值後，六千萬畫素（長邊 9528、3.7 倍）
/// 的配對縮到 1.05～1.17（殘留 48%～10% 的那一段：42↔45、52↔55、58↔65、60↔70），
/// 3.7^0.07 ≈ 1.10 落在中間
const AUTO_POOL_EXP: f32 = 0.07;
/// 補償倍率的上限。再大的照片也不該把強度整個推到頂
const AUTO_POOL_MAX: f32 = 1.5;

/// 同樣的強度，在原尺寸上要比在縮圖上多扣幾倍才留下一樣的殘留。
///
/// 原圖大於 [`WORK_LONG_EDGE`] 時，估煙霧層的下採樣取的是區塊最小值：區塊越大
/// 就越常取到感光元件雜訊的谷底，煙霧層被低估，同樣的強度就扣得比較少。縮圖沒有
/// 這一段——縮的時候雜訊早被平均掉了，所以無論縮到 1600 還是 2560，煙都會扣得
/// 比原尺寸乾淨（最早實測強度 80：1600 留 23%、2560 留 22%、原尺寸 8251 留 28%）。
/// [`downsample`] 現在取最小值前會先抹掉雜訊，差距已縮小到一成上下
/// （六千萬畫素實測強度 60：1600 留 10%、原尺寸留 22%；強度 80：0% 對 4%），
/// 剩下的來自煙自己的紋理，仍照這條公式補
///
/// 兩個地方要用到它：[`auto_params`] 拿縮圖量參數，要把回推的係數放大；
/// GUI 拿縮圖畫預覽，則要反過來把強度折小，預覽看到的才是存檔會拿到的東西
fn pool_gain(long: u32) -> f32 {
    pool_gain_at(long, WORK_LONG_EDGE)
}

/// 同上，但指定工作解析度。「速度優先」換掉工作解析度時，同一個強度扣掉的量
/// 會跟著變（區塊變大、最小值取得更低、煙霧層被低估更多），要用這條公式折算
/// 回來，滑桿上的數字才在兩種模式下代表同一件事
fn pool_gain_at(long: u32, edge: u32) -> f32 {
    (long as f32 / edge as f32)
        .max(1.0)
        .powf(AUTO_POOL_EXP)
        .min(AUTO_POOL_MAX)
}

/// 預覽用的強度：把使用者調的 `strength` 折算成「在 `preview_long` 的縮圖上
/// 畫出來，會等於原尺寸 `source_long` 存檔結果」的那個值（見 [`pool_gain`]）。
///
/// 滑桿上的數字因此一律代表**成品**的去煙程度，不會出現預覽比成品乾淨的落差。
/// 存檔走的是原圖，不必也不能折算
pub fn preview_strength(strength: i32, source_long: u32, preview_long: u32) -> i32 {
    let k = pool_gain(preview_long) / pool_gain(source_long);
    ((strength as f32 * k).round() as i32).clamp(0, 100)
}

/// 天空「最亮的那一片」與「最暗的那一級」各取在哪個分位數。
/// 亮的一頭要蓋掉的是成片的雲，取 0.9 等於「至少佔掉一成天空」才算數；
/// 暗的一頭則是這張照片的夜空能有多暗，取極低的分位數而非最小值，
/// 免得單顆暗雜訊點把目標訂到照片上根本不存在的位置
const AUTO_GLOW_Q: f32 = 0.9;
const AUTO_DEEP_Q: f32 = 0.02;

/// 清雲強度的下限與上限。下限同時也是「量不到雲時要用的值」：照片一開進來
/// 清雲就與去除一起生效，天空不必再自己補一刀。訂在這個量級只把殘餘輝光壓掉
/// 一層，星點與薄雲的層次還在；上限則留住一點層次，不把天空壓成純黑
const AUTO_CLEAN: (i32, i32) = (20, 75);

/// 判定範圍的上下限。下限守住的是紋理那一邊：範圍同時放寬亮度與紋理門檻
/// （見 [`sky_mask`]），只照亮度算會訂出一個連夜空的雜訊都過不了的值
const AUTO_RANGE: (i32, i32) = (30, 100);

/// 天空判定門檻要蓋過雲的亮度幾倍。1 倍是剛好判不到、3.3 倍是雲落在全效果區；
/// 取中間讓雲拿到八成左右的權重
const AUTO_RANGE_HEADROOM: f32 = 2.2;

/// 一張照片自動量出來的建議值。只涵蓋四條數值滑桿——遮色片、保護色與
/// 夜空顏色取決於使用者想留下什麼，不是照片本身量得出來的
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AutoParams {
    pub strength: i32,
    pub detail: i32,
    pub sky_clean: i32,
    pub sky_range: i32,
}

impl AutoParams {
    /// 把建議值套進一份既有參數；遮色片、色票等其他設定原樣保留
    pub fn apply_to(&self, p: &mut SmokeParams) {
        p.strength = self.strength;
        p.detail = self.detail;
        p.sky_clean = self.sky_clean;
        p.sky_range = self.sky_range;
    }

    /// 這份參數的四條數值滑桿是不是就停在建議值上（其他設定不看）。
    /// GUI 拿它決定要不要顯示「🪄 回自動預設值」
    #[allow(dead_code)]
    pub fn matches(&self, p: &SmokeParams) -> bool {
        p.strength == self.strength
            && p.detail == self.detail
            && p.sky_clean == self.sky_clean
            && p.sky_range == self.sky_range
    }
}

impl Default for AutoParams {
    /// 量不出來時的退路：三條回到原本的預設值，清雲取下限——量不到雲不代表
    /// 要關著它（見 [`AUTO_CLEAN`]），照片一開進來清雲就與去除一起生效
    fn default() -> Self {
        let d = SmokeParams::default();
        Self {
            strength: d.strength,
            detail: d.detail,
            sky_clean: AUTO_CLEAN.0,
            sky_range: d.sky_range,
        }
    }
}

/// 一組樣本的分位數（q 為 0~1）；沒有樣本就回傳 `alt`
fn quantile(v: &mut [f32], q: f32, alt: f32) -> f32 {
    if v.is_empty() {
        return alt;
    }
    let i = (((v.len() - 1) as f32) * q.clamp(0.0, 1.0)).round() as usize;
    *v.select_nth_unstable_by(i, |a, b| a.total_cmp(b)).1
}

/// 等比縮到長邊不超過 `long`；本來就夠小就原樣複製
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

/// 量一張照片，回推四條滑桿該停在哪裡。
///
/// `source_long` 是**原始照片**的長邊；`img` 可以是縮圖（GUI 就是拿預覽底圖來量），
/// 兩者不同時要靠它補上下採樣的差（見 [`AUTO_POOL_EXP`]）。
/// 直接傳原圖時給 `img.width().max(img.height())` 即可
pub fn auto_params(img: &RgbImage, source_long: u32) -> AutoParams {
    let small = shrink(img, AUTO_LONG_EDGE);
    let (fw, fh) = (small.width() as usize, small.height() as usize);
    if fw < 32 || fh < 32 {
        return AutoParams::default();
    }
    let lut = srgb_lut();
    let lin: Vec<[f32; 3]> = small
        .pixels()
        .map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]])
        .collect();
    let y: Vec<f32> = lin
        .iter()
        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
        .collect();
    let n = fw * fh;

    // --- 1. 天空在哪 ---
    // 線條判據先算，兩份天空遮罩都要用它分辨「煙火自己」與「沒有紋路的煙霧」
    let streaks = streak_plane(&small, WORK_LONG_EDGE);
    // 兩份遮罩各有各的用途：寬鬆的把飄在天上的煙一起圈進來，量煙要用它；
    // 嚴格的只留下真正乾淨的夜空，「扣完該有多暗」要用它
    let sky = sky_mask(&lin, fw, fh, AUTO_SKY_WIDE, None, Some(&streaks));
    let clear = sky_mask(&lin, fw, fh, AUTO_SKY_CLEAN, None, Some(&streaks));
    if sky.d.iter().sum::<f32>() < n as f32 * AUTO_SKY_MIN {
        return AutoParams::default();
    }

    // 乾淨夜空的亮度，也就是扣完之後的目標。取低分位數而非最小值：
    // 單顆暗雜訊點不能決定整張照片的目標
    let mut floor: Vec<f32> = (0..n).filter(|&i| clear.d[i] > 0.5).map(|i| y[i]).collect();
    let target = quantile(&mut floor, 0.2, NIGHT_FLOOR).clamp(NIGHT_FLOOR * 0.15, NIGHT_FLOOR);

    // 地平線：由下往上找最後一列還有夠多天空的（見 [`AUTO_HORIZON_KEEP`]），
    // 中間被煙火整片擋住的列不算數。這一列以上是要量的範圍，以下的地景一律不碰
    let horizon = {
        let row: Vec<f32> = (0..fh)
            .map(|r| sky.d[r * fw..(r + 1) * fw].iter().sum::<f32>() / fw as f32)
            .collect();
        let top = row.iter().copied().fold(0.0f32, f32::max);
        (0..fh)
            .rev()
            .find(|&r| row[r] >= top * AUTO_HORIZON_KEEP)
            .unwrap_or(fh - 1)
    };
    let above = fw * (horizon + 1);

    // --- 2. 細節：煙火線條有多密 ---
    // 線條越密就越要保留：細節調高會讓煙霧層貼緊原圖的邊緣，扣的時候才不會
    // 連線條一起削掉；空曠的煙霧面沒有東西要保，細節放低反而讓煙霧層平順、扣得乾淨。
    // 問的是「有線條的那一帶有多密」而不是「線條佔了畫面幾成」：
    // 天空留白多寡是構圖的事，不該左右要不要保護煙火
    let hot: Vec<f32> = (0..above)
        .map(|i| streaks.d[i])
        .filter(|&v| v > AUTO_LINE_MIN)
        .collect();
    let dense = if hot.is_empty() {
        0.0
    } else {
        hot.iter().sum::<f32>() / hot.len() as f32 / STREAK_MAX
    };
    let detail = (AUTO_DETAIL.0 + (dense.clamp(0.0, 1.0) * AUTO_DETAIL.1 as f32).round() as i32)
        .clamp(0, 100);

    // --- 3. 去除：把煙壓到目標要扣掉多少 ---
    // 直接拿真正的煙霧層反推：某點估出的煙霧層亮度是 ys、原亮度是 y，
    // 要讓 y − k·ys 落在 target 上，k 就是 (y − target)/ys；強度是 k 除以 DEHAZE_GAIN
    let probe = SmokeParams {
        strength: 0,
        detail,
        ..SmokeParams::default()
    };
    let smoke = estimate_smoke(&small, &probe, &lut);
    // 夜空底色：去煙是以它為零點扣的（見 [`floor_from`]），這裡回推強度也要一樣——
    // 煙霧層只算高出底色的那一截，扣完該落在的位置至少是底色
    let floor = floor_from(&smoke, &clear);
    let fy = 0.2126 * floor[0] + 0.7152 * floor[1] + 0.0722 * floor[2];
    let ys: Vec<f32> = (0..n)
        .map(|i| {
            0.2126 * (smoke[0].d[i] - floor[0]).max(0.0)
                + 0.7152 * (smoke[1].d[i] - floor[1]).max(0.0)
                + 0.0722 * (smoke[2].d[i] - floor[2]).max(0.0)
        })
        .collect();
    let target = target.max(fy);
    let smoky = |i: usize| i < above && streaks.d[i] < AUTO_STREAK_MAX;
    // 煙要濃過乾淨夜空的水準才算數，而且要佔得夠廣。整片天空都在這條線以下、
    // 或只有零星幾點過關時，這張照片根本沒有煙可扣，強度取下限就好——
    // 再高也只是把雜訊當成煙一起壓暗
    let mut thick: Vec<f32> = (0..n)
        // ys 已經是高出夜空底色的那一截，「濃過乾淨夜空」就是高出一個夜色的量
        .filter(|&i| smoky(i) && ys[i] > NIGHT_FLOOR)
        .map(|i| ys[i])
        .collect();
    let enough = thick.len() as f32 >= above as f32 * AUTO_SMOKE_COVER;
    // 強度要照濃煙那一頭定：薄霧處 k 幾乎為 0，讓它參與平均會把整張的強度拉垮
    let hi = quantile(&mut thick, AUTO_THICK_Q, 0.0);
    let pool = pool_gain(source_long);
    let strength = if !enough {
        AUTO_STRENGTH.0
    } else {
        let mut ks: Vec<f32> = (0..n)
            .filter(|&i| smoky(i) && ys[i] >= hi)
            .map(|i| ((y[i] - target) / ys[i]).max(0.0))
            .collect();
        let default_k = SmokeParams::default().strength as f32 / 100.0 * DEHAZE_GAIN;
        let k = quantile(&mut ks, AUTO_K_Q, default_k) * pool;
        ((k / DEHAZE_GAIN * 100.0).round() as i32).clamp(AUTO_STRENGTH.0, AUTO_STRENGTH.1)
    };

    // --- 4. 雲朵：把扣完之後仍然不平的天空壓到同一級 ---
    // 去除的目標是「這張照片自己的乾淨夜空」，而那條線取的是低分位數、不是最暗的
    // 那一點——它要套用在整張照片上，訂得太狠會連煙火與地景一起傷到。
    // 天空遮罩裡就不必這麼客氣：那裡只有夜色，可以一路壓到這張照片最暗的夜空。
    // 兩者的落差就是清雲該收的那一截，也正好是眼睛看得出來的「雲」——
    // 天空本來就均勻的照片，落差接近 0，清雲就停在下限（見 [`AUTO_CLEAN`]）：
    // 那是「沒有雲要清」時仍然開著的量，不是量出來有這麼多雲
    //
    // 量的是扣完之後的天空，所以要把 pool 補償先除回去：那個倍率補的是原尺寸
    // 估不足的部分，直接套在縮圖量出來的煙霧層上會把殘留算得比實際少
    let k = strength as f32 / 100.0 * DEHAZE_GAIN / pool;
    let mut left: Vec<f32> = (0..n)
        .filter(|&i| sky.d[i] > 0.5 && streaks.d[i] < AUTO_STREAK_MAX)
        // 量的是「高出底色多少」：清雲壓的也是這一截
        .map(|i| (y[i] - k * ys[i] - fy).max(0.0))
        .collect();
    let deep = quantile(&mut left.clone(), AUTO_DEEP_Q, 0.0).max(NIGHT_FLOOR * 0.15);
    let glow = quantile(&mut left, AUTO_GLOW_Q, 0.0);
    let d = SmokeParams::default();
    // 扣完的天空已經在乾淨夜空的水準（[`NIGHT_FLOOR`]）以下就沒有雲好清，
    // 清雲取下限、範圍用預設值。這個門檻要用絕對值：兩頭都已經趨近全黑時，
    // 它們的比值只是在比雜訊，拿去當強度會在一片乾淨的夜空上憑空調出一個很大的數字
    let (sky_clean, sky_range) = if glow <= NIGHT_FLOOR {
        (AUTO_CLEAN.0, d.sky_range)
    } else {
        // 清雲是把天空乘上 (1 − clean)，要讓最亮的那一成落到最暗那一級就是這個比例。
        // 夾在上下限之間：量到的雲比下限還少時就照下限開著，不歸零
        let clean = ((1.0 - deep / glow) * 100.0).round() as i32;
        // 判定門檻要蓋過這片輝光，否則它根本不會被當成天空，清雲也就碰不到它
        // 判定門檻看的是絕對亮度，底色要加回去
        let range = sky_range_for((glow + fy) * AUTO_RANGE_HEADROOM);
        (
            clean.clamp(AUTO_CLEAN.0, AUTO_CLEAN.1),
            range.clamp(AUTO_RANGE.0, AUTO_RANGE.1),
        )
    };
    AutoParams {
        strength,
        detail,
        sky_clean,
        sky_range,
    }
}

/// 把使用者給的參數整理成真的拿去算的那一份：夾住範圍；「速度優先」的工作解析度
/// 比較小，同樣的強度會扣得少一點，照 pool_gain 的模型折算回來，兩種模式下同一個
/// 滑桿數字才是同一種效果（見 [`pool_gain_at`]）
fn normalized(params: &SmokeParams, img: &RgbImage) -> SmokeParams {
    let mut p = params.clamped();
    if p.fast {
        let long = img.width().max(img.height());
        let k = pool_gain_at(long, FAST_WORK_EDGE) / pool_gain_at(long, WORK_LONG_EDGE);
        p.strength = ((p.strength as f32 * k).round() as i32).clamp(0, 100);
    }
    p
}

/// 去煙裡「跟著整個畫面慢慢變」的那幾樣，事先算好的一份：煙霧層、夜空底色、
/// 天空範圍。它們佔掉一格八成的時間（1080p 實測 585＋116＋68 ms，其餘不到 110 ms）。
///
/// 照片一張算一次就用掉（[`remove_smoke`]）；影片相鄰兩格幾乎一樣，這幾樣可以隔格
/// 算、中間那格用前後兩格的內插（[`SmokeField::lerp`]，見 `movie::dehaze_frames`），
/// 跟細節有關的部分（補回軌跡、亮芯、逐像素相減）仍逐格算，煙火線條一格都不含糊
#[derive(Clone)]
pub struct SmokeField {
    /// 三通道煙霧層（線性光，原尺寸）；強度 0 時是空的
    smoke: Vec<Plane>,
    /// 夜空底色（見 [`floor_from`]）
    floor: [f32; 3],
    /// 天際線以上的權重（原尺寸）；沒勾「只處理天空」時是 None
    sky: Option<Plane>,
    /// 影像尺寸（內插前確認兩份對得上）
    w: usize,
    h: usize,
}

impl SmokeField {
    /// 兩份的中間：`t`＝0 是自己、1 是 `other`。兩份要同尺寸、同一組參數算的
    /// （見 [`same_field_params`]）；對不上就直接回自己那份
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        if self.w != other.w
            || self.h != other.h
            || self.smoke.len() != other.smoke.len()
            || self.sky.is_some() != other.sky.is_some()
        {
            return self.clone();
        }
        let t = t.clamp(0.0, 1.0);
        let mix = |a: &Plane, b: &Plane| {
            let mut o = Plane::new(a.w, a.h);
            for ((o, a), b) in o.d.iter_mut().zip(&a.d).zip(&b.d) {
                *o = a + (b - a) * t;
            }
            o
        };
        SmokeField {
            smoke: self
                .smoke
                .iter()
                .zip(&other.smoke)
                .map(|(a, b)| mix(a, b))
                .collect(),
            floor: std::array::from_fn(|c| self.floor[c] + (other.floor[c] - self.floor[c]) * t),
            sky: self
                .sky
                .as_ref()
                .zip(other.sky.as_ref())
                .map(|(a, b)| mix(a, b)),
            w: self.w,
            h: self.h,
        }
    }
}

/// 兩組參數算出來的 [`SmokeField`] 是不是同一種。遮色片、保護色與強度的大小都
/// 不影響那幾樣（強度只在相減時用），只有這幾項會：內插只能在同一種之間做
pub fn same_field_params(a: &SmokeParams, b: &SmokeParams) -> bool {
    a.detail == b.detail
        && a.sky_only == b.sky_only
        && a.fast == b.fast
        && a.preview_of == b.preview_of
        && (a.strength > 0) == (b.strength > 0)
}

/// 先把慢慢變的那幾樣算好（見 [`SmokeField`]）
pub fn smoke_field(img: &RgbImage, params: &SmokeParams) -> SmokeField {
    field_of(img, &normalized(params, img))
}

/// 用事先算好的那份 [`SmokeField`] 去煙。與 [`remove_smoke`] 的差別只在那幾樣
/// 是誰算的：同一格自己算的那份餵進來，結果逐位元相同
pub fn remove_smoke_with(img: &RgbImage, params: &SmokeParams, field: &SmokeField) -> RgbImage {
    let p = normalized(params, img);
    if p.is_neutral() {
        return img.clone();
    }
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    // 尺寸對不上的那份不能用（呼叫端配錯了），寧可自己重算也不要拿錯的去扣
    let own;
    let field = if field.w == fw && field.h == fh {
        field
    } else {
        own = field_of(img, &p);
        &own
    };
    finish_with(img, &p, field)
}

/// 移除照片中的煙霧，保留煙火細節
pub fn remove_smoke(img: &RgbImage, params: &SmokeParams) -> RgbImage {
    let p = normalized(params, img);
    if p.is_neutral() {
        return img.clone();
    }
    let field = field_of(img, &p);
    finish_with(img, &p, &field)
}

/// [`SmokeField`] 的實作：煙霧層、夜空探測、天空範圍。`p` 要先過 [`normalized`]
fn field_of(img: &RgbImage, p: &SmokeParams) -> SmokeField {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    if p.is_neutral() {
        return SmokeField {
            smoke: Vec::new(),
            floor: [0.0; 3],
            sky: None,
            w: fw,
            h: fh,
        };
    }
    let lut = srgb_lut();
    let mut t = std::time::Instant::now();
    // 只想清雲或改夜空色時強度會是 0，這時不必花時間估煙霧層
    let smoke = if p.strength > 0 {
        estimate_smoke(img, p, &lut)
    } else {
        Vec::new()
    };
    tick("estimate_smoke 合計", &mut t);
    // 先探一次夜空：底色（相減以它為零點，見 [`floor_from`]）與水平線（天空範圍
    // 不比它低，見 [`SkyProbe::sea`]）
    let SkyProbe { floor, sea } = sky_probe(img);
    tick("sky_probe", &mut t);
    // 只在天空去煙：地景與水面沒有煙，卻一樣估得出「煙霧層」，
    // 扣下去只是把岸邊與倒影整片壓暗（見 [`sky_region`]）。強度 0 時不必算
    let sky = (p.sky_only && p.strength > 0)
        .then(|| sky_region_of(img, p.source_long(fw.max(fh) as u32), sea));
    tick("sky_region_of", &mut t);
    SmokeField {
        smoke,
        floor,
        sky,
        w: fw,
        h: fh,
    }
}

/// 去煙的後半：拿算好的場逐像素相減，再做天空處理、轉回 sRGB。`p` 要先過 [`normalized`]
fn finish_with(img: &RgbImage, p: &SmokeParams, field: &SmokeField) -> RgbImage {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let mut t = std::time::Instant::now();
    let (mut buf, weight, floor) = dehaze_to_linear(img, p, field);
    tick("dehaze_to_linear 合計", &mut t);

    if p.touches_sky() {
        // 線條判據要拿原圖算：去煙後的影像線條已經被改過，
        // 拿它判「哪裡是煙火」會比實際少認得一些
        apply_sky(
            &mut buf,
            &weight,
            fw,
            fh,
            &p,
            cloud_mask(img, &p).as_ref(),
            Some(&streak_plane(img, p.work_edge())),
            floor,
        );
        tick("apply_sky", &mut t);
    }

    let mut out = RgbImage::new(fw as u32, fh as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        *px = Rgb([
            (linear_to_srgb(buf[i][0].min(1.0)) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
            (linear_to_srgb(buf[i][1].min(1.0)) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
            (linear_to_srgb(buf[i][2].min(1.0)) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
        ]);
    }
    tick("線性→sRGB 輸出", &mut t);
    out
}

/// 夜空底色（線性光，三通道）：乾淨夜空裡煙霧層的水準。
///
/// 去煙模型原本假設夜空是黑的（I = J + S，J 在夜空處為 0）。煙霧層估的是像素的
/// 下包絡，藍色的夜空會連底色一起被當成煙扣掉：煙濃處扣得多就變黑、遠處扣得少
/// 還留著藍，畫面上一道藍黑分界（實照 B0045103 回報過）。所以先量出夜空自己的
/// 顏色，去煙只扣高出它的那一截，煙散掉的地方回到這個顏色而不是黑。
///
/// `clear` 是「真正乾淨的夜空」遮罩（[`sky_mask`] 在 [`AUTO_SKY_CLEAN`] 下的判定）：
/// 暗、平、從畫面上緣連得過來；濃煙、煙火、地景都不在裡面。整張都是煙、
/// 或天空太亮的照片量不到，回 0，就是原本「夜空是黑的」那條路
fn floor_from(smoke: &[Plane], clear: &Plane) -> [f32; 3] {
    let n = clear.d.len();
    let idx: Vec<usize> = (0..n).filter(|&i| clear.d[i] > 0.5).collect();
    if (idx.len() as f32) < n as f32 * SKY_FLOOR_MIN {
        return [0.0; 3];
    }
    let mut f = [0.0f32; 3];
    for c in 0..3 {
        let mut v: Vec<f32> = idx.iter().map(|&i| smoke[c].d[i]).collect();
        f[c] = quantile(&mut v, SKY_FLOOR_Q, 0.0);
    }
    let y = 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
    if y > SKY_FLOOR_MAX {
        let s = SKY_FLOOR_MAX / y;
        for v in f.iter_mut() {
            *v *= s;
        }
    }
    f
}

/// 一張照片的夜空量出來的兩件事，去煙前先探一次
struct SkyProbe {
    /// 夜空底色（見 [`floor_from`]）
    floor: [f32; 3],
    /// 水平線：天空最低到畫面高度的哪個比例（0~1，1＝量不到、不設限）。
    ///
    /// 水面同樣平坦，[`sky_region`] 的種子擋不住，天空會沿著水面鋪到畫面底，
    /// 倒影被當成煙扣出暗斑（六張實照沒有水平線時全都漏）。用「各欄乾淨夜空
    /// 最低那一列」定線（[`SEA_LEVEL_Q`]）：水面是亮的倒影、地景有紋理，
    /// 都不會被判成乾淨夜空。
    ///
    /// 這條線**寧可估得高一點**：切在真正的水平線上面，頂多是岸邊上方那一截煙
    /// 沒扣到（整片下半天空都是煙的照片會這樣，例如 PS9_1379）；切低了則整片
    /// 地景與水面被當成天空扣煙，那是更明顯的破圖。
    /// 試過另外三條線索（寬鬆天空最低列、地景下緣投票、各欄天際線中位數）
    /// 取最低的一條，想把煙帶也涵蓋進來——實測反而讓多數照片的界線往下跑進地景，
    /// 使用者回報比原本差，已退回單一線索
    sea: f32,
}

/// 在縮圖上量：兩者都是整張的統計量，縮到 [`AUTO_LONG_EDGE`] 就夠準，
/// 預覽與成品也才量到同一個值，原尺寸也不必再算一次天空遮罩
fn sky_probe(img: &RgbImage) -> SkyProbe {
    let none = SkyProbe {
        floor: [0.0; 3],
        sea: 1.0,
    };
    let small = shrink(img, AUTO_LONG_EDGE);
    let (fw, fh) = (small.width() as usize, small.height() as usize);
    if fw < 32 || fh < 32 {
        return none;
    }
    let lut = srgb_lut();
    let lin: Vec<[f32; 3]> = small
        .pixels()
        .map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]])
        .collect();
    let streaks = streak_plane(&small, WORK_LONG_EDGE);
    let clear = sky_mask(&lin, fw, fh, AUTO_SKY_CLEAN, None, Some(&streaks));
    // 乾淨夜空太少就什麼都量不到：底色當黑、水平線不設限
    let clean = clear.d.iter().filter(|&&v| v > 0.5).count();
    if (clean as f32) < (fw * fh) as f32 * SKY_FLOOR_MIN {
        return none;
    }
    let probe = SmokeParams {
        strength: 0,
        ..SmokeParams::default()
    };
    let smoke = estimate_smoke(&small, &probe, &lut);
    // 水平線：各欄乾淨夜空最低到哪，取偏低的分位數
    let mut lows: Vec<f32> = (0..fw)
        .map(|x| {
            (0..fh)
                .rev()
                .find(|&y| clear.d[y * fw + x] > 0.5)
                .map_or(0.0, |y| (y + 1) as f32 / fh as f32)
        })
        .collect();
    SkyProbe {
        floor: floor_from(&smoke, &clear),
        sea: quantile(&mut lows, SEA_LEVEL_Q, 1.0),
    }
}

/// 去煙的主體：回傳線性光的結果、每個像素的作用權重
/// （框選範圍外與命中保護色處為 0，天空處理要沿用同一份權重），
/// 以及這張的夜空底色（清雲要往它壓、不是往黑壓）
fn dehaze_to_linear(
    img: &RgbImage,
    p: &SmokeParams,
    field: &SmokeField,
) -> (Vec<[f32; 3]>, Vec<f32>, [f32; 3]) {
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let lut = srgb_lut();
    let mut t = std::time::Instant::now();
    // 煙霧層、夜空底色與天空範圍都在 field 裡先算好了（見 [`SmokeField`]）
    let smoke = &field.smoke;
    let floor = field.floor;
    let sky = field.sky.as_ref();

    // --- 4. 相減：J = I − k·S ---
    // 煙霧散射光是加性的，直接扣掉即可；煙火線條的亮度是自身發光，
    // 扣掉底下那層煙霧後仍完整保留。
    // 最小值池化取的是區塊下界，估出的煙霧層比實際低一截，
    // 係數放大到 DEHAZE_GAIN 倍才能在強度 100 時把煙霧完全扣乾淨
    let k = p.strength as f32 / 100.0 * DEHAZE_GAIN;
    let mask = ShapeMask::new(&p.shapes, p.feather, p.mask_density, fw, fh);
    let protect = ColorMatch::new(p.protect.iter().flatten().copied(), p.tolerance);
    // 只在天空去煙：地景與水面沒有煙，卻一樣估得出「煙霧層」，
    // 扣下去只是把岸邊與倒影整片壓暗（見 [`sky_region`]）。
    // 強度 0 時不必算——那時什麼都不會扣
    // 原圖的亮度（線性光）。這一份是在**原圖**上量的——相減之後軌跡已經被扣平、
    // 亮芯也被壓暗，那時再量就來不及了；補回軌跡與亮芯判定都讀它
    let lum = (p.strength > 0).then(|| {
        let mut y = Plane::new(fw, fh);
        for (i, px) in img.pixels().enumerate() {
            y.d[i] = 0.2126 * lut[px[0] as usize]
                + 0.7152 * lut[px[1] as usize]
                + 0.0722 * lut[px[2] as usize];
        }
        y
    });
    tick("lum", &mut t);
    // 亮芯的「一帶有多亮」（見 [`CORE_KEEP_AREA`]），已換算成保護權重 0~1，
    // 平台再往外暈開一圈（見 [`CORE_SKIRT`]）
    let core_area = lum.as_ref().map(|y| {
        let long = fw.max(fh) as f32;
        let r = ((long * CORE_AREA_RADIUS).round() as usize).clamp(3, 80);
        // 門檻換算成線性光才不必為每個像素做一次 sRGB 轉換
        let area_lo = srgb_to_linear(CORE_KEEP_AREA.0 / 255.0);
        let area_hi = srgb_to_linear(CORE_KEEP_AREA.1 / 255.0);
        let glow_lo = srgb_to_linear(CORE_SKIRT_GLOW.0 / 255.0);
        let glow_hi = srgb_to_linear(CORE_SKIRT_GLOW.1 / 255.0);
        // 一帶的平均亮度；平台＝亮到門檻以上的那一片
        let area = box_mean(y, r);
        let mut w = area.clone();
        for v in w.d.iter_mut() {
            *v = smoothstep(area_lo, area_hi, *v);
        }
        // 暈開的方式：連抹三次方框平均（三次方框疊起來近似高斯，等高線是圓角的），
        // 再把平台邊緣上的 0.5 拉回 1——平台裡仍是 1，邊緣之外一路平滑降到 0，
        // 兩頭都沒有折角。
        //
        // 之前是「先方形膨脹再方框平均」：暈出來的是一圈帶直邊的方框，斜坡又是
        // 線性的、到底時有個折角，100% 檢視就看得到一道階（實照 L1003436 的
        // 噴泉周圍回報過）
        let r_s = ((long * CORE_SKIRT).round() as usize).clamp(2, 120);
        let s = box_mean(&w, r_s);
        drop(w);
        let mut s = box_mean(&box_mean(&s, r_s), r_s);
        for (i, v) in s.d.iter_mut().enumerate() {
            // 暈開的圈再照「這一帶本身亮不亮」打折（見 [`CORE_SKIRT_GLOW`]），
            // 平台本身則一律保住
            let skirt = smoothstep(0.0, 0.5, *v) * smoothstep(glow_lo, glow_hi, area.d[i]);
            *v = skirt.max(smoothstep(area_lo, area_hi, area.d[i]));
        }
        s
    });
    tick("core_area（4 次 box_mean）", &mut t);
    // 補回煙裡的軌跡：先量出每個像素「比周圍高出多少」
    let excess = (p.restore_trails && p.strength > 0).then(|| {
        let long = fw.max(fh) as f32;
        // 先把比軌跡還細的起伏抹掉再量（見 [`RESTORE_GRAIN`]）：
        // 原尺寸上每個像素都帶著感光雜訊，不抹掉的話下面的「周圍最低點」
        // 量到的是雜訊的谷底，整片煙都像「高出周圍」，被當成軌跡撈回來
        let lum = lum.as_ref().expect("strength > 0 時一定量過亮度");
        let r_grain = if p.preview_of.is_some() {
            // 縮圖（預覽）上一個像素已經是原圖好幾個像素的平均，雜訊早被抹掉：
            // 不到一個像素就不抹，免得把一兩個像素寬的線條壓扁，
            // 預覽的煙火比成品瘦一圈
            (long * RESTORE_GRAIN).floor() as usize
        } else {
            // 原圖每個像素都帶著感光雜訊，再小的照片也至少抹 3×3
            ((long * RESTORE_GRAIN).round() as usize).max(1)
        };
        let mut y = if r_grain > 0 {
            box_mean(lum, r_grain)
        } else {
            lum.clone()
        };
        let r = ((long * RESTORE_RADIUS).round() as usize).clamp(2, 40);
        // 「周圍」是形態學開運算（腐蝕再膨脹）的結果：比結構元素細的東西被掃掉，
        // 比它寬的原樣留著。不取平均、也不是「最低點再抹平」。
        //
        // 密集的煙火——金柳、水面的噴泉、擠成一團的柳枝——比取樣窗還大，
        // 整個窗裡都是它自己：平均會被它自己拉高，高出量算出來接近 0，
        // 軌跡就補不回來，扣完整叢跟著變細變暗（實照上看到的正是這個）。
        // 縫隙裡透出來的才是它底下的煙，腐蝕量到的就是那一層，膨脹再把它
        // 鋪回線條底下——線條本身於是整條「高出周圍」。
        //
        // 之前是「最低點再抹平」：抹平會把縫隙的低值往外暈到旁邊的煙上，
        // 煙自己的紋理（幾十到上百像素寬的一團團）在暈開的底之上就都算
        // 「高出周圍」，扣完的煙火中央浮出一層豹紋般的斑塊。開運算的膨脹
        // 把比結構元素寬的東西整個還原回去，煙的紋理高出量因此是 0，
        // 只有比結構元素細的線條才會被撈起來。
        // 半徑取 r 的一半：抹掉雜訊那一步（[`RESTORE_GRAIN`]）會把線條變寬一截，
        // 結構元素要蓋得過變寬之後的線條，否則線條本身也被當成「周圍」
        let lo = open_filter(&y, (r / 2).max(2));
        for i in 0..fw * fh {
            // 只留「高出周圍」的那一截：低於周圍的地方本來就是煙自己，
            // 一起撈回來等於沒去煙
            let ex = (y.d[i] - lo.d[i]).max(0.0);
            // 再看這一截「相對於周圍有多突出」。煙自己的紋理也高高低低，
            // 但起伏相對於它的亮度很小；軌跡是又細又亮的一條，比值高得多。
            // 不分這一關的話，濃煙區會浮出一層斑駁的煙紋，等於沒去乾淨
            let rel = ex / (lo.d[i] + SKY_CONTRAST_FLOOR);
            y.d[i] = ex * smoothstep(RESTORE_REL.0, RESTORE_REL.1, rel);
        }
        y
    });
    tick("excess（補回軌跡：box_mean + open_filter）", &mut t);
    // 結果先留在線性空間：天空處理要在這上面做，最後才一次轉回 sRGB
    let mut buf: Vec<[f32; 3]> = vec![[0.0; 3]; fw * fh];
    // 每個像素的作用權重，天空處理要沿用（框外與保護色一樣不能動）
    let mut weight: Vec<f32> = vec![0.0; fw * fh];
    for y in 0..fh {
        for x in 0..fw {
            let src = img.get_pixel(x as u32, y as u32);
            // 框選範圍外、命中保護色、或不在天空的像素完全不動，
            // 省下整段運算也保證原樣輸出
            let idx = y * fw + x;
            let w = mask.at(x, y)
                * protect.at(src)
                * sky.as_ref().map_or(1.0, |s| s.d[idx].clamp(0.0, 1.0));
            weight[idx] = w;
            if w <= 0.0 {
                buf[y * fw + x] = [
                    lut[src[0] as usize],
                    lut[src[1] as usize],
                    lut[src[2] as usize],
                ];
                continue;
            }
            let k = k * w;
            // 以夜空底色為零點（見 [`floor_from`]）：像素與煙霧層都先扣掉底色，
            // 後面的相減、保色相、補軌跡全在「高出夜空的那一截」上做，
            // 最後再把底色加回去——煙散掉的地方於是回到夜空原本的顏色，不是壓成黑。
            // 底色為 0（黑夜空）時與原本完全相同
            let i = [
                lut[src[0] as usize] - floor[0],
                lut[src[1] as usize] - floor[1],
                lut[src[2] as usize] - floor[2],
            ];
            let s = if smoke.is_empty() {
                [0.0; 3]
            } else {
                [
                    (smoke[0].d[idx] - floor[0]).max(0.0),
                    (smoke[1].d[idx] - floor[1]).max(0.0),
                    (smoke[2].d[idx] - floor[2]).max(0.0),
                ]
            };
            let smoky = s[0] > 0.0 || s[1] > 0.0 || s[2] > 0.0;

            let mut rgb = [0.0f32; 3];
            for c in 0..3 {
                rgb[c] = i[c] - k * s[c];
            }
            // 通道耦合：某通道被扣到比底色還暗，代表這裡的煙霧估得比實際還濃，
            // 等量從其他通道一併扣掉。否則紫煙的藍色先歸零、紅色留下來，
            // 會在濃煙散去處留下暗紅褐色的斑塊。
            // 只在真的扣了煙的地方做：本來就比底色暗的像素（乾淨夜空較暗的那一半）
            // 不是扣過頭，不能動它
            let m = rgb[0].min(rgb[1]).min(rgb[2]);
            if m < 0.0 && smoky {
                for c in 0..3 {
                    rgb[c] += m;
                }
            }
            // 扣了煙的地方最低到底色為止（夜空在煙的後面，扣不到比它更暗）；
            // 沒扣煙的地方原樣，但無論如何不低於全黑
            let lo = if smoky { 0.0 } else { f32::NEG_INFINITY };
            for c in 0..3 {
                rgb[c] = rgb[c].max(lo).max(-floor[c]);
            }

            // 逐通道相減會改掉煙火線條的顏色：煙霧偏暖，紅通道被扣掉最多，
            // 金黃色的線條就一路偏成橄欖綠。以下三種像素改走「等比例壓暗」——
            // 亮度照扣（縮放係數就是讓亮度剛好掉 k·ys），但三通道等比例，色相原封不動：
            //   1. 過曝像素：某通道已頂到 255，真實亮度被截斷，
            //      逐通道相減會扣掉「看不見的那一截」，把橘紅的亮球算成青綠色。
            //   2. 自發光的煙火線條：亮度遠高於底下那層煙霧。煙霧層是用最小值池化
            //      加開運算估的，細線在那一步就被抹掉，所以線條處的 yi/ys 特別大。
            //   3. 被扣掉一大半的像素（見 [`FADE_KEEP`]）：殘差小到色相已經由
            //      煙霧層的估計誤差說了算。
            // 其餘的——薄煙底下、只扣掉一小截的像素——仍走逐通道相減：那裡殘差夠大，
            // 估計誤差擾不動色相，逐通道相減才能把煙霧壓在物體上的色偏真的扣掉。
            let yi = 0.2126 * i[0] + 0.7152 * i[1] + 0.0722 * i[2];
            let ys = 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2];
            let vmax8 = src[0].max(src[1]).max(src[2]) as f32;
            let clip_w = ((vmax8 - 235.0) / 20.0).clamp(0.0, 1.0);
            // 亮到沒有細節的像素連壓暗都不做：那裡的真實亮度早被感光元件截斷，
            // 扣掉底下那層煙霧只會把煙火最亮的芯變成一團灰，原圖的白就該留成白
            // 亮到沒有細節（頂到 255 那一截）＋ 亮而純白的煙火芯（見
            // [`CORE_KEEP_Y`]）：兩者都是「原樣留著」，取大的
            let vmin8 = src[0].min(src[1]).min(src[2]) as f32;
            let core_w = (smoothstep(CORE_KEEP_Y.0, CORE_KEEP_Y.1, vmax8)
                * smoothstep(
                    CORE_KEEP_NEUTRAL.0,
                    CORE_KEEP_NEUTRAL.1,
                    vmin8 / vmax8.max(1.0),
                ))
            // 暖色的芯過不了「夠白」那一關，改看這一帶亮不亮（見 [`CORE_KEEP_AREA`]）
            .max(core_area.as_ref().map_or(0.0, |a| a.d[idx]));
            // 這個像素比底下的煙霧層亮幾倍；沒有煙霧層就當作無限大
            let above = if ys > 0.0 { yi / ys } else { f32::INFINITY };
            // 過曝的保護給「不是煙霧層本身」（見 [`CLIP_KEEP_ABOVE`]）或
            // 「顏色不夠濃」（見 [`CLIP_KEEP_NEUTRAL`]）的像素：
            // 紅通道頂到 255、又平又紅的濃煙兩關都過不了，一樣是煙，該扣
            let clip_keep = ((vmax8 - CLIP_KEEP) / (255.0 - CLIP_KEEP)).clamp(0.0, 1.0)
                * smoothstep(CLIP_KEEP_ABOVE.0, CLIP_KEEP_ABOVE.1, above).max(smoothstep(
                    CLIP_KEEP_NEUTRAL.0,
                    CLIP_KEEP_NEUTRAL.1,
                    vmin8 / vmax8.max(1.0),
                ));
            let keep_bright = clip_keep.max(core_w);
            // DEHAZE_GAIN 是相減係數 k 的上限：低於這個倍率的像素本來就會被扣成全黑，
            // 保不保色相都一樣，從這裡才開始漸進生效
            let glow_w = if ys > 0.0 {
                ((above - DEHAZE_GAIN) / 1.4).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let fade_w = if yi > 0.0 {
                smoothstep(FADE_KEEP.0, FADE_KEEP.1, k * ys / yi)
            } else {
                0.0
            };
            let keep_hue = clip_w.max(glow_w).max(keep_bright).max(fade_w);
            if keep_hue > 0.0 && yi > 0.0 {
                // 壓暗的幅度隨「亮到沒有細節」的程度收斂到 1.0（完全不壓）
                let scale = (1.0 - k * ys / yi).max(0.0);
                let scale = scale + (1.0 - scale) * keep_bright;
                for c in 0..3 {
                    rgb[c] = rgb[c] * (1.0 - keep_hue) + i[c] * scale * keep_hue;
                }
            }
            // 補回煙裡的軌跡：這一點比周圍高出多少，扣完就至少要留下多少。
            // 濃煙裡的軌跡本來就會被連煙一起扣成 0，這一步把它撈回來；
            // 顏色照原像素的色度給，不會憑空生出別的顏色。
            // 純粹是煙的地方高出量接近 0，所以煙照樣去得乾淨
            if let Some(ex) = &excess {
                // 扣掉越多才補越多（見 [`RESTORE_ON`]）：乾淨夜空幾乎沒被扣，
                // 在那裡補只會把雜訊當成軌跡撈上來
                let att = if yi > 0.0 {
                    (k * ys / yi).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let want = ex.d[idx] * w * smoothstep(RESTORE_ON.0, RESTORE_ON.1, att);
                if want > 0.0 && yi > 0.0 {
                    let out_y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
                    if want > out_y {
                        let s = want / yi;
                        for c in 0..3 {
                            rgb[c] = i[c] * s;
                        }
                    }
                }
            }
            // 把夜空底色加回去，才是絕對亮度
            buf[idx] = [rgb[0] + floor[0], rgb[1] + floor[1], rgb[2] + floor[2]];
        }
    }
    tick("逐像素相減", &mut t);
    (buf, weight, floor)
}

/// 清雲與夜空上色：都在去煙後的影像上做，作用範圍由天空遮罩決定
fn apply_sky(
    buf: &mut [[f32; 3]],
    weight: &[f32],
    fw: usize,
    fh: usize,
    p: &SmokeParams,
    cloud: Option<&Plane>,
    streaks: Option<&Plane>,
    floor: [f32; 3],
) {
    let sky = sky_mask(buf, fw, fh, p.sky_range, cloud, streaks);
    let clean = p.sky_clean as f32 / 100.0;
    let tint = p.sky_tint as f32 / 100.0;
    // 目標夜空色（線性光）與它的亮度。
    // 不能只換色度：清完雲的夜空已經接近純黑，而黑色沒有色度可換，
    // 乘上任何色度都還是黑。夜空的顏色本來就是「天空自己的微光」，
    // 所以改成把這個顏色補上去
    let tint_target = p.sky_color.filter(|_| tint > 0.0).map(|c| {
        let lin = [
            srgb_to_linear(c[0] as f32 / 255.0),
            srgb_to_linear(c[1] as f32 / 255.0),
            srgb_to_linear(c[2] as f32 / 255.0),
        ];
        let y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
        (lin, y)
    });

    for i in 0..fw * fh {
        let s = sky.d[i] * weight[i];
        if s <= 0.0 {
            continue;
        }
        let rgb = &mut buf[i];
        // 清雲：把天空往夜空底色壓，雲與殘餘輝光跟著消失。
        // 壓的是高出底色的那一截——藍色的夜空清完仍是藍的，不是黑的；
        // 本來就比底色暗的地方不動（見 [`floor_from`]）
        if clean > 0.0 {
            let k = 1.0 - clean * s;
            for c in 0..3 {
                rgb[c] = floor[c] + (rgb[c] - floor[c]).max(0.0) * k;
            }
        }
        // 上色：把夜空色補到天空上。已經比目標色亮的地方（星點、刻意留下的
        // 薄雲）幾乎不受影響，整片天空才不會被填成一塊死板的純色
        if let Some((target, ty)) = tint_target {
            if ty > 1e-6 {
                let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
                let deficit = (1.0 - y / ty).clamp(0.0, 1.0);
                let a = tint * s * deficit;
                for c in 0..3 {
                    rgb[c] += target[c] * a;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 測試用的夜景：上半漸層的夜空罩著一層煙、一顆亮球，下段是有紋理的地景
    fn field_scene(shift: u8) -> RgbImage {
        let (w, h) = (160u32, 96u32);
        RgbImage::from_fn(w, h, |x, y| {
            let mut v = if y < 60 {
                // 煙：由上往下變亮的一片灰
                40 + (y * 2) as u8
            } else {
                // 地景：格子紋理
                (60 + ((x / 6 + y / 6) % 5) as u8 * 25) as u8
            };
            let (dx, dy) = (x as i32 - 80, y as i32 - 30);
            if dx * dx + dy * dy < 100 {
                v = 240;
            }
            let v = v.saturating_add(shift);
            Rgb([v, v.saturating_sub(5), v.saturating_sub(10)])
        })
    }

    /// 事先算好的場餵回去，要與一氣呵成的 remove_smoke 逐位元相同；
    /// 兩份場的內插要落在中間；尺寸對不上的兩份不能混，回自己那份
    #[test]
    fn a_precomputed_field_reproduces_remove_smoke_exactly() {
        let img = field_scene(0);
        let p = SmokeParams {
            strength: 70,
            detail: 50,
            ..SmokeParams::default()
        };
        let direct = remove_smoke(&img, &p);
        let field = smoke_field(&img, &p);
        let via = remove_smoke_with(&img, &p, &field);
        assert_eq!(direct.as_raw(), via.as_raw(), "同一格自己算的場餵回去結果該一模一樣");
        assert_eq!(field.smoke.len(), 3, "強度大於 0 要有三通道煙霧層");
        assert!(field.sky.is_some(), "預設只處理天空，要有天空範圍");

        // 內插：亮一點的同一景，中間那份的每一項都要落在兩者中間
        let other = smoke_field(&field_scene(30), &p);
        let mid = field.lerp(&other, 0.5);
        for c in 0..3 {
            let want = (field.floor[c] + other.floor[c]) / 2.0;
            assert!((mid.floor[c] - want).abs() < 1e-6, "底色沒內插到中間");
        }
        let i = 20 * 160 + 80;
        let want = (field.smoke[1].d[i] + other.smoke[1].d[i]) / 2.0;
        assert!((mid.smoke[1].d[i] - want).abs() < 1e-5, "煙霧層沒內插到中間");
        assert!((field.lerp(&other, 0.0).smoke[1].d[i] - field.smoke[1].d[i]).abs() < 1e-6);
        assert!((field.lerp(&other, 1.0).smoke[1].d[i] - other.smoke[1].d[i]).abs() < 1e-6);

        // 尺寸不同的兩份對不上：回自己那份
        let small = smoke_field(
            &image::imageops::resize(&img, 80, 48, image::imageops::FilterType::Triangle),
            &p,
        );
        let kept = field.lerp(&small, 0.5);
        assert_eq!((kept.w, kept.h), (160, 96));
        assert_eq!(kept.smoke[0].d, field.smoke[0].d);

        // 哪些參數不影響場：遮色片、保護色與強度的大小；細節、只處理天空、速度優先會
        let mut q = p.clone();
        q.strength = 30;
        q.shapes.push(Shape::Rect(Region {
            x0: 0.1,
            y0: 0.1,
            x1: 0.5,
            y1: 0.5,
        }));
        assert!(same_field_params(&p, &q));
        assert!(!same_field_params(&p, &SmokeParams { detail: 51, ..p.clone() }));
        assert!(!same_field_params(&p, &SmokeParams { sky_only: false, ..p.clone() }));
        assert!(!same_field_params(&p, &SmokeParams { strength: 0, ..p.clone() }));
    }

    /// 產生一張純色圖
    fn solid(w: u32, h: u32, c: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(c))
    }

    #[test]
    fn boxing_an_object_traces_it_and_leaves_the_sky_alone() {
        // 「物件」工具：框住一團東西，程式沿著它自己的輪廓選，
        // 框裡的夜空與框外的東西都不該被一起選走
        let (w, h) = (200u32, 140u32);
        let mut img = solid(w, h, [12, 14, 20]); // 乾淨夜空
        let blob = |img: &mut RgbImage, cx: f32, cy: f32, r: f32, c: [u8; 3]| {
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                    if (dx * dx + dy * dy).sqrt() < r {
                        *img.get_pixel_mut(x, y) = Rgb(c);
                    }
                }
            }
        };
        blob(&mut img, 50.0, 70.0, 25.0, [220, 180, 120]); // 左邊那團
        blob(&mut img, 150.0, 70.0, 25.0, [210, 170, 110]); // 右邊那團（中間隔著夜空）

        let at = |x: f32, y: f32| (x / w as f32, y / h as f32);
        // 框住左邊那一團（框比它大一圈就好，貼合的事交給程式）
        let sel = Region {
            x0: 15.0 / w as f32,
            y0: 35.0 / h as f32,
            x1: 85.0 / w as f32,
            y1: 105.0 / h as f32,
        };
        let o = select_object(&img, sel, 0, 0).expect("框住一團東西要選得出來");
        let (cu, cv) = at(50.0, 70.0);
        assert!(o.at(cu, cv) > 0.8, "團的正中央要在選取內");
        // 框裡剩下的夜空、框外的另一團都不算數
        let (su, sv) = at(20.0, 40.0);
        assert!(o.at(su, sv) < 0.2, "框裡的夜空不該被一起選走");
        let (ou, ov) = at(150.0, 70.0);
        assert!(o.at(ou, ov) < 0.2, "框外那一團不該被選走");

        // 框在乾淨的夜空上：框裡框外長得一模一樣，分不出東西來——
        // 與其交出半片隨機的選取，不如讓 UI 提示使用者重框一次
        let empty = Region {
            x0: 0.3,
            y0: 0.02,
            x1: 0.7,
            y1: 0.2,
        };
        assert!(
            select_object(&img, empty, 0, 0).is_none(),
            "框在乾淨夜空上不該選出一整片"
        );
    }

    #[test]
    fn a_dark_patch_inside_the_object_is_selected_too() {
        // 大樓沒打燈的那一面、煙火線條之間的夜空——被物件整個包住的暗處
        // 也是這個東西的一部分，要一起選進來（見 select_object 的補洞）
        let (w, h) = (200u32, 140u32);
        let mut img = solid(w, h, [12, 14, 20]);
        for y in 40..100 {
            for x in 60..140 {
                *img.get_pixel_mut(x, y) = Rgb([220, 180, 120]);
            }
        }
        // 正中央挖一塊與夜空同色的全黑
        for y in 60..80 {
            for x in 85..115 {
                *img.get_pixel_mut(x, y) = Rgb([12, 14, 20]);
            }
        }
        let sel = Region {
            x0: 0.22,
            y0: 0.2,
            x1: 0.78,
            y1: 0.85,
        };
        let o = select_object(&img, sel, 0, 0).expect("框住方塊要選得出來");
        assert!(
            o.at(100.0 / w as f32, 70.0 / h as f32) > 0.8,
            "被包在裡面的暗塊也要一起選進來"
        );
        // 框裡但在方塊外的夜空仍然不算——補的是洞，不是把整個框填滿
        assert!(
            o.at(50.0 / w as f32, 70.0 / h as f32) < 0.2,
            "方塊外的夜空不該被選走"
        );
    }

    #[test]
    fn feather_and_edge_reshape_the_same_selection() {
        // 羽化與邊緣是選完之後才調的：分割結果留著不動，
        // 只重算實際要用的那張權重圖（見 Object::raw）
        let (w, h) = (200u32, 140u32);
        let mut img = solid(w, h, [12, 14, 20]);
        for y in 40..100 {
            for x in 60..140 {
                *img.get_pixel_mut(x, y) = Rgb([220, 180, 120]);
            }
        }
        let sel = Region {
            x0: 0.22,
            y0: 0.2,
            x1: 0.78,
            y1: 0.85,
        };
        let sharp = select_object(&img, sel, 0, 0).expect("框住方塊要選得出來");
        // 邊界外面一點點：羽化會讓它從 0 暈開，沒羽化則還是 0
        let (eu, ev) = (142.0 / w as f32, 70.0 / h as f32);
        let soft = sharp.refined(60, 0);
        assert!(sharp.at(eu, ev) < 0.05, "沒羽化時邊界外面應該是 0");
        assert!(soft.at(eu, ev) > 0.05, "羽化要讓邊界往外暈開");
        // 邊緣往內收：原本在裡面一點點的地方會被收掉
        let (iu, iv) = (137.0 / w as f32, 70.0 / h as f32);
        let shrunk = sharp.refined(0, -100);
        assert!(sharp.at(iu, iv) > 0.8, "沒收邊時邊界內側應該是選取內");
        assert!(shrunk.at(iu, iv) < sharp.at(iu, iv), "邊緣往內收要把邊界縮進去");
        // 分割本身沒重跑：中心一律還在選取內
        let (cu, cv) = (100.0 / w as f32, 70.0 / h as f32);
        for o in [&soft, &shrunk] {
            assert!(o.at(cu, cv) > 0.8, "調羽化／邊緣不該動到中心");
        }
    }

    #[test]
    fn trails_buried_in_thick_smoke_come_back() {
        // 濃煙很亮時估出來的煙霧層比煙裡的軌跡還高，相減會把兩者一起扣成 0，
        // 畫面上就是煙火被咬掉一塊。軌跡的訊號其實還在原圖裡（騎在煙上面），
        // 「補回煙裡的軌跡」就是把它撈回來。
        //
        // 另一邊同樣要顧：純粹是煙的地方不能跟著被撈上來，否則等於沒去煙
        let (w, h) = (200u32, 140u32);
        // 整片又亮又平的煙
        let mut img = solid(w, h, [210, 150, 96]);
        // 上面壓幾條細的亮軌跡（只比煙亮一點點，正是會被扣掉的那種）
        for y in 0..h {
            for x in 0..w {
                if x % 20 == 0 || x % 20 == 1 {
                    *img.get_pixel_mut(x, y) = Rgb([245, 196, 140]);
                }
            }
        }
        let run = |restore: bool| {
            let out = remove_smoke(
                &img,
                &SmokeParams {
                    strength: 100,
                    restore_trails: restore,
                    sky_only: false,
                    ..Default::default()
                },
            );
            let lum = |p: [u8; 3]| {
                0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
            };
            // 軌跡上與軌跡旁的煙，各取畫面中間一點
            (
                lum(out.get_pixel(100, 70).0),
                lum(out.get_pixel(110, 70).0),
            )
        };
        let (trail_off, smoke_off) = run(false);
        let (trail_on, smoke_on) = run(true);

        assert!(
            trail_on > trail_off + 8.0,
            "補回軌跡要讓煙裡的軌跡亮回來：關 {trail_off:.1} → 開 {trail_on:.1}"
        );
        assert!(
            smoke_on < smoke_off + 6.0,
            "純粹是煙的地方不該跟著被撈上來：關 {smoke_off:.1} → 開 {smoke_on:.1}"
        );
    }

    /// 原尺寸的照片每個像素都帶著感光雜訊：補回軌跡量「高出周圍多少」時要先把
    /// 雜訊抹掉，否則整片煙都「高出」雜訊谷底一截，被當成軌跡撈回來——
    /// 實照上就是一層斑駁的殘留，預覽縮圖（雜訊已平均掉）卻看不出來
    /// （見 [`RESTORE_GRAIN`]）
    #[test]
    fn noisy_smoke_is_not_mistaken_for_trails() {
        let (w, h) = (240u32, 160u32);
        let mut img = RgbImage::new(w, h);
        // 一片平順的煙，加上逐像素 ±24 的雜訊（線性同餘亂數，每次跑都一樣）
        let mut seed = 12345u32;
        for y in 0..h {
            for x in 0..w {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((seed >> 8) % 49) as i32 - 24;
                let px = [180 + n, 120 + n * 2 / 3, 80 + n / 2];
                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        px[0].clamp(0, 255) as u8,
                        px[1].clamp(0, 255) as u8,
                        px[2].clamp(0, 255) as u8,
                    ]),
                );
            }
        }
        let mean = |i: &RgbImage| {
            i.pixels()
                .map(|p| p.0.iter().map(|&v| v as f64).sum::<f64>())
                .sum::<f64>()
                / (i.width() * i.height() * 3) as f64
        };
        let run = |restore: bool| {
            mean(&remove_smoke(
                &img,
                &SmokeParams {
                    strength: 100,
                    restore_trails: restore,
                    sky_only: false,
                    ..Default::default()
                },
            ))
        };
        let (src, off, on) = (mean(&img), run(false), run(true));
        assert!(
            on < src * 0.2,
            "雜訊被當成軌跡撈回來了：原本 {src:.1}，關補回 {off:.1}、開補回 {on:.1}"
        );
    }

    /// 被紅色煙火照亮的濃煙紅通道會頂到 255，但它仍是煙，不能因為「過曝」就整片
    /// 留著（見 [`CLIP_KEEP_ABOVE`]）；騎在同一片煙上的過曝線條則照樣要留住
    #[test]
    fn red_clipped_smoke_is_still_removed() {
        let (w, h) = (160u32, 120u32);
        // 實照煙火中央量到的紅煙
        let mut img = solid(w, h, [255, 117, 74]);
        for y in 0..h {
            img.put_pixel(80, y, Rgb([255, 215, 150]));
            img.put_pixel(81, y, Rgb([255, 215, 150]));
        }
        let out = remove_smoke(
            &img,
            &SmokeParams {
                strength: 100,
                sky_only: false,
                ..Default::default()
            },
        );
        let smoke = out.get_pixel(30, 60).0;
        assert!(
            smoke.iter().all(|&v| v < 60),
            "紅通道過曝的煙沒被扣掉：{smoke:?} 應該幾乎全黑"
        );
        let trail = out.get_pixel(80, 60).0;
        assert!(
            trail[0] >= 240,
            "煙上的過曝線條被壓暗了：{trail:?} 的紅色應維持接近 255"
        );
    }

    /// 藍色的夜空（天還沒全黑、或後製調藍的照片）去煙後要還是藍的：
    /// 只扣高出夜空底色的那一截，煙散掉的地方回到夜空原本的顏色，不是壓成黑
    /// （見 [`floor_from`]；實照 B0045103 回報過藍黑分界）
    #[test]
    fn a_blue_night_sky_keeps_its_colour() {
        let (w, h) = (240u32, 160u32);
        let sky = [12u8, 30, 80];
        let mut img = solid(w, h, sky);
        // 一團白煙飄在藍天上
        for y in 60..100 {
            for x in 60..120 {
                img.put_pixel(x, y, Rgb([150, 140, 130]));
            }
        }
        // 強度 80 以上係數 k ≥ 1.28，煙被整截扣掉、停在底色上；
        // 更低的強度本來就會留一小截，不在這個測試的範圍
        for strength in [80, 100] {
            let out = remove_smoke(
                &img,
                &SmokeParams {
                    strength,
                    sky_only: false,
                    ..Default::default()
                },
            );
            let clean = out.get_pixel(20, 20).0;
            assert!(
                (0..3).all(|c| (clean[c] as i32 - sky[c] as i32).abs() <= 4),
                "強度 {strength}：乾淨的藍天被動了：{clean:?} 應維持 {sky:?}"
            );
            let gone = out.get_pixel(90, 80).0;
            assert!(
                (0..3).all(|c| (gone[c] as i32 - sky[c] as i32).abs() <= 12),
                "強度 {strength}：煙散掉的地方應回到藍天 {sky:?}，卻是 {gone:?}"
            );
        }
    }

    /// 地景沒有橫貫整張（中間是海口）時，天空不能從缺口流進水面：
    /// 水面是亮的倒影、不會被判成乾淨夜空，乾淨夜空最低那一列就是水平線，
    /// 缺口那幾欄不該比它低太多（見 [`SkyProbe::sea`]、[`REGION_SEA_MARGIN`]；
    /// 實照 B0045103、A1202723 的水面被扣出暗斑）
    #[test]
    fn a_sea_gap_does_not_let_the_sky_leak_into_the_water() {
        let (w, h) = (300u32, 200u32);
        let mut img = solid(w, h, [12, 14, 20]);
        for y in 100..h {
            for x in 0..w {
                if y < 130 && (x < 110 || x >= 190) {
                    // 兩側的地景：亮暗交錯的紋理
                    let c = if (x + y) % 2 == 0 {
                        [200, 190, 170]
                    } else {
                        [40, 36, 30]
                    };
                    img.put_pixel(x, y, Rgb(c));
                } else if x >= 110 && x < 190 || y >= 130 {
                    // 水面：平坦而亮的粉紅倒影，從水平線起慢慢亮起來（沒有硬邊可擋）
                    let t = ((y - 100) as f32 / 30.0).min(1.0);
                    let c = [
                        (12.0 + 218.0 * t) as u8,
                        (14.0 + 186.0 * t) as u8,
                        (20.0 + 190.0 * t) as u8,
                    ];
                    img.put_pixel(x, y, Rgb(c));
                }
            }
        }
        let probe = sky_probe(&img);
        assert!(
            probe.sea < 0.65,
            "水平線應在地景那一帶（約 0.5~0.6），量到 {}",
            probe.sea
        );
        let lut = srgb_lut();
        let lin: Vec<[f32; 3]> = img
            .pixels()
            .map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]])
            .collect();
        let (sw, sh) = (w as usize, h as usize);
        let wt = sky_region(&lin, sw, sh, w as f32, probe.sea).d;
        let at = |x: usize, y: usize| wt[y * sw + x];
        let column = |x: usize| -> Vec<String> {
            (0..sh).step_by(10).map(|y| format!("{:.2}", at(x, y))).collect()
        };
        assert!(at(150, 50) > 0.8, "缺口上方的天空應該是天空：{}", at(150, 50));
        assert!(at(50, 50) > 0.8, "地景上方的天空應該是天空：{}", at(50, 50));
        assert!(
            at(150, 170) < 0.2,
            "缺口下方的水面不該被當成天空：{}（第 150 欄由上而下每 10 列：{}）",
            at(150, 170),
            column(150).join(" ")
        );
        assert!(at(50, 170) < 0.2, "地景下方的水面不該被當成天空：{}", at(50, 170));
    }

    #[test]
    fn the_white_core_of_a_burst_keeps_its_brightness() {
        // 煙火的芯是「三個通道一起頂上去」的白，壓暗會整團變成灰色。
        // 這個問題復發過一次：CLIP_KEEP 只看亮度、門檻又訂在 246，
        // 芯沒頂到 255（實照多半落在 230~250）就保不住。
        //
        // 同時要顧另一邊：被照亮的濃煙一樣可以很亮，不能因為「亮」就一起保住，
        // 否則等於不去煙了。兩者差在顏色——芯是白的、煙是暖的
        let (w, h) = (160u32, 120u32);
        let mut img = solid(w, h, [20, 22, 30]);
        for y in 0..h {
            for x in 0..w {
                // 左半：又亮又暖的煙（該被扣掉）
                if x < 80 {
                    *img.get_pixel_mut(x, y) = Rgb([238, 186, 128]);
                }
            }
        }
        // 右半中間放一團亮白的芯（該原樣留著）
        for y in 40..80 {
            for x in 100..140 {
                *img.get_pixel_mut(x, y) = Rgb([240, 238, 236]);
            }
        }
        let out = remove_smoke(
            &img,
            &SmokeParams {
                strength: 100,
                sky_only: false,
                ..Default::default()
            },
        );
        let core_src = img.get_pixel(120, 60).0;
        let core_out = out.get_pixel(120, 60).0;
        let drop: i32 = (0..3)
            .map(|c| core_src[c] as i32 - core_out[c] as i32)
            .max()
            .unwrap();
        assert!(
            drop <= 12,
            "白色亮芯要保持原來的亮度，卻從 {core_src:?} 掉到 {core_out:?}"
        );

        let smoke_src = img.get_pixel(40, 60).0;
        let smoke_out = out.get_pixel(40, 60).0;
        let removed: i32 = (0..3)
            .map(|c| smoke_src[c] as i32 - smoke_out[c] as i32)
            .max()
            .unwrap();
        assert!(
            removed > 40,
            "暖色的濃煙還是要扣掉，卻只從 {smoke_src:?} 變成 {smoke_out:?}"
        );
    }

    #[test]
    fn sky_only_decides_whether_the_ground_is_touched() {
        // 「只處理天空」開著時地景一個像素都不能動；關掉才整張一視同仁。
        // 沒有這個開關，煙真的飄到地面的照片就沒辦法處理
        let (w, h) = (200u32, 160u32);
        let mut img = solid(w, h, [30, 32, 45]);
        for y in 0..h {
            for x in 0..w {
                if y >= 110 {
                    // 地景：帶點細節的亮面（估煙霧層時一樣會估出東西）
                    let n = ((x * 7 + y * 13) % 90) as u8;
                    *img.get_pixel_mut(x, y) = Rgb([120 + n, 110 + n, 100 + n]);
                }
            }
        }
        let ground = |p: &SmokeParams| {
            let out = remove_smoke(&img, p);
            let a = img.get_pixel(100, 145).0;
            let b = out.get_pixel(100, 145).0;
            (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).sum::<i32>()
        };
        let on = SmokeParams {
            strength: 100,
            sky_only: true,
            ..Default::default()
        };
        let off = SmokeParams {
            sky_only: false,
            ..on.clone()
        };
        assert_eq!(ground(&on), 0, "只處理天空開著時地景不該被動到");
        assert!(
            ground(&off) > 0,
            "關掉之後地景該跟著處理，否則這個開關等於沒作用"
        );
    }

    #[test]
    fn sky_region_keeps_the_sky_and_the_fireworks_but_drops_the_ground() {
        // 上半：平順的煙霧（要去煙）
        // 中間：一團有紋理的東西＝煙火（它周圍的煙最該清，不能被判成地景，
        //       也不能在它下方的天空拖出一道陰影）
        // 下段：橫貫整排、佈滿細節的地景與水面（一個像素都不該動）
        let (w, h) = (240u32, 200u32);
        let mut img = solid(w, h, [40, 42, 55]);
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel_mut(x, y);
                let noisy = |seed: u32| ((seed.wrapping_mul(2654435761)) >> 24) as u8;
                if y >= 150 {
                    // 地景：又亮又碎
                    let n = noisy(x * 7 + y * 13);
                    *p = Rgb([n, n / 2 + 60, n / 3 + 80]);
                } else if (100..140).contains(&x) && (60..100).contains(&y) {
                    // 煙火：細線交錯
                    let v = if (x + y) % 3 == 0 { 230 } else { 45 };
                    *p = Rgb([v, v, v]);
                }
            }
        }
        let lut = srgb_lut();
        let lin: Vec<[f32; 3]> = img
            .pixels()
            .map(|px| [lut[px[0] as usize], lut[px[1] as usize], lut[px[2] as usize]])
            .collect();
        let r = sky_region(&lin, w as usize, h as usize, w.max(h) as f32, 1.0);
        let at = |x: u32, y: u32| r.d[y as usize * w as usize + x as usize];

        assert!(at(20, 20) > 0.8, "上方的天空要去煙，卻只有 {}", at(20, 20));
        assert!(
            at(120, 80) > 0.8,
            "煙火本身要算天空（周圍的煙才清得到），卻只有 {}",
            at(120, 80)
        );
        assert!(
            at(120, 130) > 0.8,
            "煙火下方的天空不該被它擋住，卻只有 {}",
            at(120, 130)
        );
        assert!(
            at(120, 190) < 0.2,
            "地景一個像素都不該動，卻有 {}",
            at(120, 190)
        );
    }

    #[test]
    fn select_sky_keeps_the_sky_but_skips_the_fireworks_lines_and_the_ground() {
        // 與上面同一張合成圖：上半平順的天空、中間細線交錯的煙火、
        // 下段佈滿細節的地景。「天空」遮色片要圈到天空與煙火下方的天空、
        // 避開煙火的線條本身、不碰地景
        let (w, h) = (240u32, 200u32);
        let mut img = solid(w, h, [40, 42, 55]);
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel_mut(x, y);
                let noisy = |seed: u32| ((seed.wrapping_mul(2654435761)) >> 24) as u8;
                if y >= 150 {
                    let n = noisy(x * 7 + y * 13);
                    *p = Rgb([n, n / 2 + 60, n / 3 + 80]);
                } else if (100..140).contains(&x) && (60..100).contains(&y) {
                    let v = if (x + y) % 3 == 0 { 230 } else { 45 };
                    *p = Rgb([v, v, v]);
                }
            }
        }
        let o = select_sky(&img, 0, 0, 0).expect("有天空的照片要選得出來");
        let at = |x: u32, y: u32| o.at(x as f32 / w as f32, y as f32 / h as f32);
        assert!(at(20, 20) > 0.8, "上方的天空要選進來，卻只有 {}", at(20, 20));
        assert!(
            at(120, 130) > 0.7,
            "煙火下方的天空不該被它擋住，卻只有 {}",
            at(120, 130)
        );
        // 煙火那一塊：線條交錯的地方要避開
        assert!(at(120, 80) < 0.3, "煙火的紋路要避開，卻有 {}", at(120, 80));
        assert!(at(120, 190) < 0.2, "地景不該選進來，卻有 {}", at(120, 190));

        // 羽化與邊緣照物件那套調：換一組不重跑分割，鋪滿整張的範圍不變
        let soft = o.refined(50, 0);
        assert_eq!((soft.w, soft.h), (o.w, o.h));
        assert_eq!(soft.area, o.area);
    }

    #[test]
    fn the_preview_measures_the_sky_at_the_same_scale_as_the_output() {
        // 預覽是原圖縮到 1600 的縮圖，天空範圍的統計半徑要照原圖換算回去。
        //
        // 少了這一步（拿縮圖自己的尺寸去夾，也就是下面的 (2, 16)），煙火簇
        // 那一帶的紋理統計會被拉高、種子判定過不了關，那幾欄從水面一路擋到
        // 畫面上緣：預覽中央出現一根完全不去煙的「柱子」，存出來的成品卻是
        // 好的——實際回報過的狀況
        let k = 1600.0 / 10007.0;
        let (hi, area) = region_radii(1600.0, 10007.0);
        assert!(
            (hi as f32 - 6.0 * k).abs() <= 0.5,
            "細節半徑沒照原圖換算：{hi}"
        );
        assert!(
            (area as f32 - 40.0 * k).abs() <= 0.5,
            "區域半徑沒照原圖換算：{area}"
        );

        // 手上這張就是原圖時，行為與加上這條路之前**完全相同**——
        // 存檔走的是這一條，成品不該因為修好預覽而改變
        assert_eq!(region_radii(10007.0, 10007.0), (6, 40));
        assert_eq!(region_radii(1600.0, 1600.0), (2, 16));
        assert_eq!(region_radii(200.0, 200.0), (1, 4), "再小的圖也要量得到東西");
    }

    #[test]
    fn params_are_clamped_into_range() {
        let p = SmokeParams {
            strength: 500,
            detail: -20,
            feather: -5,
            tolerance: 400,
            ..Default::default()
        }
        .clamped();
        assert_eq!(
            (p.strength, p.detail, p.feather, p.tolerance),
            (100, 0, 0, 100)
        );
    }

    #[test]
    fn zero_strength_returns_the_original_untouched() {
        let img = solid(8, 6, [90, 70, 130]);
        let out = remove_smoke(
            &img,
            &SmokeParams {
                strength: 0,
                ..Default::default()
            },
        );
        assert_eq!(out, img);
    }

    #[test]
    fn output_keeps_the_input_dimensions() {
        let img = solid(37, 19, [40, 40, 60]);
        let out = remove_smoke(&img, &SmokeParams::default());
        assert_eq!((out.width(), out.height()), (37, 19));
    }

    /// 全黑的夜空不該被去煙處理弄出雜訊或抬亮
    #[test]
    fn pure_black_stays_black() {
        let img = solid(16, 16, [0, 0, 0]);
        let out = remove_smoke(&img, &SmokeParams::default());
        assert!(out.pixels().all(|p| p.0 == [0, 0, 0]));
    }

    /// 整片均勻的煙霧沒有任何細節，應該被扣到近乎全黑
    #[test]
    fn uniform_smoke_is_removed() {
        let img = solid(64, 64, [120, 100, 170]);
        let out = remove_smoke(
            &img,
            &SmokeParams {
                strength: 100,
                detail: 60,
                ..Default::default()
            },
        );
        let brightest = out
            .pixels()
            .map(|p| p.0.iter().copied().max().unwrap())
            .max();
        assert_eq!(brightest, Some(0), "均勻煙霧沒被扣乾淨");
    }

    /// 扣掉一大半亮度的像素要以原本的色相為準（見 [`FADE_KEEP`]）：
    /// 殘差只剩一小截時，逐通道相減的結果是煙霧層的估計誤差放大出來的，
    /// 實照上就是金黃的煙火線條與橘色的煙一起翻成橄欖綠
    #[test]
    fn heavily_faded_pixels_keep_their_hue() {
        // 暖煙上畫幾條只比煙亮一點點的細線（夠細，估煙霧層的開運算會把它們掃掉）
        let mut img = solid(64, 64, [200, 140, 80]);
        for y in 0..64 {
            for x in (6..64).step_by(12) {
                img.put_pixel(x, y, Rgb([205, 160, 140]));
                img.put_pixel(x + 1, y, Rgb([205, 160, 140]));
            }
        }
        // 強度拉滿會把這種像素直接扣成全黑，看不出色相
        let out = remove_smoke(
            &img,
            &SmokeParams {
                strength: 55,
                ..Default::default()
            },
        );
        let p = out.get_pixel(6, 32).0;
        assert!(
            p[0] > p[1] && p[1] > p[2],
            "線條扣完變色了：{p:?} 應維持原本 [205, 160, 140] 的紅 > 綠 > 藍"
        );
    }

    /// 暖色的亮芯也不能被壓暗（見 [`CORE_KEEP_AREA`]）：金柳、橘紅牡丹的芯
    /// 過不了「夠白」那一關，只看單點就會被當成煙扣掉，整團跟著發灰
    #[test]
    fn warm_cores_keep_their_brightness() {
        // 暖煙上放一團夠大的暖色亮芯（不夠白，過不了 CORE_KEEP_NEUTRAL）
        let mut img = solid(128, 128, [190, 150, 110]);
        for y in 44..84 {
            for x in 44..84 {
                img.put_pixel(x, y, Rgb([250, 225, 170]));
            }
        }
        let out = remove_smoke(&img, &SmokeParams::default());
        let p = out.get_pixel(64, 64).0;
        assert!(
            p[0] >= 245 && p[1] >= 220,
            "暖色亮芯被壓暗了：{p:?} 應維持接近原本的 [250, 225, 170]"
        );
        // 同一張裡的煙照樣要被扣掉，不能因為旁邊有芯就整片留著
        let smoke = out.get_pixel(4, 4).0;
        assert!(
            smoke.iter().all(|&v| v < 60),
            "亮芯的保護漫到煙上了：角落的煙 {smoke:?} 應該幾乎被扣光"
        );
    }

    /// 亮到沒有細節的煙火芯不能被壓暗：濃煙底下的白芯扣完仍要是白的，
    /// 壓暗只會讓它變成一團灰（煙越濃、ys 越大，不修的話掉得越多）
    #[test]
    fn clipped_highlights_keep_their_brightness() {
        // 很濃的暖煙裡放一顆過曝的白芯
        let mut img = solid(64, 64, [190, 170, 150]);
        for y in 28..36 {
            for x in 28..36 {
                img.put_pixel(x, y, Rgb([255, 252, 246]));
            }
        }
        let out = remove_smoke(&img, &SmokeParams::default());
        let p = out.get_pixel(32, 32).0;
        assert!(
            p[0] >= 250 && p[1] >= 245,
            "過曝的煙火芯被壓暗了：{p:?} 應維持接近原本的 [255, 252, 246]"
        );
    }

    /// 過曝的煙火亮點在扣掉煙霧後仍須維持原本的色相（不能由橘紅翻成青綠）
    #[test]
    fn clipped_highlights_keep_their_hue() {
        // 紫色煙霧背景中放一顆過曝的橘紅亮球
        let mut img = solid(64, 64, [120, 100, 170]);
        for y in 30..34 {
            for x in 30..34 {
                img.put_pixel(x, y, Rgb([255, 160, 60]));
            }
        }
        let out = remove_smoke(&img, &SmokeParams::default());
        let p = out.get_pixel(32, 32).0;
        assert!(
            p[0] > p[1] && p[1] >= p[2],
            "亮球色相反轉了：{p:?} 應維持 R > G >= B"
        );
    }

    /// 沒有過曝的煙火線條也不能被改色：煙霧偏暖時逐通道相減會把紅色扣掉最多，
    /// 金黃色的線條就一路偏成橄欖綠
    #[test]
    fn unclipped_streaks_keep_their_hue() {
        // 暖色煙霧背景中拉一條金黃色的細線（最亮通道 210，離過曝門檻 235 還有一段）
        let mut img = solid(64, 64, [110, 80, 60]);
        for y in 0..64 {
            img.put_pixel(31, y, Rgb([210, 170, 70]));
            img.put_pixel(32, y, Rgb([210, 170, 70]));
        }
        let out = remove_smoke(&img, &SmokeParams::default());
        let p = out.get_pixel(31, 32).0;
        // 以原始比例為準：扣完之後 R/G 不該掉太多（掉太多就是偏綠了）
        let before = 210.0 / 170.0;
        let after = p[0] as f32 / p[1].max(1) as f32;
        assert!(
            after > before * 0.98,
            "線條偏綠了：{p:?} R/G={after:.3} 應接近原本的 {before:.3}"
        );
    }

    /// 極小尺寸不能讓濾波器的視窗計算越界
    #[test]
    fn tiny_images_do_not_panic() {
        for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2)] {
            let img = solid(w, h, [100, 90, 140]);
            let out = remove_smoke(&img, &SmokeParams::default());
            assert_eq!((out.width(), out.height()), (w, h));
        }
    }

    /// 框選範圍外的像素必須原封不動
    #[test]
    fn region_leaves_the_outside_untouched() {
        let img = solid(80, 80, [120, 100, 170]);
        let p = SmokeParams {
            strength: 100,
            feather: 0,
            shapes: vec![Shape::Rect(Region {
                x0: 0.5,
                y0: 0.0,
                x1: 1.0,
                y1: 1.0,
            })],
            ..Default::default()
        };
        let out = remove_smoke(&img, &p);
        // 最左側完全在框外，右側在框內且已被扣掉
        assert_eq!(out.get_pixel(2, 40).0, [120, 100, 170], "框外被動到了");
        assert!(out.get_pixel(78, 40).0[2] < 170, "框內沒有去煙");
    }

    /// 從右下往左上拉出來的框也要正規化成同一塊
    #[test]
    fn region_is_normalized_whichever_way_it_is_dragged() {
        let forward = Region {
            x0: 0.2,
            y0: 0.3,
            x1: 0.8,
            y1: 0.9,
        };
        let backward = Region {
            x0: 0.8,
            y0: 0.9,
            x1: 0.2,
            y1: 0.3,
        };
        assert_eq!(forward.normalized(), backward.normalized());
    }

    /// 退化成一條線的框視為沒框選（整張處理），不是完全不處理
    #[test]
    fn degenerate_region_falls_back_to_whole_image() {
        let p = SmokeParams {
            shapes: vec![Shape::Rect(Region {
                x0: 0.5,
                y0: 0.2,
                x1: 0.5005,
                y1: 0.9,
            })],
            ..Default::default()
        }
        .clamped();
        assert!(p.shapes.is_empty());
    }

    /// 遮色片的輔助工具：直接看權重，不必繞過整套去煙。
    /// 濃度固定 100（只看形狀本身的幾何），要驗濃度用 [`mask_at_d`]
    fn mask_at(shapes: Vec<Shape>, feather: i32, w: usize, h: usize, x: usize, y: usize) -> f32 {
        mask_at_d(shapes, feather, 100, w, h, x, y)
    }

    fn mask_at_d(
        shapes: Vec<Shape>,
        feather: i32,
        density: i32,
        w: usize,
        h: usize,
        x: usize,
        y: usize,
    ) -> f32 {
        ShapeMask::new(&shapes, feather, density, w, h).at(x, y)
    }

    /// 一筆的筆跡（橫著刷過畫面中央）
    fn stroke() -> Shape {
        Shape::Brush(Brush {
            pts: vec![[0.2, 0.5], [0.8, 0.5]],
            radius: 0.05,
        })
    }

    /// 線性漸層：起點端全效果、終點端歸零，中間單調遞減
    #[test]
    fn linear_gradient_fades_from_start_to_end() {
        // 由上往下拖：上緣全去煙，下緣完全不動
        let s = vec![Shape::Linear(Linear {
            x0: 0.5,
            y0: 0.2,
            x1: 0.5,
            y1: 0.8,
        })];
        let at = |y: usize| mask_at(s.clone(), 0, 100, 100, 50, y);
        assert!(at(5) > 0.99, "起點之前應是全效果：{}", at(5));
        assert!(at(95) < 0.01, "終點之後應完全不動：{}", at(95));
        assert!((at(50) - 0.5).abs() < 0.05, "中點應約為一半：{}", at(50));
        assert!(at(30) > at(50) && at(50) > at(70), "沒有單調遞減");
        // 與拖曳方向垂直的兩側無限延伸：同一高度不論左右都一樣
        assert!((at(50) - mask_at(s, 0, 100, 100, 5, 50)).abs() < 1e-4);
    }

    /// 放射性漸層：橢圓內去煙、外面不動；反轉之後剛好對調
    #[test]
    fn radial_gradient_covers_the_ellipse() {
        let r = Radial {
            cx: 0.5,
            cy: 0.5,
            rx: 0.3,
            ry: 0.2,
            invert: false,
        };
        let inside = mask_at(vec![Shape::Radial(r)], 0, 100, 100, 50, 50);
        let outside = mask_at(vec![Shape::Radial(r)], 0, 100, 100, 95, 50);
        assert!(inside > 0.99, "橢圓內沒有去煙：{inside}");
        assert!(outside < 0.01, "橢圓外被動到了：{outside}");
        // 短軸只有 0.2：y 方向 30 像素外就該出界，x 方向同樣距離還在裡面
        assert!(mask_at(vec![Shape::Radial(r)], 0, 100, 100, 50, 15) < 0.01);
        assert!(mask_at(vec![Shape::Radial(r)], 0, 100, 100, 25, 50) > 0.5);
        let inv = Radial { invert: true, ..r };
        assert!(mask_at(vec![Shape::Radial(inv)], 0, 100, 100, 50, 50) < 0.01);
        assert!(mask_at(vec![Shape::Radial(inv)], 0, 100, 100, 95, 50) > 0.99);
    }

    /// 筆刷：刷過的地方去煙，沒刷到的不動；粗細由半徑決定
    #[test]
    fn brush_covers_what_it_paints() {
        let b = vec![stroke()];
        assert!(
            mask_at(b.clone(), 0, 100, 100, 50, 50) > 0.99,
            "筆跡上沒去煙"
        );
        assert!(
            mask_at(b.clone(), 0, 100, 100, 50, 90) < 0.01,
            "刷不到的地方被動到了"
        );
        // 半徑 0.05 × 長邊 100＝5 像素；折線兩端之外也一樣收在半徑內
        assert!(mask_at(b.clone(), 0, 100, 100, 50, 56) < 0.01);
        assert!(mask_at(b, 0, 100, 100, 5, 50) < 0.01);
    }

    /// 濃度只管筆刷：筆跡上就是那幾成，沒畫任何形狀時也不作用
    #[test]
    fn density_only_applies_to_the_brush() {
        let b = vec![stroke()];
        let w = mask_at_d(b.clone(), 0, 80, 100, 100, 50, 50);
        assert!((w - 0.8).abs() < 0.01, "筆跡上應該剛好八成：{w}");
        assert!(mask_at_d(b, 0, 80, 100, 100, 50, 90) < 0.01, "沒刷到的被動到了");
        // 一個形狀都沒畫＝整張都要處理，這時濃度不該把整張打八折
        assert!(mask_at_d(Vec::new(), 0, 80, 100, 100, 50, 50) > 0.99);
        // 框、漸層、物件一律 100%，濃度拉到多低都一樣
        let rect = vec![Shape::Rect(Region {
            x0: 0.1,
            y0: 0.1,
            x1: 0.9,
            y1: 0.9,
        })];
        assert!(
            mask_at_d(rect.clone(), 0, 20, 100, 100, 50, 50) > 0.99,
            "框被濃度打了折"
        );
        let lin = vec![Shape::Linear(Linear {
            x0: 0.5,
            y0: 0.2,
            x1: 0.5,
            y1: 0.8,
        })];
        assert!(mask_at_d(lin, 0, 20, 100, 100, 50, 5) > 0.99, "漸層被濃度打了折");
    }

    /// 筆刷一筆加一次：同一塊刷兩筆就滿了（濃度 80 時 0.8＋0.8 夾回 1）。
    /// 同一筆自己交疊則不會變濃
    #[test]
    fn brush_strokes_stack_but_one_stroke_does_not() {
        let one = vec![stroke()];
        let two = vec![stroke(), stroke()];
        assert!((mask_at_d(one, 0, 80, 100, 100, 50, 50) - 0.8).abs() < 0.01);
        assert!(mask_at_d(two, 0, 80, 100, 100, 50, 50) > 0.99, "刷兩筆沒疊上去");
        // 同一筆繞回來重疊自己：仍是一筆的濃度
        let looped = vec![Shape::Brush(Brush {
            pts: vec![[0.2, 0.5], [0.8, 0.5], [0.2, 0.5]],
            radius: 0.05,
        })];
        let w = mask_at_d(looped, 0, 80, 100, 100, 50, 50);
        assert!((w - 0.8).abs() < 0.01, "同一筆自己交疊變濃了：{w}");
    }

    /// 框選／漸層／物件也會疊加：每一個都算數，畫幾個就是它們的聯集
    #[test]
    fn flat_shapes_stack_at_full_density() {
        let left = Shape::Rect(Region {
            x0: 0.0,
            y0: 0.0,
            x1: 0.3,
            y1: 1.0,
        });
        let right = Shape::Rect(Region {
            x0: 0.7,
            y0: 0.0,
            x1: 1.0,
            y1: 1.0,
        });
        // 濃度拉到 20 也一樣：這幾種不吃濃度
        let s = vec![left, right];
        assert!(mask_at_d(s.clone(), 0, 20, 100, 100, 10, 50) > 0.99, "左邊那塊沒生效");
        assert!(mask_at_d(s.clone(), 0, 20, 100, 100, 90, 50) > 0.99, "右邊那塊沒生效");
        assert!(mask_at_d(s, 0, 20, 100, 100, 50, 50) < 0.01, "中間不該被蓋到");
    }

    /// 框選與筆刷混用：框是 100%，筆刷再照濃度加上去
    #[test]
    fn brush_adds_on_top_of_a_box() {
        let rect = Shape::Rect(Region {
            x0: 0.0,
            y0: 0.0,
            x1: 0.3,
            y1: 1.0,
        });
        let s = vec![rect, stroke()];
        // 框內沒刷到的地方：框本來就是 100%
        assert!(mask_at_d(s.clone(), 0, 80, 100, 100, 10, 10) > 0.99);
        // 框外但刷到的地方：筆刷照濃度
        assert!((mask_at_d(s.clone(), 0, 80, 100, 100, 70, 50) - 0.8).abs() < 0.01);
        // 兩者都蓋到：早就滿了
        assert!(mask_at_d(s, 0, 80, 100, 100, 25, 50) > 0.99);
    }

    /// 命中保護色的像素不該被去煙
    #[test]
    fn protected_colour_survives() {
        // 紫煙背景中放一塊要保住的青色
        let keep = [40, 200, 210];
        let mut img = solid(80, 80, [120, 100, 170]);
        for y in 20..60 {
            for x in 20..60 {
                img.put_pixel(x, y, Rgb(keep));
            }
        }
        let mut p = SmokeParams {
            strength: 100,
            tolerance: 20,
            ..Default::default()
        };
        assert!(p.add_protect(keep));
        let out = remove_smoke(&img, &p);
        assert_eq!(out.get_pixel(40, 40).0, keep, "保護色被扣掉了");
        // 保護色以外的煙霧照樣要被扣掉
        assert!(out.get_pixel(2, 2).0[2] < 170, "保護色以外沒有去煙");
    }

    /// 指定多個保護色時，每一個都要生效
    #[test]
    fn every_protected_colour_survives() {
        let keeps = [[40, 200, 210], [230, 90, 40], [90, 240, 110]];
        let mut img = solid(160, 60, [120, 100, 170]);
        // 三塊各自塗上一個要保住的顏色
        for (k, c) in keeps.iter().enumerate() {
            for y in 20..40 {
                for x in (k as u32 * 50 + 10)..(k as u32 * 50 + 40) {
                    img.put_pixel(x, y, Rgb(*c));
                }
            }
        }
        let mut p = SmokeParams {
            strength: 100,
            tolerance: 20,
            ..Default::default()
        };
        for c in &keeps {
            assert!(p.add_protect(*c));
        }
        let out = remove_smoke(&img, &p);
        for (k, c) in keeps.iter().enumerate() {
            assert_eq!(
                out.get_pixel(k as u32 * 50 + 25, 30).0,
                *c,
                "第 {k} 個保護色被扣掉了"
            );
        }
        assert!(out.get_pixel(2, 2).0[2] < 170, "保護色以外沒有去煙");
    }

    /// 保護色有數量上限，重複的顏色不重複佔位
    #[test]
    fn protect_list_rejects_duplicates_and_overflow() {
        let mut p = SmokeParams::default();
        assert!(p.add_protect([10, 20, 30]));
        assert!(!p.add_protect([10, 20, 30]), "同色不該重複加入");
        for i in 1..MAX_PROTECT {
            assert!(p.add_protect([i as u8, 0, 0]));
        }
        assert!(!p.add_protect([9, 9, 9]), "滿了還能再加");
        assert_eq!(p.protect_colors().count(), MAX_PROTECT);
        p.remove_protect(0);
        assert_eq!(p.protect_colors().count(), MAX_PROTECT - 1);
        assert!(p.add_protect([9, 9, 9]), "移除後空出來的位置沒被用到");
        p.clear_protect();
        assert!(!p.has_protect());
    }

    /// 容差 0 時只有幾乎完全同色才受保護，不會整張都不處理
    #[test]
    fn zero_tolerance_barely_protects_anything() {
        let img = solid(64, 64, [120, 100, 170]);
        let mut p = SmokeParams {
            strength: 100,
            tolerance: 0,
            ..Default::default()
        };
        p.add_protect([40, 200, 210]);
        let out = remove_smoke(&img, &p);
        let brightest = out
            .pixels()
            .map(|p| p.0.iter().copied().max().unwrap())
            .max();
        assert_eq!(brightest, Some(0), "與保護色差很遠卻沒被去煙");
    }

    /// 只清雲、不去煙時也要生效（強度 0 不能被當成「什麼都不做」）
    #[test]
    fn sky_clean_works_without_dehazing() {
        // 上半部是均勻的暗雲、下半部純黑，中間沒有東西隔開
        let mut img = solid(80, 80, [0, 0, 0]);
        for y in 0..40 {
            for x in 0..80 {
                img.put_pixel(x, y, Rgb([40, 44, 60]));
            }
        }
        let p = SmokeParams {
            strength: 0,
            sky_clean: 100,
            sky_range: 60,
            ..Default::default()
        };
        assert!(!p.is_neutral(), "只清雲時被當成什麼都不做");
        let out = remove_smoke(&img, &p);
        let before = img.get_pixel(40, 10).0[2];
        let after = out.get_pixel(40, 10).0[2];
        assert!(after < before, "雲沒有被壓暗（{before} → {after}）");
    }

    /// 什麼都沒開時要原圖輸出
    #[test]
    fn nothing_enabled_returns_the_original() {
        let img = solid(32, 32, [90, 80, 130]);
        let p = SmokeParams {
            strength: 0,
            sky_clean: 0,
            sky_color: None,
            ..Default::default()
        };
        assert!(p.is_neutral());
        assert_eq!(remove_smoke(&img, &p), img);
    }

    /// 指定夜空色時，純黑的天空要被染上顏色（只換色度對黑色無效）
    #[test]
    fn sky_tint_colours_a_black_sky() {
        let img = solid(80, 80, [0, 0, 0]);
        let p = SmokeParams {
            strength: 0,
            sky_clean: 0,
            sky_range: 60,
            sky_color: Some([30, 60, 140]),
            sky_tint: 100,
            ..Default::default()
        };
        let out = remove_smoke(&img, &p);
        let px = out.get_pixel(40, 40).0;
        assert!(px[2] > px[0], "夜空沒有被染成偏藍：{px:?}");
        assert!(px[2] > 10, "夜空幾乎沒有上到色：{px:?}");
    }

    /// 「雲朵」是天空裡**沒有煙火紋路**的東西：被燈火或煙火照亮的煙霧一樣算，
    /// 不是只有暗面。亮到什麼程度還算，由「範圍」決定
    #[test]
    fn cloud_clean_reaches_lit_smoke() {
        // 上半是一層被照亮、平坦無紋路的煙霧，下半是純黑的夜空
        let smoke = [150, 150, 158];
        let mut img = solid(120, 120, [0, 0, 0]);
        for y in 0..60 {
            for x in 0..120 {
                img.put_pixel(x, y, Rgb(smoke));
            }
        }
        let at = |p: &SmokeParams| remove_smoke(&img, p).get_pixel(60, 20).0[1];
        // 範圍小＝只當暗面是天空，這麼亮的煙霧碰不到
        let narrow = SmokeParams {
            strength: 0,
            sky_clean: 100,
            sky_range: 20,
            ..Default::default()
        };
        let before = at(&narrow);
        assert!(before >= 145, "範圍很小卻壓到了亮煙霧（{before}）");
        // 範圍放寬到蓋得住它的亮度，就該被壓回夜色
        let wide = SmokeParams {
            sky_range: 90,
            ..narrow.clone()
        };
        let after = at(&wide);
        assert!(
            after < 60,
            "沒有煙火紋路的亮煙霧沒被當成雲朵（{before} → {after}）"
        );
    }

    /// 亮到判定不出來的雲，用吸管指名之後要被壓回夜色
    #[test]
    fn picked_cloud_colour_gets_pushed_back_to_night() {
        // 上半部是被城市燈光照亮的亮灰色雲（亮度遠超過天空判定的門檻），下半部純黑
        let cloud = [150, 150, 158];
        let mut img = solid(80, 80, [0, 0, 0]);
        for y in 0..40 {
            for x in 0..80 {
                img.put_pixel(x, y, Rgb(cloud));
            }
        }
        let base = SmokeParams {
            strength: 0,
            sky_clean: 100,
            sky_range: 40,
            ..Default::default()
        };
        let kept = remove_smoke(&img, &base).get_pixel(40, 10).0[1];
        assert!(kept >= 145, "這麼亮的雲本來就不該被亮度判定抓到（{kept}）");

        let mut p = base.clone();
        assert!(p.add_cloud(cloud));
        let after = remove_smoke(&img, &p).get_pixel(40, 10).0[1];
        assert!(after < 40, "吸了雲色卻沒被壓回夜色（{kept} → {after}）");
    }

    /// 吸雲色不能波及地面上同色的東西（天空仍要從畫面上緣連得過來）
    #[test]
    fn picked_cloud_colour_stops_at_the_skyline() {
        let cloud = [150, 150, 158];
        let mut img = solid(80, 80, [0, 0, 0]);
        for y in 0..80 {
            for x in 0..80 {
                let c = match y {
                    // 天空的雲、乾淨夜空、岸邊燈火那一排、以及同色的地面
                    0..=19 => cloud,
                    20..=49 => [0, 0, 0],
                    50..=54 => [255, 180, 60],
                    _ => cloud,
                };
                img.put_pixel(x, y, Rgb(c));
            }
        }
        let mut p = SmokeParams {
            strength: 0,
            sky_clean: 100,
            sky_range: 40,
            ..Default::default()
        };
        p.add_cloud(cloud);
        let out = remove_smoke(&img, &p);
        assert!(out.get_pixel(40, 8).0[1] < 40, "天上的雲沒被壓掉");
        assert!(
            out.get_pixel(40, 70).0[1] >= 145,
            "地面被當成雲壓掉了：{:?}",
            out.get_pixel(40, 70).0
        );
    }

    /// 值噪聲：格點雜訊做雙線性內插，做出有起伏的煙霧紋理
    fn value_noise(x: f32, y: f32, cell: f32, seed: u32) -> f32 {
        fn hash(ix: i32, iy: i32, seed: u32) -> f32 {
            let mut h = (ix as u32)
                .wrapping_mul(0x8da6_b343)
                .wrapping_add((iy as u32).wrapping_mul(0xd816_3841))
                .wrapping_add(seed.wrapping_mul(0xcbbc_9dfb));
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1_e995);
            h ^= h >> 15;
            (h & 0xffff) as f32 / 65535.0
        }
        let (gx, gy) = (x / cell, y / cell);
        let (ix, iy) = (gx.floor() as i32, gy.floor() as i32);
        let (fx, fy) = (gx - ix as f32, gy - iy as f32);
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let a = hash(ix, iy, seed);
        let b = hash(ix + 1, iy, seed);
        let c = hash(ix, iy + 1, seed);
        let d = hash(ix + 1, iy + 1, seed);
        let top = a + (b - a) * sx;
        let bot = c + (d - c) * sx;
        top + (bot - top) * sy
    }

    /// 模擬的夜間煙火照，以及兩組要分開看的取樣點
    struct Scene {
        img: RgbImage,
        /// 空曠的煙霧面（不含線條、地景、煙火簇）：這裡要扣得乾淨
        smoke: Vec<(u32, u32)>,
        /// 密集煙火簇裡、線條之間的縫隙。那是被煙火照亮的煙，
        /// 扣掉是對的——所以只拿來看，不當成必須留下的東西
        burst: Vec<(u32, u32)>,
        /// 煙火線條本身：這些點的亮度扣完必須留著
        trails: Vec<(u32, u32)>,
        /// 沒有煙也沒有煙火的乾淨夜空
        sky: Vec<(u32, u32)>,
    }

    /// 畫一朵煙火：從中心散出 `n` 條細線
    fn draw_burst(img: &mut RgbImage, c: (f32, f32), n: usize, len: f32, seed: u32) {
        let (wf, hf) = (img.width() as f32, img.height() as f32);
        for k in 0..n {
            let ang = k as f32 / n as f32 * std::f32::consts::TAU + 0.21;
            let l = len * (0.7 + 0.6 * value_noise(k as f32 * 13.0, 0.0, 3.0, seed));
            for s in 0..(l as usize * 2) {
                let t = s as f32 / (l as usize * 2) as f32;
                let (px, py) = (c.0 + ang.cos() * l * t, c.1 + ang.sin() * l * t);
                if px < 0.0 || py < 0.0 || px >= wf || py >= hf {
                    break;
                }
                // 尾端漸暗
                let f = (1.0 - t * 0.75).clamp(0.0, 1.0);
                let cur = img.get_pixel(px as u32, py as u32).0;
                let mut out = [0u8; 3];
                for (i, v) in [255.0f32, 205.0, 110.0].iter().enumerate() {
                    out[i] = (cur[i] as f32).max(v * f) as u8;
                }
                img.put_pixel(px as u32, py as u32, Rgb(out));
            }
        }
    }

    /// 暗夜空、有紋理的煙霧團、一疏一密兩朵煙火、底部地景
    fn smoky_fireworks(w: u32, h: u32) -> Scene {
        let (wf, hf) = (w as f32, h as f32);
        let mut img = RgbImage::new(w, h);
        // 密集的那朵：整叢罩在一片被它照亮的濃煙上，就是實照裡被壓黑的情形
        let dense = (wf * 0.78, hf * 0.45);
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f32, y as f32);
                // 煙霧團：橢圓形衰減，再乘上三個尺度的紋理
                let dx = (xf - wf * 0.42) / (wf * 0.42);
                let dy = (yf - hf * 0.40) / (hf * 0.32);
                let fall = (1.0 - (dx * dx + dy * dy)).max(0.0);
                let tex = 0.55 * value_noise(xf, yf, wf / 7.0, 1)
                    + 0.35 * value_noise(xf, yf, wf / 18.0, 2)
                    + 0.10 * value_noise(xf, yf, wf / 45.0, 3);
                // 密集煙火那一帶另外罩一團被照亮的濃煙
                let dr = ((xf - dense.0).powi(2) + (yf - dense.1).powi(2)).sqrt() / (wf * 0.16);
                let lit = (1.0 - dr * dr).max(0.0) * 0.8;
                let a = (fall * fall * tex * 1.6 + lit).clamp(0.0, 1.0);
                // 暖灰的煙 + 感光元件雜訊
                let n = value_noise(xf, yf, 1.0, 7) * 4.0 - 2.0;
                let mut px = [0u8; 3];
                for (c, base) in [3.0f32, 4.0, 7.0].iter().enumerate() {
                    let smoke = [132.0f32, 116.0, 104.0][c] * a;
                    px[c] = (base + smoke + n).clamp(0.0, 255.0) as u8;
                }
                img.put_pixel(x, y, Rgb(px));
            }
        }
        let sparse = (wf * 0.32, hf * 0.38);
        draw_burst(&mut img, sparse, 48, wf * 0.26, 5);
        draw_burst(&mut img, dense, 200, wf * 0.14, 9);
        // 底部地景：暗的岸邊加幾盞燈
        let ground = (hf * 0.88) as u32;
        for y in ground..h {
            for x in 0..w {
                let c = if x % 37 == 5 && y < ground + 6 {
                    [255, 190, 90]
                } else {
                    [11, 10, 13]
                };
                img.put_pixel(x, y, Rgb(c));
            }
        }
        // 取樣點：都要避開線條本身，再依離密集煙火多遠分成兩組。
        // 空曠處讓開 3 px（線條旁邊本來就會留一點自身的輝光，不該算成殘留），
        // 簇裡的縫隙只有幾像素寬，只能讓開 1 px
        let (mut smoke, mut burst, mut sky, mut trails) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let lit = |x: u32, y: u32, r: u32| {
            (y.saturating_sub(r)..=(y + r).min(h - 1)).any(|yy| {
                (x.saturating_sub(r)..=(x + r).min(w - 1))
                    .any(|xx| img.get_pixel(xx, yy).0[0] > 140)
            })
        };
        let dist = |x: u32, y: u32, c: (f32, f32)| {
            ((x as f32 - c.0).powi(2) + (y as f32 - c.1).powi(2)).sqrt()
        };
        for y in (0..ground).step_by(3) {
            for x in (0..w).step_by(3) {
                let v = img.get_pixel(x, y).0[0];
                let d = dist(x, y, dense);
                if v <= 12 && !lit(x, y, 6) {
                    sky.push((x, y));
                } else if v <= 45 {
                    continue;
                } else if d < wf * 0.09 {
                    if v > 140 {
                        // 線條本身：這些點的亮度扣完必須留著
                        trails.push((x, y));
                    } else if !lit(x, y, 1) {
                        burst.push((x, y));
                    }
                } else if d > wf * 0.22
                    // 疏的那朵中心附近線條也很密，同樣算煙火簇，不能拿來量殘留
                    && dist(x, y, sparse) > wf * 0.13
                    && !lit(x, y, 3)
                {
                    smoke.push((x, y));
                }
            }
        }
        Scene {
            img,
            smoke,
            burst,
            trails,
            sky,
        }
    }

    /// 量測工具：取樣點上的平均最大通道值
    fn mean_level(img: &RgbImage, pts: &[(u32, u32)]) -> f32 {
        let s: f32 = pts
            .iter()
            .map(|&(x, y)| img.get_pixel(x, y).0.iter().copied().max().unwrap() as f32)
            .sum();
        s / pts.len().max(1) as f32
    }

    /// 有紋理的煙霧團要被扣得夠乾淨——不能只扣掉它的最暗底，
    /// 把紋理的起伏留成一片斑駁的薄霧（只剝一層時 p90 會留到 36）
    #[test]
    fn textured_smoke_is_cleared() {
        let s = smoky_fireworks(2400, 1700);
        let before = mean_level(&s.img, &s.smoke);
        assert!(before > 60.0, "合成場景的煙霧太淡，量不出殘留");
        // 煙扣到夜空底色為止（這個場景的底色是 (3, 4, 7) 加 ±2 的雜訊），
        // 不再是全黑，所以強度 100 的上限留到 10
        for (strength, limit) in [(80, 20u8), (100, 10)] {
            let out = remove_smoke(
                &s.img,
                &SmokeParams {
                    strength,
                    ..Default::default()
                },
            );
            // 用 p90 而非平均：殘留是一塊一塊的，平均會被大片乾淨的夜空稀釋
            let mut v: Vec<u8> = s
                .smoke
                .iter()
                .map(|&(x, y)| out.get_pixel(x, y).0.iter().copied().max().unwrap())
                .collect();
            v.sort_unstable();
            let p90 = v[v.len() * 9 / 10];
            println!(
                "strength={strength} 煙霧 {before:.1} → 平均 {:.1} / p90 {p90}",
                mean_level(&out, &s.smoke)
            );
            assert!(p90 <= limit, "strength={strength} 還有殘留煙霧：p90={p90}");
        }
    }

    /// 線條佔掉多少面積的判據要分得開：煙火簇高、空曠的煙霧面與乾淨夜空低
    #[test]
    fn streak_fill_separates_bursts_from_smoke() {
        let s = smoky_fireworks(2400, 1700);
        let (fw, fh) = (s.img.width() as usize, s.img.height() as usize);
        // 走與 estimate_smoke 相同的路：判據要在真正用得到的工作解析度上分得開。
        // 判據若跟著工作解析度走，大圖小圖會有一邊的格子細到量不出東西
        let (ww, wh) = work_size(fw, fh, WORK_LONG_EDGE);
        let b = downsample(&s.img, &srgb_lut(), ww, wh);
        let gate = streak_gate(&b, fw, ww, wh);
        let at = |p: &Plane, pts: &[(u32, u32)]| {
            let sum: f32 = pts
                .iter()
                .map(|&(x, y)| {
                    let (bx, by) = (x as usize * ww / fw, y as usize * wh / fh);
                    p.d[by.min(wh - 1) * ww + bx.min(ww - 1)]
                })
                .sum();
            sum / pts.len() as f32
        };
        let (burst, smoke, sky) = (at(&gate, &s.burst), at(&gate, &s.smoke), at(&gate, &s.sky));
        println!("少扣多少：煙火簇 {burst:.3} / 空曠煙霧 {smoke:.3} / 乾淨夜空 {sky:.3}");
        assert!(burst > 0.15, "煙火簇沒被認出來：{burst:.3}");
        assert!(
            burst > smoke * 4.0,
            "煙火簇與煙霧面分不開：{burst:.3} vs {smoke:.3}"
        );
        // 夜空的亮暗差全是雜訊，相對量很大，最容易被誤判
        assert!(sky < 0.02, "乾淨的夜空被雜訊誤判成有線條：{sky:.3}");
    }

    /// 密集煙火簇裡、線條之間的縫隙是煙火自己的光，不能被當成煙霧整片扣掉。
    /// 門檻只守住「沒有被壓成全黑」（不設判據時只剩 2%）——
    /// 少扣多少由 [`STREAK_GAIN`] 決定，那要拿真實照片校，合成場景校不準
    #[test]
    fn dense_burst_keeps_its_trails() {
        let s = smoky_fireworks(2400, 1700);
        assert!(s.trails.len() > 200, "線條的取樣點太少：{}", s.trails.len());
        let before = mean_level(&s.img, &s.trails);
        let out = remove_smoke(&s.img, &SmokeParams::default());
        let after = mean_level(&out, &s.trails);
        // 線條之間那層是被煙火照亮的煙，扣掉是對的；線條本身連帶少掉的
        // 也就是它背後那層煙的亮度（保色相那條路徑照 k·ys 等比例壓暗），
        // 所以要求的不是「線條一點都不能掉」，而是掉完仍遠比煙亮
        let (gap_before, gap_after) = (
            mean_level(&s.img, &s.burst),
            mean_level(&out, &s.burst).max(1.0),
        );
        println!(
            "煙火線條 {before:.1} → {after:.1}（留 {:.0}%）、\
             線條之間的煙 {gap_before:.1} → {gap_after:.1}",
            after / before * 100.0,
        );
        assert!(
            after > before * 0.3,
            "煙火線條被扣暗了：{before:.1} → {after:.1}"
        );
        assert!(
            after / gap_after > before / gap_before * 2.0,
            "扣完之後線條沒有比煙更突出：{:.1} 倍 → {:.1} 倍",
            before / gap_before,
            after / gap_after,
        );
    }

    /// 在天空加一片沒有紋理的亮雲。它平坦到煙霧層整團估得出來，
    /// 卻亮到扣完仍剩一截——正是清雲存在的理由。
    /// 刻意不畫到最上緣：天空判定要從畫面上緣往下傳播，頂到邊會影響的是別的事
    fn add_cloud(img: &mut RgbImage, level: f32) {
        let (w, h) = (img.width() as f32, img.height() as f32);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let dx = (x as f32 - w * 0.5) / (w * 0.55);
                let dy = (y as f32 - h * 0.18) / (h * 0.12);
                let a = (1.0 - (dx * dx + dy * dy)).max(0.0).sqrt()
                    * (0.45
                        + 0.55
                            * value_noise(x as f32, y as f32, w / 22.0, 21)
                            * value_noise(x as f32, y as f32, w / 60.0, 22));
                if a <= 0.0 {
                    continue;
                }
                let px = img.get_pixel_mut(x, y);
                for c in 0..3 {
                    px[c] = (px[c] as f32 + level * a).min(255.0) as u8;
                }
            }
        }
    }

    /// 自動判出來的參數，實際套下去就該把煙霧面扣乾淨、
    /// 又不把煙火簇壓成黑洞——與手動調出一組堪用的值是同一個標準
    #[test]
    fn auto_params_clear_the_smoke_without_crushing_the_burst() {
        let s = smoky_fireworks(2400, 1700);
        let auto = auto_params(&s.img, s.img.width());
        let mut p = SmokeParams::default();
        auto.apply_to(&mut p);
        println!("自動判參數 {auto:?}");
        let before = mean_level(&s.img, &s.smoke);
        let out = remove_smoke(&s.img, &p);
        let after = mean_level(&out, &s.smoke);
        // 要保住的是煙火線條本身；線條之間那層被照亮的煙該扣就扣
        let trails = mean_level(&out, &s.trails);
        println!(
            "煙霧面 {before:.1} → {after:.1}（留 {:.0}%）、煙火線條 {:.1} → {trails:.1}",
            after / before * 100.0,
            mean_level(&s.img, &s.trails),
        );
        assert!(after < before * 0.35, "自動參數沒把煙扣掉：{after:.1}");
        assert!(
            trails > mean_level(&s.img, &s.trails) * 0.5,
            "自動參數把煙火線條扣暗了：{trails:.1}"
        );
    }

    /// 量的是照片本身，不是餵進去的尺寸：拿縮圖量要與拿原圖量得到同一組值，
    /// GUI 才能用手上現成的預覽底圖量，不必為了判參數再解一次原尺寸
    #[test]
    fn auto_params_read_the_same_off_a_thumbnail() {
        let s = smoky_fireworks(2000, 1400);
        let long = s.img.width();
        let full = auto_params(&s.img, long);
        let small = auto_params(&shrink(&s.img, 500), long);
        println!("原圖 {full:?} / 縮圖 {small:?}");
        assert!(
            (full.strength - small.strength).abs() <= 4
                && (full.detail - small.detail).abs() <= 4
                && full.sky_clean == small.sky_clean,
            "縮圖量出來差太多：{full:?} vs {small:?}"
        );
    }

    /// 煙已經扣掉的照片再判一次，強度就該退下來——
    /// 判的是「這張還剩多少煙」，不是套一個固定的數字
    #[test]
    fn auto_asks_for_less_once_the_smoke_is_gone() {
        let s = smoky_fireworks(1600, 1100);
        let smoky = auto_params(&s.img, s.img.width());
        let cleared = remove_smoke(
            &s.img,
            &SmokeParams {
                strength: 100,
                ..Default::default()
            },
        );
        let done = auto_params(&cleared, cleared.width());
        println!("有煙 {smoky:?} / 扣過 {done:?}");
        assert!(
            done.strength < smoky.strength,
            "扣過的照片仍判出一樣重的強度：{smoky:?} → {done:?}"
        );
    }

    /// 清雲收的是「去除扣不完」的那一截，不是「照片裡有沒有雲」。
    /// 開檔時清雲一律照下限開著（見 [`AUTO_CLEAN`]），但那是保底值——
    /// 天上加一片去除扣得掉的亮雲，判出來的清雲仍該停在下限，不能被推高。
    /// 這一關擋的是「看到雲就把清雲拉高」那種寫法：清雲會連星點與薄雲的層次
    /// 一起抹平，去除做得到的事就不該交給它
    #[test]
    fn a_cloud_the_dehaze_can_reach_does_not_push_cloud_clean_past_the_floor() {
        let s = smoky_fireworks(1600, 1100);
        let clear = auto_params(&s.img, s.img.width());
        assert_eq!(
            clear.sky_clean, AUTO_CLEAN.0,
            "乾淨的夜空該停在保底值：{clear:?}"
        );
        let mut cloudy = s.img.clone();
        add_cloud(&mut cloudy, 70.0);
        let with_cloud = auto_params(&cloudy, cloudy.width());
        println!("無雲 {clear:?} / 有雲 {with_cloud:?}");
        assert_eq!(
            with_cloud.sky_clean, AUTO_CLEAN.0,
            "去除扣得掉的雲不該把清雲推高：{with_cloud:?}"
        );
    }

    /// 預覽的強度要折得比滑桿小（縮圖沒有原尺寸的最小值池化，同樣的強度扣得更乾淨），
    /// 而且照片本來就不大時不能亂折——那時預覽與成品走的是同一條路
    #[test]
    fn preview_strength_only_bends_for_photos_bigger_than_the_work_size() {
        // 四千萬畫素的照片（長邊 8251）：滑桿 55 在 1600 的預覽上要畫成 51，
        // 才等於原尺寸 55 存出來的樣子。倍率照 AUTO_POOL_EXP 的實測校正
        // （3.2^0.07 ≈ 1.085）；取最小值前先抹雜訊之後，差距只剩一成上下
        assert_eq!(preview_strength(55, 8251, 1600), 51);
        assert_eq!(preview_strength(80, 8251, 1600), 74);
        // 原圖沒有大過工作解析度就沒有那一段落差，折算必須是恆等的
        assert_eq!(preview_strength(80, WORK_LONG_EDGE, 1600), 80);
        assert_eq!(preview_strength(80, 1200, 1200), 80);
        // 預覽尺寸換成 2560 也一樣要折：差的是池化，不是預覽多大
        assert_eq!(
            preview_strength(80, 8251, 2560),
            preview_strength(80, 8251, 1600)
        );
        assert_eq!(preview_strength(0, 8251, 1600), 0);
    }

    /// 天空太少就不自作聰明：量不到夜空的照片一律維持預設值
    #[test]
    fn auto_keeps_the_defaults_when_there_is_no_sky() {
        let img = solid(400, 300, [180, 170, 160]);
        assert_eq!(auto_params(&img, 400), AutoParams::default());
    }

    /// 原圖越大，估煙霧層的下採樣就吃掉越多煙，強度要跟著補上去。
    /// 同一張照片只是宣稱的原尺寸不同，判出來的強度就該不同
    #[test]
    fn auto_compensates_for_the_downsampling_of_big_photos() {
        let s = smoky_fireworks(1600, 1100);
        let small = auto_params(&s.img, WORK_LONG_EDGE);
        let huge = auto_params(&s.img, WORK_LONG_EDGE * 4);
        println!("原尺寸 {small:?} / 四倍大 {huge:?}");
        assert!(
            huge.strength > small.strength,
            "大照片沒補回下採樣吃掉的那一截：{small:?} vs {huge:?}"
        );
        assert_eq!(
            (huge.detail, huge.sky_clean),
            (small.detail, small.sky_clean),
            "補償只該動強度"
        );
    }
    /// 診斷用：把合成場景與去煙結果寫到暫存資料夾，用眼睛看殘留長什麼樣
    #[test]
    #[ignore]
    fn dump_synthetic_scene() {
        let dir = std::env::var("SMOKE_DUMP").expect("設定 SMOKE_DUMP=輸出資料夾");
        let s = smoky_fireworks(2400, 1700);
        s.img.save(format!("{dir}/in.png")).unwrap();
        for strength in [60, 80, 100] {
            let out = remove_smoke(
                &s.img,
                &SmokeParams {
                    strength,
                    ..Default::default()
                },
            );
            out.save(format!("{dir}/out{strength}.png")).unwrap();
        }
        let (layer, _) = debug_smoke_layer(&s.img, &SmokeParams::default());
        layer.save(format!("{dir}/layer.png")).unwrap();
        // 少扣多少：白＝煙火簇，煙霧層會留下來
        let (fw, fh) = (s.img.width() as usize, s.img.height() as usize);
        let (ww, wh) = work_size(fw, fh, WORK_LONG_EDGE);
        let b = downsample(&s.img, &srgb_lut(), ww, wh);
        for (name, p) in [("gate", streak_gate(&b, fw, ww, wh))] {
            let mut g = RgbImage::new(ww as u32, wh as u32);
            for (i, px) in g.pixels_mut().enumerate() {
                let v = (p.d[i].clamp(0.0, 1.0) * 255.0) as u8;
                *px = Rgb([v, v, v]);
            }
            g.save(format!("{dir}/{name}.png")).unwrap();
        }
        // 順便量一張全片幅照片的處理時間（預覽要即時）
        let big = smoky_fireworks(6000, 4000).img;
        let t = std::time::Instant::now();
        remove_smoke(&big, &SmokeParams::default());
        println!("6000x4000 去煙耗時 {:?}", t.elapsed());
    }

    /// 滑動極值濾波要與暴力法一致
    #[test]
    fn sliding_extremes_match_brute_force() {
        let w = 23;
        let h = 5;
        let mut p = Plane::new(w, h);
        for (i, v) in p.d.iter_mut().enumerate() {
            // 有起伏又不單調的樣本
            *v = ((i * 7 % 13) as f32 - 6.0) / 6.0;
        }
        for r in [1usize, 3, 8, 30] {
            let lo = min_filter(&p, r);
            let hi = max_filter(&p, r);
            for y in 0..h {
                for x in 0..w {
                    let (mut bmin, mut bmax) = (f32::INFINITY, f32::NEG_INFINITY);
                    for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                        for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                            bmin = bmin.min(p.d[yy * w + xx]);
                            bmax = bmax.max(p.d[yy * w + xx]);
                        }
                    }
                    assert_eq!(lo.d[y * w + x], bmin, "min r={r} at ({x},{y})");
                    assert_eq!(hi.d[y * w + x], bmax, "max r={r} at ({x},{y})");
                }
            }
        }
    }
}


