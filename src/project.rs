//! 各模組的專案檔：把這一次編輯的每一項設定寫成 JSON，下次開回來接著做。
//!
//! 五個模組（除了檔案管理）共用同一個副檔名 `.p2v`，靠檔案裡的 `module`
//! 欄位分辨是哪一種——開檔時就能自動切到對的模組，使用者不必先切好再開。
//! 影片模組的那一份在 [`crate::ProjectFile`]（先有的，欄位名不動），
//! 其餘四個在這裡。
//!
//! 每一份都掛 `#[serde(default)]`：新版加欄位之後，舊版程式開新版的檔
//! （或反之）只會少掉那幾項，不會整份開不起來。HashMap 一律轉成陣列再存，
//! JSON 物件的鍵順序不保證穩定，同一份專案每次存出來的內容才一樣。

use std::path::{Path, PathBuf};

use crate::{dehaze, edit, enhance, stack};
use crate::{Adjustments, Crop, CropAspect, EnhanceParams, GradeTarget};
use crate::{MovieGrade, MovieSize, MovieText, Segment};

/// 專案檔放在使用者那批照片（或影片）所在資料夾底下的這個子資料夾，
/// 底下再照模組各分一層（見 [`project_dir`]）
pub const PROJECT_DIR: &str = "專案";

/// `module` 欄位的值。**改動會讓舊專案檔認不出模組**，勿隨意更名
pub const KIND_VIDEO: &str = "video";
pub const KIND_DEHAZE: &str = "dehaze";
pub const KIND_STACK: &str = "stack";
pub const KIND_MOVIE: &str = "movie";
pub const KIND_ENHANCE: &str = "enhance";

/// 目前的專案檔格式版本
pub const VERSION: u32 = 1;

pub fn app_version() -> String {
    env!("CARGO_PKG_VERSION").into()
}

// ---------- 去煙霧 ----------

/// 「去煙霧」的專案檔。逐張的那幾份（參數覆寫、後製覆寫、筆跡、自動值）
/// 都以照片路徑為鍵，開檔時照片還在才套得回去
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DehazeProject {
    pub version: u32,
    pub module: String,
    /// 存檔當下的程式版本（僅供除錯參考）
    pub app_version: String,
    pub photos: Vec<PathBuf>,
    /// 存檔時停在第幾張
    pub cur: usize,
    /// 共用的去煙參數：沒有個別設定的照片都套這組
    pub params: dehaze::SmokeParams,
    /// 個別照片的去煙參數覆寫
    pub overrides: Vec<(PathBuf, dehaze::SmokeParams)>,
    /// 共用的後製（調色、文字、疊上去的圖片）
    pub finish: edit::Finish,
    pub finish_overrides: Vec<(PathBuf, edit::Finish)>,
    /// 逐張的手動清除筆跡
    pub wipes: Vec<(PathBuf, Vec<edit::Wipe>)>,
    /// 每張量出來的自動值。存著開檔就不必再量一次——一批幾十張要等上一陣子，
    /// 而且量出來本來就會是同一組
    pub auto: Vec<(PathBuf, dehaze::AutoParams)>,
    pub auto_on: bool,
    pub per_photo: bool,
    /// 文字樣式的字型**名稱**；開檔時依名稱找回索引，找不到就用第一個字型
    /// （字型清單依系統而異，存索引換一臺電腦就對不上）
    pub text_font: String,
    pub text_color: [u8; 4],
    pub text_outline_w: i32,
    pub text_outline_color: [u8; 4],
    pub text_boxed: bool,
    /// 遮色片工具的設定
    pub brush_size: i32,
    pub radial_invert: bool,
    pub object_feather: i32,
    pub object_edge: i32,
    /// 手動清除筆刷的設定
    pub wipe_size: i32,
    pub wipe_feather: i32,
    pub wipe_flow: i32,
    pub wipe_density: i32,
    pub wipe_keep: bool,
    pub crop_aspect: CropAspect,
    /// 縮圖列上挑起來的那幾張（存檔只存這幾張）
    pub multi_sel: Vec<PathBuf>,
    /// 右側面板與設定區各區塊的收合狀態
    pub sky_open: bool,
    pub grade_open: bool,
    pub text_open: bool,
    pub image_open: bool,
    pub wipe_open: bool,
}

impl Default for DehazeProject {
    fn default() -> Self {
        let params = dehaze::SmokeParams::default();
        Self {
            version: VERSION,
            module: KIND_DEHAZE.into(),
            app_version: String::new(),
            photos: Vec::new(),
            cur: 0,
            params,
            overrides: Vec::new(),
            finish: edit::Finish::default(),
            finish_overrides: Vec::new(),
            wipes: Vec::new(),
            auto: Vec::new(),
            auto_on: true,
            per_photo: true,
            text_font: String::new(),
            text_color: [255, 255, 255, 255],
            text_outline_w: 2,
            text_outline_color: [0, 0, 0, 255],
            text_boxed: false,
            brush_size: 10,
            radial_invert: false,
            object_feather: dehaze::OBJECT_FEATHER,
            object_edge: 0,
            wipe_size: 6,
            wipe_feather: 45,
            wipe_flow: 100,
            wipe_density: 100,
            wipe_keep: true,
            crop_aspect: CropAspect::Free,
            multi_sel: Vec::new(),
            sky_open: false,
            grade_open: true,
            text_open: true,
            image_open: true,
            wipe_open: true,
        }
    }
}

// ---------- 煙火疊圖 ----------

/// 「煙火疊圖」的專案檔
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StackProject {
    pub version: u32,
    pub module: String,
    pub app_version: String,
    /// 照片清單與順序（順序決定誰疊在誰上面）
    pub photos: Vec<PathBuf>,
    /// 哪一張當地景
    pub ground: usize,
    pub cur: usize,
    pub pro: bool,
    pub mode: stack::BlendMode,
    pub blend_noted: bool,
    /// 逐張的遮色片
    pub masks: Vec<(PathBuf, Vec<dehaze::Shape>)>,
    /// 逐張的遮色片方向（true＝只疊畫到的地方）
    pub mask_polarity: Vec<(PathBuf, bool)>,
    pub feather: i32,
    pub mask_density: i32,
    /// 使用者自己拖出來的位移（逐張）
    pub xforms: Vec<(PathBuf, stack::Xform)>,
    /// 每一層疊進去之前先套的調色（逐張）
    pub grades: Vec<(PathBuf, Adjustments)>,
    /// 自動對齊算出來的位移；None＝那張對不上地景。存著開檔就不必重算一次
    pub auto_offsets: Vec<(PathBuf, Option<stack::Xform>)>,
    pub auto_align: bool,
    pub protect_land: bool,
    /// 疊完之後才套的調色（含裁切）
    pub grade: Adjustments,
    pub grade_target: GradeTarget,
    pub grade_open: bool,
    pub crop_aspect: CropAspect,
    pub brush_size: i32,
    pub radial_invert: bool,
    pub object_feather: i32,
    pub object_edge: i32,
    pub move_ghost: bool,
    pub move_mode: bool,
    pub show_mask: bool,
}

impl Default for StackProject {
    fn default() -> Self {
        Self {
            version: VERSION,
            module: KIND_STACK.into(),
            app_version: String::new(),
            photos: Vec::new(),
            ground: 0,
            cur: 0,
            pro: false,
            mode: stack::BlendMode::Lighten,
            blend_noted: false,
            masks: Vec::new(),
            mask_polarity: Vec::new(),
            feather: 25,
            mask_density: 80,
            xforms: Vec::new(),
            grades: Vec::new(),
            auto_offsets: Vec::new(),
            auto_align: true,
            protect_land: true,
            grade: Adjustments::default(),
            grade_target: GradeTarget::Output,
            grade_open: true,
            crop_aspect: CropAspect::Free,
            brush_size: 10,
            radial_invert: false,
            object_feather: dehaze::OBJECT_FEATHER,
            object_edge: 0,
            move_ghost: false,
            move_mode: false,
            show_mask: false,
        }
    }
}

// ---------- 煙火影片去煙霧 ----------

/// 「煙火影片去煙霧」的專案檔。這個模組的設定整支影片共用一組，
/// 沒有「逐張」的東西；分段的遮色片存在 `segments` 裡
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MovieProject {
    pub version: u32,
    pub module: String,
    pub app_version: String,
    /// 正在預覽的那支影片
    pub src: Option<PathBuf>,
    /// 排在後面、要套同一組設定一起處理的其他影片
    pub queue: Vec<PathBuf>,
    pub merge: bool,
    /// 預覽停在第幾秒
    pub at: f64,
    /// 去煙參數（整支共用）
    pub params: dehaze::SmokeParams,
    /// 分區調色（整支共用）
    pub grade: MovieGrade,
    /// 照時間切的段，每段各有自己的遮色片
    pub segments: Vec<Segment>,
    pub size: MovieSize,
    pub crop: Crop,
    pub crop_aspect: CropAspect,
    pub auto_on: bool,
    pub pro: bool,
    pub grade_open: bool,
    pub brush_size: i32,
    pub radial_invert: bool,
    pub object_feather: i32,
    pub object_edge: i32,
    /// 試播要抓幾秒
    pub clip_secs: f64,
    /// 疊在畫面上的文字（整支共用，每段各有自己的秒數）
    pub texts: Vec<MovieText>,
    /// 字型名稱；開檔時依名稱找回索引，找不到就用第一個字型
    /// （與去煙霧模組同一套，見 [`DehazeProject::text_font`]）
    pub text_font: String,
    pub text_color: [u8; 4],
    pub text_outline_w: i32,
    pub text_outline_color: [u8; 4],
    pub text_boxed: bool,
    /// 背景音樂：檔案，以及音樂與影片原聲各自的音量（百分比）
    pub music_path: Option<PathBuf>,
    pub music_volume: i32,
    pub src_volume: i32,
    pub music_fade: bool,
    /// 三個可收合區塊當時是開是合
    pub text_open: bool,
    pub mask_open: bool,
    pub music_open: bool,
}

impl Default for MovieProject {
    fn default() -> Self {
        Self {
            version: VERSION,
            module: KIND_MOVIE.into(),
            app_version: String::new(),
            src: None,
            queue: Vec::new(),
            merge: false,
            at: 0.0,
            params: dehaze::SmokeParams::default(),
            grade: MovieGrade::default(),
            segments: vec![Segment::default()],
            size: MovieSize::Source,
            crop: Crop::default(),
            crop_aspect: CropAspect::Free,
            auto_on: false,
            pro: false,
            grade_open: false,
            brush_size: 10,
            radial_invert: false,
            object_feather: dehaze::OBJECT_FEATHER,
            object_edge: 0,
            clip_secs: 5.0,
            texts: Vec::new(),
            text_font: String::new(),
            text_color: [255, 255, 255, 255],
            text_outline_w: 2,
            text_outline_color: [0, 0, 0, 255],
            text_boxed: false,
            music_path: None,
            music_volume: 100,
            src_volume: 100,
            music_fade: true,
            text_open: false,
            mask_open: true,
            music_open: false,
        }
    }
}

// ---------- 優化影像 ----------

/// 「優化影像」的專案檔
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EnhanceProject {
    pub version: u32,
    pub module: String,
    pub app_version: String,
    pub photos: Vec<PathBuf>,
    pub cur: usize,
    /// 整批共用的類型
    pub preset: enhance::Preset,
    /// 替某一張單獨挑過的類型
    pub preset_overrides: Vec<(PathBuf, enhance::Preset)>,
    /// 每張量出來的自動建議，連同「是用哪個類型量的」。
    /// 存著開檔就不必再量一次（見 [`DehazeProject::auto`]）
    pub auto: Vec<(PathBuf, enhance::Preset, enhance::Auto)>,
    pub auto_on: bool,
    pub auto_amount: i32,
    /// 共用設定（強化那兩條滑桿，以及沒量到自動值時當底的調色）
    pub params: EnhanceParams,
    /// 個別照片的調色覆寫
    pub grade_overrides: Vec<(PathBuf, Adjustments)>,
    /// 個別照片的強化覆寫
    pub local_overrides: Vec<(PathBuf, enhance::Local)>,
    pub per_photo: bool,
    pub crop_aspect: CropAspect,
    pub grade_open: bool,
    pub show_mask: bool,
}

impl Default for EnhanceProject {
    fn default() -> Self {
        let preset = enhance::Preset::Bird;
        Self {
            version: VERSION,
            module: KIND_ENHANCE.into(),
            app_version: String::new(),
            photos: Vec::new(),
            cur: 0,
            preset,
            preset_overrides: Vec::new(),
            auto: Vec::new(),
            auto_on: true,
            auto_amount: 100,
            params: EnhanceParams::neutral(preset),
            grade_overrides: Vec::new(),
            local_overrides: Vec::new(),
            per_photo: false,
            crop_aspect: CropAspect::Free,
            grade_open: true,
            show_mask: false,
        }
    }
}

// ---------- 開檔：先認模組，再當成那一種讀 ----------

/// 讀進來的一份專案檔。`module` 欄位決定是哪一種——開檔的人不必先切到
/// 對的模組，[`crate::App::open_project_path`] 會自己切過去
pub enum AnyProject {
    Video(Box<crate::ProjectFile>),
    Dehaze(Box<DehazeProject>),
    Stack(Box<StackProject>),
    Movie(Box<MovieProject>),
    Enhance(Box<EnhanceProject>),
}

impl AnyProject {
    /// 這份專案是哪個模組的
    pub fn module(&self) -> crate::Module {
        match self {
            AnyProject::Video(_) => crate::Module::Video,
            AnyProject::Dehaze(_) => crate::Module::Dehaze,
            AnyProject::Stack(_) => crate::Module::Stack,
            AnyProject::Movie(_) => crate::Module::Movie,
            AnyProject::Enhance(_) => crate::Module::Enhance,
        }
    }
}

/// 解析專案檔內容。先只看 `module` 一個欄位（其餘欄位這時不必成立），
/// 再照它挑對應的結構完整讀一次。
///
/// 沒有 `module` 欄位的一律當成影片模組：那是這個欄位還不存在時
/// 存出來的 .p2v，全都是影片專案
pub fn parse(txt: &str) -> Result<AnyProject, String> {
    // 拿掉開頭的 UTF-8 BOM：我們自己寫出來的沒有，但使用者用記事本開來改過
    // 存回去就會多這三個位元組，serde 認不得它、整份專案就開不起來
    let txt = txt.strip_prefix('\u{feff}').unwrap_or(txt);
    let probe: serde_json::Value =
        serde_json::from_str(txt).map_err(|e| format!("內容不是有效的專案檔：{e}"))?;
    let kind = probe
        .get("module")
        .and_then(|v| v.as_str())
        .unwrap_or(KIND_VIDEO);
    let bad = |e: serde_json::Error| format!("專案內容讀不起來：{e}");
    Ok(match kind {
        KIND_DEHAZE => AnyProject::Dehaze(Box::new(serde_json::from_str(txt).map_err(bad)?)),
        KIND_STACK => AnyProject::Stack(Box::new(serde_json::from_str(txt).map_err(bad)?)),
        KIND_MOVIE => AnyProject::Movie(Box::new(serde_json::from_str(txt).map_err(bad)?)),
        KIND_ENHANCE => AnyProject::Enhance(Box::new(serde_json::from_str(txt).map_err(bad)?)),
        // 認不得的 module（更新版寫的、或手改壞了）也照影片模組試一次：
        // 讀得起來就當影片專案，讀不起來才報錯
        _ => AnyProject::Video(Box::new(serde_json::from_str(txt).map_err(bad)?)),
    })
}

// ---------- 檔名與存放位置 ----------

/// 專案檔要放哪個資料夾：使用者開進來的那批照片（或影片）所在的資料夾底下，
/// 再加一層「專案」。
///
/// 「專案」底下**再照模組各分一層**（`sub` 給模組名，如「疊圖」）：五個模組
/// 共用同一個副檔名，Windows 的存檔對話框只能照副檔名過濾，分資料夾才不會
/// 在疊圖的存檔視窗裡看到去煙霧的專案、一個手滑就存錯。
///
/// `src` 給的是那批的第一個檔案。`create` 為真（存檔）時資料夾不存在就建
/// 起來，建不起來（唯讀磁碟、權限不足、網路磁碟斷線）就退回照片自己的
/// 資料夾——存得進去比存在指定位置重要。
///
/// 為假（開檔）時不會建任何東西：那一層還沒有就退回「專案」本身（還沒分層
/// 之前存的檔都在那裡，不然舊檔會變得找不到），再沒有就回 None 讓呼叫端
/// 改用上次的位置
pub fn project_dir(src: Option<&Path>, sub: &str, create: bool) -> Option<PathBuf> {
    let parent = src?.parent().filter(|p| !p.as_os_str().is_empty())?;
    let root = parent.join(PROJECT_DIR);
    let dir = if sub.is_empty() {
        root.clone()
    } else {
        root.join(sub)
    };
    if dir.is_dir() {
        return Some(dir);
    }
    if !create {
        return root.is_dir().then_some(root);
    }
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(dir),
        Err(_) => Some(parent.to_path_buf()),
    }
}

/// 在名字後面接上日期時間：「去煙霧」→「去煙霧-20260914-1300」
/// （1300＝下午一點）。
///
/// `stamp` 由呼叫端給，**一批要共用同一個**——整批存出來的好幾個檔帶同一個
/// 時間，才看得出是同一次處理的（各自取的話會跨到不同分鐘）。
/// None＝取不到本機時間，那就只有名字那一段
pub fn with_stamp(name: &str, stamp: Option<&str>) -> String {
    let name = name.trim();
    match (name.is_empty(), stamp) {
        (false, Some(s)) => format!("{name}-{s}"),
        (false, None) => name.to_string(),
        (true, Some(s)) => s.to_string(),
        (true, None) => String::new(),
    }
}

/// 「去煙霧-20260914-1300」：只存一個檔時用，時間就取現在這一刻
pub fn file_stem(prefix: &str) -> String {
    with_stamp(prefix, crate::date_stamp().as_deref())
}

/// 把使用者自己取的名字從檔名裡拆回來：尾巴那一段
/// `-YYYYMMDD-HHMM` 是程式加的，下次存檔要換成新的時間，不能一起留著。
/// 沒有那一段（使用者整個改掉了）就原樣當成他取的名字
pub fn strip_stamp(stem: &str) -> &str {
    // 「-20260914-1300」固定 14 個字元，且日期與時間都是數字
    let Some(cut) = stem.len().checked_sub(14) else {
        return stem;
    };
    let tail = &stem[cut..];
    let ok = tail.len() == 14
        && tail.starts_with('-')
        && tail[1..9].bytes().all(|b| b.is_ascii_digit())
        && tail.as_bytes()[9] == b'-'
        && tail[10..].bytes().all(|b| b.is_ascii_digit());
    if ok {
        stem[..cut].trim_end_matches('-')
    } else {
        stem
    }
}

/// config.json 裡記「使用者上次替這個模組的專案取了什麼名字」的欄位。
/// 成品的檔名是算出來的（見 `crate::video_file_stem` 與各模組的存檔），
/// 不記名字，所以這裡只有專案這一種
fn name_key(kind: &str) -> String {
    format!("project_name_{kind}")
}

/// 使用者上次替這個模組的專案取的名字；沒存過就用起始值
pub fn saved_name(kind: &str, fallback: &str) -> String {
    crate::load_config()
        .get(name_key(kind))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// 記下這次用的名字，下次存檔就預帶它（時間的部分照樣換成新的）
pub fn remember_name(kind: &str, name: &str) {
    let name = name.trim();
    if name.is_empty() {
        return;
    }
    crate::update_config(&name_key(kind), serde_json::json!(name));
}
