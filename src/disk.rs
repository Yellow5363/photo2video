//! 硬碟管理 ▸ 硬碟檔案管理：把一顆硬碟底下的資料夾掃成一張表。
//!
//! 做的事只有一件：走過指定的硬碟，把**每一個資料夾**記成一列——硬碟編號、
//! 完整路徑、資料夾名稱、裡面檔案的大小總和、修改時間。掃完的結果全部收進
//! **同一張總表**（所有硬碟共用一張），日後要找「那批照片在哪一顆碟裡」就查
//! 這張表，不必把碟一顆一顆插回去開來看。
//!
//! 兩個規矩決定了這張表長什麼樣，兩個都是使用者定的：
//!
//! 1. **大小只算那個資料夾自己的檔案**，不含子資料夾。`照片_ok\_縮小` 那一列
//!    的數字，就是點進 `_縮小` 看到的那幾個檔案加起來——不是整棵子樹。
//!    （整棵子樹的總和在這裡沒有用：要找的是「東西放在哪一層」。）
//! 2. **掃描停在指定的層數**，不是「只列到那一層」。第 5 層底下還有東西也
//!    不走進去，所以掃一顆碟的時間是可預期的。

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// 使用者按了中止時回報的訊息（main.rs 靠它分辨「中止」與「出錯」，
/// 與 [`crate::backup::CANCELLED`] 同一套用法）
pub const CANCELLED: &str = "已中止";

/// 總表的檔名（索引檔，程式自己讀的那一份）。
/// 匯出的 Excel 與 PDF 用同一個主檔名，三個檔放在同一個資料夾裡才看得出是一組
pub const TABLE_STEM: &str = "硬碟總表";

/// Excel 一張工作表最多放得下幾列（1048576 減掉標題那一列）。
/// 超過就不是「寫不漂亮」而是寫不進去，要先擋下來講清楚
pub const XLSX_MAX_ROWS: usize = 1_048_575;

/// 總表裡的一列。
///
/// 畫面與匯出的六個欄位裡，「資料夾大小」是從 `bytes` 算出來的
/// （見 [`fmt_gb`]），不另外存一份——同一個數字存兩次，日後一定有一邊是錯的。
///
/// 欄位名在 JSON 裡縮成一個字母：一顆碟動輒幾萬列，`"disk"`、`"path"` 這種
/// 長名字會讓索引檔大上一截，而這份檔案是給程式讀的（人要看的是 Excel）
#[derive(Clone, Serialize, Deserialize)]
pub struct Row {
    /// 硬碟編號（使用者給的那個號碼，同一顆碟底下每一列都一樣）
    #[serde(rename = "d")]
    pub disk: u32,
    /// 資料夾路徑：完整路徑，含磁碟機代號
    #[serde(rename = "p")]
    pub path: String,
    /// 資料夾名稱（路徑的最後一段）
    #[serde(rename = "n")]
    pub name: String,
    /// 資料夾大小：**這個資料夾裡的檔案**加總，不含子資料夾
    #[serde(rename = "b")]
    pub bytes: u64,
    /// 資料夾修改時間（unix 秒）
    #[serde(rename = "t")]
    pub mtime: u64,
}

/// 大小的人話寫法：GB 取到小數點後兩位，尾巴的零不留
/// （`0.30` 寫成 `0.3`），不到 0.005 GB 的一律是 `0 GB`。
///
/// 照舊工具的寫法來——使用者手上已經有一批用那個格式存的清單，
/// 兩邊的數字要能直接對照
pub fn fmt_gb(bytes: u64) -> String {
    const G: f64 = 1024.0 * 1024.0 * 1024.0;
    let gb = (bytes as f64 / G * 100.0).round() / 100.0;
    if gb == 0.0 {
        return "0 GB".into();
    }
    let mut s = format!("{gb:.2}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    format!("{s} GB")
}

/// 修改時間的寫法：`2022/12/15 09:53:11`（本機時間）。
/// 換不出來（系統呼叫失敗）就留空白，不要印出一個看不懂的數字
pub fn fmt_time(unix: u64) -> String {
    match crate::local_parts(unix) {
        Some((y, mo, d, h, mi, s)) => {
            format!("{y:04}/{mo:02}/{d:02} {h:02}:{mi:02}:{s:02}")
        }
        None => String::new(),
    }
}

/// 這個名字要不要**當作不存在**。
///
/// 判斷標準與資料備份一致（見 [`crate::backup`] 裡的 `is_junk`）：使用者在
/// 檔案總管看不到的東西，列進清單只會讓人以為程式壞了。回收筒與
/// 「System Volume Information」還讀不進去，掃到只會多一列錯誤訊息
fn is_junk(name: &str) -> bool {
    const JUNK: [&str; 8] = [
        ".DS_Store",
        ".fseventsd",
        ".Spotlight-V100",
        ".TemporaryItems",
        ".Trashes",
        "Thumbs.db",
        "$RECYCLE.BIN",
        "System Volume Information",
    ];
    name.starts_with("._") || JUNK.iter().any(|j| name.eq_ignore_ascii_case(j))
}

/// 掃完一顆碟的結果
pub struct Scanned {
    /// 掃到的每一個資料夾
    pub rows: Vec<Row>,
    /// 讀不進去而跳過的資料夾（權限不足、壞軌…）。
    /// 不是錯誤：一顆碟上有一兩個這種資料夾很常見，其餘照掃完，
    /// 但要讓使用者知道這份清單不是全部
    pub skipped: Vec<String>,
}

/// 掃一顆硬碟（或任何一個資料夾）底下的資料夾。
///
/// `depth`＝最多往下幾層，**根底下的第一層算第 1 層**；掃描本身就停在那一層，
/// 更深的資料夾不走進去、也不算進任何人的大小。
///
/// `report(已掃到幾個, 現在在哪)` 每掃完一個資料夾叫一次，給畫面更新進度用；
/// 叫得很頻繁，節流由呼叫端決定（見 `App::disk_scan`）。
///
/// 根資料夾讀不進去就直接失敗——那是碟拔掉了或選錯地方，繼續跑只會得到一份
/// 「什麼都沒有」的假清單，而那會被當成「這顆碟是空的」存進總表
pub fn scan(
    root: &Path,
    depth: u32,
    disk: u32,
    cancel: &AtomicBool,
    report: &mut dyn FnMut(usize, &Path),
) -> Result<Scanned, String> {
    let mut rows: Vec<Row> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    // 待走的資料夾與它在第幾層（根是第 0 層，不列進清單：那是整顆碟本身）
    let mut stack: Vec<(PathBuf, u32)> = vec![(root.to_path_buf(), 0)];
    while let Some((dir, level)) = stack.pop() {
        if cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        let rd = match fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) => {
                if level == 0 {
                    return Err(format!("無法讀取「{}」：{e}", dir.display()));
                }
                skipped.push(format!("{}（{e}）", dir.display()));
                continue;
            }
        };
        let mut bytes: u64 = 0;
        let mut subs: Vec<PathBuf> = Vec::new();
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            // 捷徑（符號連結／junction）不跟進去：它可能指回自己形成無窮迴圈，
            // 連結後面那份資料也不住在這顆碟的這個位置上
            if ft.is_symlink() {
                continue;
            }
            let name = e.file_name();
            let name = name.to_string_lossy();
            if is_junk(&name) {
                continue;
            }
            if ft.is_dir() {
                subs.push(e.path());
            } else if ft.is_file() {
                if let Ok(md) = e.metadata() {
                    bytes += md.len();
                }
            }
        }
        if level > 0 {
            rows.push(Row {
                disk,
                path: dir.to_string_lossy().into_owned(),
                name: dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                bytes,
                mtime: dir_mtime(&dir),
            });
        }
        // 停在第 depth 層：這一層的資料夾照樣列出來（連同它自己的檔案大小），
        // 只是不再往它底下走
        if level < depth {
            for s in subs {
                stack.push((s, level + 1));
            }
        }
        report(rows.len(), &dir);
    }
    sort_rows(&mut rows);
    Ok(Scanned { rows, skipped })
}

/// 資料夾的修改時間（unix 秒）。讀不到就是 0，對應畫面上的空白
fn dir_mtime(dir: &Path) -> u64 {
    fs::metadata(dir)
        .and_then(|md| md.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 總表的排序：**先照資料夾名稱**，同名的再照完整路徑。
///
/// 照名稱排是這張表的用法決定的——找東西時記得的是資料夾叫什麼
/// （`0722農舍`），不是它掛在哪一層。同名的資料夾（同一批照片拷過好幾份）
/// 因此會排在一起，一眼就看得出它在哪幾顆碟、哪幾個位置各有一份。
///
/// 名稱用自然排序（見 [`crate::natural_key`]）：`105` 排在 `15` 後面而不是
/// 中間，與檔案總管看到的順序一致
fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| {
        crate::natural_key(&a.name)
            .cmp(&crate::natural_key(&b.name))
            .then_with(|| a.disk.cmp(&b.disk))
            .then_with(|| crate::natural_key(&a.path).cmp(&crate::natural_key(&b.path)))
    });
}

/// 總表：所有硬碟的資料都在這一份裡。
///
/// 存成一份 JSON（索引檔），程式自己讀；人要看的是同一個資料夾裡的 Excel
/// 與 PDF。**不是一顆碟一個檔**——使用者要的就是「一張表」，分成好幾份的話
/// 「這個資料夾在哪幾顆碟裡」得開十幾個檔案才答得出來
#[derive(Default, Serialize, Deserialize)]
pub struct Table {
    /// 檔案格式版本。日後欄位有增減時，靠它認得出手上這份是舊的
    #[serde(default)]
    pub ver: u32,
    #[serde(default)]
    pub rows: Vec<Row>,
    /// 每一顆碟的附註：上次是從哪裡掃的、什麼時候掃的。
    ///
    /// **不是那六個欄位**，不會進 Excel 與 PDF，只給畫面上那一欄用——
    /// 「0001 是哪一顆碟、上次什麼時候讀的」是看表的人第一個會問的事，
    /// 而硬碟隨時在變，知道日期才知道這份清單還準不準
    #[serde(default)]
    pub notes: BTreeMap<u32, DiskNote>,
}

/// 一顆碟的附註（見 [`Table::notes`]）
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct DiskNote {
    /// 上次是從哪一個位置開始掃的（磁碟機代號會換，所以只當提示用）
    #[serde(rename = "r", default)]
    pub root: String,
    /// 上次掃完的時間（unix 秒）
    #[serde(rename = "a", default)]
    pub at: u64,
    /// 磁碟區名稱。代號會換、名字不會，碟插回來時認的是這個
    #[serde(rename = "l", default)]
    pub label: String,
    /// 磁碟區序號（見 [`Drive::serial`]）。0＝當時問不到
    #[serde(rename = "s", default)]
    pub serial: u32,
}

/// 現在寫出來的索引檔版本
const VER: u32 = 1;

impl Table {
    /// 索引檔的完整路徑（放在使用者挑的那個資料夾裡）
    pub fn path_in(dir: &Path) -> PathBuf {
        dir.join(format!("{TABLE_STEM}.json"))
    }

    /// 讀回總表。**檔案不存在不算錯誤**（第一次用就是這樣），回傳空表；
    /// 檔案壞了才是錯誤——這時不能當成空表，否則下一次存檔就把舊資料蓋掉了
    pub fn load(dir: &Path) -> Result<Table, String> {
        let p = Table::path_in(dir);
        let text = match fs::read_to_string(&p) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Table::default()),
            Err(e) => return Err(format!("讀不到總表「{}」：{e}", p.display())),
        };
        serde_json::from_str(&text).map_err(|e| format!("總表「{}」的內容壞了：{e}", p.display()))
    }

    /// 寫回總表。
    ///
    /// **先寫暫存檔再改名**（與設定檔同一套做法）：幾萬列寫到一半當掉或斷電，
    /// 留下的是半份 JSON，下次讀進來就是「總表壞了」，等於整批掃描的成果沒了
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        fs::create_dir_all(dir).map_err(|e| format!("建不出資料夾「{}」：{e}", dir.display()))?;
        let p = Table::path_in(dir);
        let tmp = p.with_extension(format!("json.{}.tmp", std::process::id()));
        let text = serde_json::to_string(&Table {
            ver: VER,
            rows: self.rows.clone(),
            notes: self.notes.clone(),
        })
        .map_err(|e| format!("總表轉不成 JSON：{e}"))?;
        if let Err(e) = fs::write(&tmp, text) {
            let _ = fs::remove_file(&tmp);
            return Err(format!("寫不進「{}」：{e}", tmp.display()));
        }
        fs::rename(&tmp, &p).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("存不進「{}」：{e}", p.display())
        })
    }

    /// 表裡已經有哪幾個硬碟編號（由小到大）
    pub fn disks(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.rows.iter().map(|r| r.disk).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 這個編號底下有幾列
    pub fn count_of(&self, disk: u32) -> usize {
        self.rows.iter().filter(|r| r.disk == disk).count()
    }

    /// 這顆碟（照磁碟區序號認）上次是幾號。認不出來就是 None。
    ///
    /// 碟插到別台電腦、或別的碟先占走代號時，同一顆碟的代號就變了；序號
    /// 不會變。靠它才能在碟插回來時直接帶出「上次的那個號」，而不是又給它
    /// 一個新號、讓同一顆碟在表裡出現兩次
    pub fn disk_of_serial(&self, serial: u32) -> Option<u32> {
        if serial == 0 {
            return None;
        }
        self.notes
            .iter()
            .find(|(_, n)| n.serial == serial)
            .map(|(no, _)| *no)
    }

    /// 下一個要用的編號：**從 1 開始找第一個空號**，中間沒有空號就接在
    /// 最大號後面加一。
    ///
    /// 從空號補起是使用者定的規矩——碟報廢、資料整批刪掉之後那個號就空出來，
    /// 不補的話號碼會一直往上跳，跟手邊貼在碟上的標籤對不起來
    pub fn next_no(&self) -> u32 {
        let used = self.disks();
        let mut want = 1;
        for n in used {
            if n > want {
                break;
            }
            if n == want {
                want += 1;
            }
        }
        want
    }

    /// 把某一個編號的資料整批換成新掃到的這一批。
    ///
    /// 硬碟隨時在新增、刪改檔案，所以同一顆碟會掃第二次、第三次——
    /// 這時要的是**換掉**，不是把兩次的結果疊在一起（疊起來的話，刪掉的
    /// 資料夾永遠留在表上，而那正是最容易找錯的一種假資料）
    pub fn replace_disk(&mut self, disk: u32, rows: Vec<Row>, note: DiskNote) {
        self.rows.retain(|r| r.disk != disk);
        self.rows.extend(rows);
        sort_rows(&mut self.rows);
        self.notes.insert(disk, note);
    }
}


// ---------- 這台電腦上有哪些硬碟 ----------

/// 一顆看得到的硬碟（磁碟機）。
///
/// 「指定硬碟」挑的是**一整顆碟**，不是某個資料夾——這個功能要記的就是
/// 「這顆碟裡有哪些資料夾」
#[derive(Clone, PartialEq)]
pub struct Drive {
    /// 根路徑，例如 `F:\`
    pub root: PathBuf,
    /// 磁碟機代號，例如 `F`
    pub letter: String,
    /// 磁碟區名稱（檔案總管上那個名字，例如「segate firecuda 530 2T #1」）。
    /// 沒取名就是空字串
    pub label: String,
    /// 磁碟區序號。同一顆碟插到別台電腦、拿到別的代號，這個數字不會變——
    /// 靠它認得出「這顆碟上次是幾號」（見 [`Table::disk_of_serial`]）
    pub serial: u32,
}

impl Drive {
    /// 畫面上的寫法，比照檔案總管：`segate firecuda 530 2T #1 (F:)`
    pub fn show(&self) -> String {
        if self.label.is_empty() {
            format!("({}:)", self.letter)
        } else {
            format!("{} ({}:)", self.label, self.letter)
        }
    }
}

/// 列出現在接著的硬碟。
///
/// 只收**固定式與可卸除式**（內接碟、外接碟、隨身碟）：光碟機與網路磁碟機
/// 不是要建檔的對象，列出來只會讓人多挑錯一次。讀不到名字的碟照樣列
/// （沒取名的碟很常見），只有連類型都問不出來的才跳過
#[cfg(windows)]
pub fn list_drives() -> Vec<Drive> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalDrives() -> u32;
        fn GetDriveTypeW(root: *const u16) -> u32;
        fn GetVolumeInformationW(
            root: *const u16,
            name: *mut u16,
            name_len: u32,
            serial: *mut u32,
            max_component: *mut u32,
            flags: *mut u32,
            fs_name: *mut u16,
            fs_name_len: u32,
        ) -> i32;
    }
    /// DRIVE_REMOVABLE / DRIVE_FIXED
    const WANT: [u32; 2] = [2, 3];

    let mask = unsafe { GetLogicalDrives() };
    let mut out = Vec::new();
    for i in 0..26u32 {
        if mask & (1 << i) == 0 {
            continue;
        }
        let letter = (b'A' + i as u8) as char;
        let root = PathBuf::from(format!("{letter}:\\"));
        let wide: Vec<u16> = root
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        if !WANT.contains(&unsafe { GetDriveTypeW(wide.as_ptr()) }) {
            continue;
        }
        // 名字與序號問不到（碟還在喚醒、沒放片）就留空，碟照樣列出來
        let mut name = [0u16; 261];
        let mut serial = 0u32;
        let ok = unsafe {
            GetVolumeInformationW(
                wide.as_ptr(),
                name.as_mut_ptr(),
                name.len() as u32,
                &mut serial,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ) != 0
        };
        let label = if ok {
            let n = name.iter().position(|c| *c == 0).unwrap_or(0);
            String::from_utf16_lossy(&name[..n])
        } else {
            String::new()
        };
        out.push(Drive {
            root,
            letter: letter.to_string(),
            label,
            serial: if ok { serial } else { 0 },
        });
    }
    out
}

/// macOS／Linux：掛在 `/Volumes` 底下的那些，加上根目錄本身。
/// 沒有磁碟區序號這種東西，序號一律 0（認碟的那一段就自動失效）
#[cfg(not(windows))]
pub fn list_drives() -> Vec<Drive> {
    let mut out = vec![Drive {
        root: PathBuf::from("/"),
        letter: "/".into(),
        label: "開機磁碟".into(),
        serial: 0,
    }];
    if let Ok(rd) = fs::read_dir("/Volumes") {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                let label = e.file_name().to_string_lossy().into_owned();
                out.push(Drive {
                    root: p,
                    letter: label.clone(),
                    label,
                    serial: 0,
                });
            }
        }
    }
    out
}

// ---------- 匯出 ----------

/// 六個欄位的標題。Excel、PDF 與畫面上的清單共用同一組，順序就是使用者
/// 手上那份舊清單的順序——兩邊要能並排對照
pub const COLUMNS: [&str; 6] = [
    "硬碟編號",
    "資料夾路徑",
    "資料夾名稱",
    "資料夾大小",
    "資料夾修改時間",
    "資料夾大小(byte)",
];

/// 硬碟編號的寫法：補到四位數（`1` → `0001`）。
/// 使用者手邊貼在碟上的標籤就是四位數，畫面、Excel、PDF 一律照這個寫
pub fn fmt_no(disk: u32) -> String {
    format!("{disk:04}")
}

/// 把整張總表寫成 Excel。
///
/// **寫的是整張表、不是只有剛掃完那一顆**：使用者要的就是「一張表」，
/// 每掃一顆碟就重寫一次，手上永遠只有一份最新的清單
pub fn export_xlsx(rows: &[Row], path: &Path) -> Result<(), String> {
    use rust_xlsxwriter::{Color, Format, FormatAlign, Workbook};

    if rows.len() > XLSX_MAX_ROWS {
        return Err(format!(
            "總表有 {} 列，超過 Excel 一張工作表放得下的 {} 列",
            rows.len(),
            XLSX_MAX_ROWS
        ));
    }
    let mut book = Workbook::new();
    let sheet = book.add_worksheet();
    sheet
        .set_name("硬碟總表")
        .map_err(|e| format!("Excel 工作表命名失敗：{e}"))?;

    let head = Format::new()
        .set_bold()
        .set_font_color(Color::White)
        .set_background_color(Color::RGB(0x5B2C6F))
        .set_align(FormatAlign::Left);
    // 編號當成數字存、但顯示成 0001：存字串的話 Excel 會整欄標成
    // 「以文字儲存的數字」，排序也會變成字典序
    let no = Format::new().set_num_format("0000");
    let int = Format::new().set_num_format("#,##0");

    for (c, title) in COLUMNS.iter().enumerate() {
        sheet
            .write_string_with_format(0, c as u16, *title, &head)
            .map_err(|e| format!("寫不進 Excel：{e}"))?;
    }
    for (i, r) in rows.iter().enumerate() {
        let y = i as u32 + 1;
        sheet
            .write_number_with_format(y, 0, f64::from(r.disk), &no)
            .and_then(|s| s.write_string(y, 1, &r.path))
            .and_then(|s| s.write_string(y, 2, &r.name))
            .and_then(|s| s.write_string(y, 3, fmt_gb(r.bytes)))
            .and_then(|s| s.write_string(y, 4, fmt_time(r.mtime)))
            .and_then(|s| s.write_number_with_format(y, 5, r.bytes as f64, &int))
            .map_err(|e| format!("寫不進 Excel：{e}"))?;
    }
    // 欄寬照內容抓個大概（路徑最長）；凍結標題列、開篩選，開起來就能用
    for (c, w) in [10.0, 70.0, 28.0, 12.0, 20.0, 16.0].into_iter().enumerate() {
        let _ = sheet.set_column_width(c as u16, w);
    }
    let _ = sheet.set_freeze_panes(1, 0);
    let last = rows.len().max(1) as u32;
    let _ = sheet.autofilter(0, 0, last, 5);
    book.save(path)
        .map_err(|e| format!("存不進「{}」：{e}", path.display()))
}

/// PDF 一頁的尺寸與留白（A4 橫式，單位 mm）。
/// 直式放不下六個欄位——光是完整路徑就要七成的寬度
const PDF_W: f32 = 297.0;
const PDF_H: f32 = 210.0;
const PDF_MARGIN: f32 = 10.0;
/// 內文字級（pt）。再大就塞不下路徑，再小列印出來看不清楚
const PDF_FONT_PT: f32 = 7.0;
/// 一列的高度（mm）
const PDF_ROW_H: f32 = 4.6;
/// 六個欄位各佔多寬（mm），加起來是 `PDF_W - 2 * PDF_MARGIN`
const PDF_COLS: [f32; 6] = [16.0, 118.0, 56.0, 20.0, 40.0, 27.0];

/// 1 mm 等於幾 pt（PDF 的字級是 pt，版面是 mm）
const MM_PT: f32 = 72.0 / 25.4;

/// 把整張總表寫成 PDF。`font` 是系統中文字型的位元組
/// （見 `load_cjk_font_bytes`）——沒有它，中文在 PDF 裡全是空白。
///
/// 字型會**只嵌入用到的字**（printpdf 預設做子集化），所以檔案不會因為
/// 嵌了一份 20MB 的黑體而爆掉
pub fn export_pdf(rows: &[Row], path: &Path, font: &[u8]) -> Result<(), String> {
    use printpdf::{
        Color, Line, LinePoint, Mm, Op, ParsedFont, PdfDocument, PdfFontHandle, PdfPage,
        PdfSaveOptions, Point, Pt, Rgb, TextItem, TextMatrix,
    };

    /// 把一格的位置（mm）換成 PDF 的文字矩陣。
    ///
    /// **一定要用 `Tm`（`SetTextMatrix`）而不是 `Td`（`SetTextCursor`）**：
    /// `Td` 是「相對上一行起點位移」，一格一格設下去會一路累加，第二格以後
    /// 就飛出紙外了（實際踩過：整份 PDF 只印得出第一格「硬碟編號」）。
    /// `Tm` 直接換掉整個矩陣，給的就是頁面左下角起算的絕對位置
    fn at(x_mm: f32, y_mm: f32) -> Op {
        Op::SetTextMatrix {
            matrix: TextMatrix::Translate(Mm(x_mm).into(), Mm(y_mm).into()),
        }
    }

    let parsed = ParsedFont::from_bytes(font, 0, &mut Vec::new())
        .ok_or_else(|| "讀不懂系統中文字型，PDF 裡的中文會變成空白".to_string())?;
    // 量字寬用另一套（ab_glyph，專案本來就有）：printpdf 的字型型別會隨著
    // 開了哪些功能而換一種，量寬度的方法也跟著換；排版這邊不跟著它走
    let metrics = ab_glyph::FontRef::try_from_slice_and_index(font, 0)
        .map_err(|e| format!("讀不懂系統中文字型：{e}"))?;
    let mut doc = PdfDocument::new("硬碟總表");
    let font_id = doc.add_font(&parsed);
    let handle = PdfFontHandle::External(font_id);

    // 每一欄的左邊界（mm）
    let mut xs = [0f32; 6];
    let mut x = PDF_MARGIN;
    for (i, w) in PDF_COLS.iter().enumerate() {
        xs[i] = x;
        x += w;
    }
    let right = x;
    // 標題列底下留一點空，再開始排內容
    let head_y = PDF_H - PDF_MARGIN - PDF_ROW_H;
    let first_y = head_y - PDF_ROW_H * 1.6;
    let rows_per_page = (((first_y - PDF_MARGIN - 8.0) / PDF_ROW_H).floor() as usize).max(1);
    let total_pages = rows.len().div_ceil(rows_per_page).max(1);
    let stamp = fmt_time(now_secs());

    let black = Color::Rgb(Rgb {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        icc_profile: None,
    });
    let gray = Color::Rgb(Rgb {
        r: 0.45,
        g: 0.45,
        b: 0.45,
        icc_profile: None,
    });

    let mut pages: Vec<PdfPage> = Vec::with_capacity(total_pages);
    for (p, chunk) in rows.chunks(rows_per_page).enumerate() {
        let mut ops: Vec<Op> = Vec::with_capacity(chunk.len() * 14 + 24);
        // 標題列底下那條橫線（整頁只畫這一條：一列一條的話，操作數會翻倍，
        // 幾萬列的 PDF 光是產生就要等很久）
        ops.push(Op::SetOutlineThickness { pt: Pt(0.5) });
        ops.push(Op::SetOutlineColor { col: gray.clone() });
        ops.push(Op::DrawLine {
            line: Line {
                points: vec![
                    LinePoint {
                        p: Point::new(Mm(PDF_MARGIN), Mm(head_y - 1.5)),
                        bezier: false,
                    },
                    LinePoint {
                        p: Point::new(Mm(right), Mm(head_y - 1.5)),
                        bezier: false,
                    },
                ],
                is_closed: false,
            },
        });
        ops.push(Op::StartTextSection);
        ops.push(Op::SetFont {
            font: handle.clone(),
            size: Pt(PDF_FONT_PT),
        });
        ops.push(Op::SetFillColor { col: black.clone() });
        // 標題
        for (c, title) in COLUMNS.iter().enumerate() {
            ops.push(at(xs[c], head_y));
            ops.push(Op::ShowText {
                items: vec![TextItem::Text((*title).to_string())],
            });
        }
        // 內容
        for (i, r) in chunk.iter().enumerate() {
            let y = first_y - i as f32 * PDF_ROW_H;
            let cells = [
                fmt_no(r.disk),
                r.path.clone(),
                r.name.clone(),
                fmt_gb(r.bytes),
                fmt_time(r.mtime),
                r.bytes.to_string(),
            ];
            for (c, cell) in cells.iter().enumerate() {
                // 放不下就截掉。路徑**從左邊截**（留最後面那幾層）：
                // 要認出是哪一個資料夾，靠的是末端那幾段，不是磁碟機代號
                let text = fit(&metrics, cell, PDF_COLS[c] - 2.0, c == 1);
                ops.push(at(xs[c], y));
                ops.push(Op::ShowText {
                    items: vec![TextItem::Text(text)],
                });
            }
        }
        // 頁尾：第幾頁、共幾頁、什麼時候匯出的
        ops.push(Op::SetFillColor { col: gray.clone() });
        ops.push(at(PDF_MARGIN, PDF_MARGIN));
        ops.push(Op::ShowText {
            items: vec![TextItem::Text(format!(
                "硬碟總表　共 {} 列　匯出時間 {stamp}　第 {} / {total_pages} 頁",
                rows.len(),
                p + 1
            ))],
        });
        ops.push(Op::EndTextSection);
        pages.push(PdfPage::new(Mm(PDF_W), Mm(PDF_H), ops));
    }
    if pages.is_empty() {
        pages.push(PdfPage::new(Mm(PDF_W), Mm(PDF_H), Vec::new()));
    }
    let bytes = doc
        .with_pages(pages)
        .save(&PdfSaveOptions::default(), &mut Vec::new());
    std::fs::write(path, bytes).map_err(|e| format!("存不進「{}」：{e}", path.display()))
}

/// 現在的 unix 秒數
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 這串字在 `pt` 字級下有多寬（mm）。
///
/// 查字型自己的字寬表；查不到的字（字型沒有那個字）當成一個全形寬——
/// 寧可算寬一點、把字截短，也不要排版超出欄位去壓到隔壁
fn pdf_text_mm(font: &ab_glyph::FontRef, s: &str, pt: f32) -> f32 {
    use ab_glyph::Font;
    let upem = font.units_per_em().unwrap_or(1000.0);
    let units: f32 = s
        .chars()
        .map(|ch| {
            let g = font.glyph_id(ch);
            // 字型沒有那個字（glyph 0）就當成一個全形寬：寧可算寬一點把字
            // 截短，也不要排版超出欄位去壓到隔壁
            if g.0 == 0 {
                upem
            } else {
                font.h_advance_unscaled(g)
            }
        })
        .sum();
    units / upem * pt / MM_PT
}

/// 把一格的字截到放得進 `limit` mm。`keep_tail`＝從左邊截、留後面那一段
/// （路徑用），否則從右邊截
fn fit(font: &ab_glyph::FontRef, s: &str, limit: f32, keep_tail: bool) -> String {
    if pdf_text_mm(font, s, PDF_FONT_PT) <= limit {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let ell = '…';
    let ell_w = pdf_text_mm(font, "…", PDF_FONT_PT);
    let budget = (limit - ell_w).max(0.0);
    let mut out: Vec<char> = Vec::new();
    let mut w = 0.0;
    if keep_tail {
        for &ch in chars.iter().rev() {
            let cw = pdf_text_mm(font, &ch.to_string(), PDF_FONT_PT);
            if w + cw > budget {
                break;
            }
            w += cw;
            out.push(ch);
        }
        out.reverse();
        std::iter::once(ell).chain(out).collect()
    } else {
        for &ch in &chars {
            let cw = pdf_text_mm(font, &ch.to_string(), PDF_FONT_PT);
            if w + cw > budget {
                break;
            }
            w += cw;
            out.push(ch);
        }
        out.into_iter().chain(std::iter::once(ell)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 匯出的_excel_與_pdf_都寫得出來() {
        let dir = std::env::temp_dir().join(format!("p2v_disk_out_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let rows = vec![
            Row {
                disk: 1,
                path: "H:\\手冊封面\\使用手冊封面\\手冊封面_JPG\\=選課系統".into(),
                name: "=選課系統".into(),
                bytes: 3_912_258,
                mtime: 1_671_069_191,
            },
            Row {
                disk: 12,
                path: "I:\\yellow2023掃瞄照片\\2023\\照片_ok\\_縮小".into(),
                name: "_縮小".into(),
                bytes: 543_404_388,
                mtime: 1_671_069_190,
            },
        ];
        let xlsx = dir.join("t.xlsx");
        export_xlsx(&rows, &xlsx).unwrap();
        assert!(fs::metadata(&xlsx).unwrap().len() > 0);

        // 中文字型是跟系統借的：借不到就只驗 Excel，不讓測試在沒有字型的
        // 機器上紅掉（CI 的 Linux 機器就可能沒有）
        if let Some(font) = crate::load_cjk_font_bytes() {
            let pdf = dir.join("t.pdf");
            export_pdf(&rows, &pdf, &font).unwrap();
            let head = fs::read(&pdf).unwrap();
            assert!(head.starts_with(b"%PDF"), "寫出來的不是 PDF");
            assert!(head.len() > 1000, "PDF 小得不像有內容");
            // 字型要**只嵌用到的那幾個字**。整份嵌進去的話，兩列的 PDF 就
            // 13MB——printpdf 的子集化只在 text_layout 功能底下才有，
            // 這一條守的就是 Cargo.toml 裡那個功能別被拿掉
            assert!(
                head.len() < 2_000_000,
                "PDF 有 {} 位元組，像是把整份中文字型都嵌進去了",
                head.len()
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn gb_照舊工具的寫法() {
        // 實照：543404388 → 0.51 GB、365723314 → 0.34 GB、325671119 → 0.3 GB
        assert_eq!(fmt_gb(543_404_388), "0.51 GB");
        assert_eq!(fmt_gb(365_723_314), "0.34 GB");
        assert_eq!(fmt_gb(325_671_119), "0.3 GB");
        // 不到 0.005 GB 的一律 0 GB（不是 0.00 GB）
        assert_eq!(fmt_gb(3_544_886), "0 GB");
        assert_eq!(fmt_gb(0), "0 GB");
        // 整數也不留小數點
        assert_eq!(fmt_gb(2 * 1024 * 1024 * 1024), "2 GB");
    }

    #[test]
    fn 空號要補回去() {
        let mut t = Table::default();
        assert_eq!(t.next_no(), 1, "空表從 1 號開始");
        let row = |d: u32| Row {
            disk: d,
            path: String::new(),
            name: String::new(),
            bytes: 0,
            mtime: 0,
        };
        t.rows = vec![row(1), row(2), row(4)];
        assert_eq!(t.next_no(), 3, "中間空出來的 3 號要補回去");
        t.rows.push(row(3));
        assert_eq!(t.next_no(), 5, "沒有空號就接在最大號後面");
    }

    #[test]
    fn 同一顆碟再掃一次是換掉不是疊上去() {
        let mut t = Table::default();
        let row = |d: u32, n: &str| Row {
            disk: d,
            path: format!("H:\\{n}"),
            name: n.into(),
            bytes: 0,
            mtime: 0,
        };
        let note = DiskNote::default();
        t.replace_disk(1, vec![row(1, "舊的"), row(1, "兩邊都有")], note.clone());
        t.replace_disk(2, vec![row(2, "別顆碟的")], note.clone());
        t.replace_disk(1, vec![row(1, "兩邊都有"), row(1, "新的")], note);
        let names: Vec<&str> = t.rows.iter().map(|r| r.name.as_str()).collect();
        assert!(!names.contains(&"舊的"), "被刪掉的資料夾不該留在表上");
        assert!(names.contains(&"新的"));
        assert_eq!(t.count_of(1), 2);
        assert_eq!(t.count_of(2), 1, "別顆碟的資料不能被動到");
    }

    #[test]
    fn 掃描停在指定的層數() {
        let base = std::env::temp_dir().join(format!("p2v_disk_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        // base/a/b/c，各層放一個 1 位元組的檔案
        let deep = base.join("a").join("b").join("c");
        fs::create_dir_all(&deep).unwrap();
        for d in [base.as_path(), &base.join("a"), &base.join("a/b"), &deep] {
            fs::write(d.join("x.bin"), [0u8]).unwrap();
        }
        let cancel = AtomicBool::new(false);
        let out = scan(&base, 2, 7, &cancel, &mut |_, _| {}).unwrap();
        let names: Vec<&str> = out.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"], "第 3 層的 c 不該被走到");
        assert!(out.rows.iter().all(|r| r.disk == 7));
        // 每一列的大小只算自己那一層的檔案
        assert!(out.rows.iter().all(|r| r.bytes == 1));
        let _ = fs::remove_dir_all(&base);
    }
}
