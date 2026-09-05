//! 檔案管理 ▸ 資料備份：比對來源與目的兩個資料夾。
//!
//! 規則就三條，和使用者在畫面上讀到的一模一樣：
//!
//! 1. 同名的檔案，**來源比較新就覆蓋掉目的那一份**。
//! 2. 目的沒有的檔案直接拷過去。
//! 3. 目的地多餘的檔案（來源沒有的），勾了「保留」就不動，沒勾就刪掉。
//!
//! 寫進去的位置是**目的資料夾底下、與來源同名的那一個**（見 [`target`]），
//! 不存在就建起來。
//!
//! **只往目的資料夾寫**：目的那份比較新時就原封不動留著，不會反過來動到
//! 來源。來源資料夾在整個過程中都是唯讀的。
//!
//! 這裡只負責「算出要做哪些事」與「做一件事」；進度、取消與畫面在 main.rs，
//! 與其他模組同一套作法。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

/// 兩邊的修改時間差多少以內算「同一份」。
///
/// FAT32／exFAT 的時間戳只有 2 秒解析度，同一個檔案拷到隨身碟再拷回來就會
/// 差個 1 秒。不放寬的話每次備份都會把整批檔案再抄一次，備份永遠跑不完
const MTIME_SLACK: Duration = Duration::from_secs(2);

/// 使用者按了中止時掃描回報的訊息（main.rs 靠它分辨「中止」與「出錯」）
pub const CANCELLED: &str = "已中止";

/// 一個檔案要做的事
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// 目的沒有這個檔：從來源拷過去
    Copy,
    /// 兩邊都有、來源比較新：覆蓋掉目的那一份
    Update,
    /// 只有目的有：勾了「保留」就不動，沒勾就刪掉
    Extra,
}

impl Kind {
    /// 清單上那一欄的字
    pub fn label(self) -> &'static str {
        match self {
            Kind::Copy => "新增",
            Kind::Update => "更新",
            Kind::Extra => "目的地多餘檔案",
        }
    }

    /// 清單的排序順序。HashMap 出來的順序是亂的，不固定下來的話
    /// 同一組資料夾比對兩次，清單的排法會不一樣
    fn order(self) -> u8 {
        match self {
            Kind::Copy => 0,
            Kind::Update => 1,
            Kind::Extra => 2,
        }
    }
}

/// 要對某一個檔案做的一件事
#[derive(Clone)]
pub struct Action {
    /// 相對於資料夾根的路徑（來源與目的共用這一段）
    pub rel: PathBuf,
    pub kind: Kind,
    /// 這件事要搬（或要刪）多少位元組，顯示總量用
    pub bytes: u64,
}

/// 比對的結果：要動的都在 `actions` 裡，沒動的只留一個數字。
///
/// 不把「兩邊一樣」的那幾萬個檔案也收進清單：使用者不會去看，
/// 但十萬筆 PathBuf 是實打實的幾十 MB
#[derive(Default)]
pub struct Plan {
    pub actions: Vec<Action>,
    /// 兩邊一模一樣、不必動的檔案數
    pub same: usize,
    /// 目的那份比較新、因此原封不動留著的檔案數。
    ///
    /// 只往目的寫，所以這幾個不是「要做的事」——但也不是「相同」，
    /// 不另外算一份的話，畫面上的數字會兜不起來
    pub newer_dst: usize,
    /// 掃描時讀不到、只好跳過的資料夾（權限不足、被別的程式鎖住之類）
    pub skipped: Vec<String>,
}

impl Plan {
    /// 某一種事有幾件
    pub fn count(&self, kind: Kind) -> usize {
        self.actions.iter().filter(|a| a.kind == kind).count()
    }

    /// 某一種事總共多少位元組
    pub fn bytes(&self, kind: Kind) -> u64 {
        self.actions
            .iter()
            .filter(|a| a.kind == kind)
            .map(|a| a.bytes)
            .sum()
    }

    /// 這份比對結果裡，真的會動到檔案的件數。
    ///
    /// 勾了「保留目的資料夾多餘的檔案」時，那幾件只是列出來給人看的，
    /// 一個檔案都不會動——所以不能算進來（算了的話，「目的地多餘檔案 82、
    /// 相同 82」這種一件事都不用做的狀況，按鈕還會是亮的）
    pub fn todo(&self, keep_extra: bool) -> usize {
        if keep_extra {
            self.actions.len() - self.count(Kind::Extra)
        } else {
            self.actions.len()
        }
    }
}

/// 檔案大小的人話。備份動輒好幾萬個檔案，位元組數字看不出多少
pub fn fmt_size(bytes: u64) -> String {
    const K: f64 = 1024.0;
    let b = bytes as f64;
    if b < K {
        format!("{bytes} B")
    } else if b < K * K {
        format!("{:.0} KB", b / K)
    } else if b < K * K * K {
        format!("{:.1} MB", b / (K * K))
    } else {
        format!("{:.2} GB", b / (K * K * K))
    }
}

/// 實際要寫進去的資料夾。
///
/// 「備份 `F:\2025_鳥_精選` 到 `O:\`」講的是 `O:\2025_鳥_精選`，不是把照片
/// 倒在 `O:\` 的根目錄——同一顆備份碟才放得下好幾組資料夾。所以挑的目的
/// 資料夾底下會再套一層與來源同名的資料夾（不存在就建起來）。
///
/// **但挑的那個自己就叫這個名字時就用它**：使用者已經直接挑到
/// `O:\2025_鳥_精選` 了，再套一層會變成 `O:\2025_鳥_精選\2025_鳥_精選`。
///
/// 來源是磁碟機根目錄（沒有名字）時也直接寫進去
pub fn target(src: &Path, dst: &Path) -> PathBuf {
    let Some(name) = src.file_name() else {
        return dst.to_path_buf();
    };
    if dst.file_name().is_some_and(|d| same_name(d, name)) {
        return dst.to_path_buf();
    }
    dst.join(name)
}

/// 兩個資料夾名稱算不算同一個（Windows 的檔名不分大小寫）
fn same_name(a: &std::ffi::OsStr, b: &std::ffi::OsStr) -> bool {
    if cfg!(windows) {
        a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

/// 這兩個資料夾能不能拿來備份（`dst` 傳的是 [`target`] 算出來的那一個）。
///
/// 一個在另一個裡面一定要擋：目的在來源底下時，剛拷進去的東西下一輪又被
/// 當成來源的一部分，會一層一層拷到磁碟滿為止
pub fn validate(src: &Path, dst: &Path) -> Result<(), String> {
    if !src.is_dir() {
        return Err("來源資料夾不存在（可能已被刪除、改名，或在拔掉的隨身碟上）".into());
    }
    // 目的還不存在沒關係——第一次備份就會把它建出來。但它的上一層一定要在，
    // 不然就是隨身碟被拔掉或那個位置已經被刪掉了
    if !dst.is_dir() {
        let parent_ok = dst.parent().is_some_and(|p| p.is_dir());
        if !parent_ok || dst.exists() {
            return Err("目的資料夾不存在（可能已被刪除、改名，或在拔掉的隨身碟上）".into());
        }
    }
    let s = norm(src);
    let d = norm(dst);
    if s == d {
        return Err("來源與目的是同一個資料夾".into());
    }
    if d.starts_with(&s) {
        return Err("目的資料夾在來源資料夾裡面，會一層一層拷貝下去".into());
    }
    if s.starts_with(&d) {
        return Err("來源資料夾在目的資料夾裡面，會一層一層拷貝下去".into());
    }
    Ok(())
}

/// 比對路徑用的正規化形式：解到真實路徑（跟隨替身與捷徑），Windows 再壓成
/// 小寫（檔名不分大小寫）。
///
/// 路徑可能**還不存在**（目的資料夾等著被建起來），這時往上找到第一個解得開
/// 的祖先再把剩下的段接回去——直接拿原樣的路徑比會變成「一邊是
/// `\\?\F:\照片`、一邊是 `O:\照片`」，包含關係就檢查不出來了
fn norm(p: &Path) -> PathBuf {
    let lower = |path: PathBuf| {
        if cfg!(windows) {
            PathBuf::from(path.to_string_lossy().to_lowercase())
        } else {
            path
        }
    };
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = p.to_path_buf();
    loop {
        if let Ok(c) = fs::canonicalize(&cur) {
            let mut out = c;
            for seg in rest.iter().rev() {
                out.push(seg);
            }
            return lower(out);
        }
        let Some(name) = cur.file_name().map(|n| n.to_os_string()) else {
            return lower(p.to_path_buf());
        };
        rest.push(name);
        if !cur.pop() {
            return lower(p.to_path_buf());
        }
    }
}

/// 掃到的一個檔案
struct Found {
    /// 相對於資料夾根的路徑（原樣，寫檔時要用）
    rel: PathBuf,
    len: u64,
    mtime: SystemTime,
}

/// 兩邊配對用的鍵。
///
/// Windows 的檔名不分大小寫，同一個檔案只是大小寫寫法不同，不該被當成
/// 「來源多一個、目的多一個」——那會多拷一次，沒勾保留時還會把另一份刪掉
fn key_of(rel: &Path) -> String {
    let s = rel.to_string_lossy().replace('\\', "/");
    if cfg!(windows) {
        s.to_lowercase()
    } else {
        s
    }
}

/// 掃出一個資料夾底下的所有檔案。
///
/// 回傳（配對鍵 → 檔案, 讀不到而跳過的資料夾）。根資料夾讀不到就直接失敗——
/// 那是選錯地方或沒有權限，繼續跑只會得到一份「什麼都沒有」的假結果，
/// 而「來源什麼都沒有」在沒勾保留時等於把目的整個清空
fn scan(
    root: &Path,
    recursive: bool,
    cancel: &AtomicBool,
) -> Result<(HashMap<String, Found>, Vec<String>), String> {
    let mut out: HashMap<String, Found> = HashMap::new();
    let mut skipped: Vec<String> = Vec::new();
    // 待掃的子資料夾（相對路徑；空的那個＝根本身）
    let mut dirs: Vec<PathBuf> = vec![PathBuf::new()];
    let mut is_root = true;
    while let Some(rel_dir) = dirs.pop() {
        if cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        let dir = root.join(&rel_dir);
        let rd = match fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) => {
                if is_root {
                    return Err(format!("無法讀取「{}」：{e}", dir.display()));
                }
                skipped.push(format!("{}（{e}）", dir.display()));
                continue;
            }
        };
        is_root = false;
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            // 捷徑（符號連結／junction）不跟進去：它可能指回自己形成無窮迴圈，
            // 連結後面那份資料也不屬於這個資料夾，不該被備份走
            if ft.is_symlink() {
                continue;
            }
            let rel = rel_dir.join(e.file_name());
            if ft.is_dir() {
                if recursive {
                    dirs.push(rel);
                }
            } else if ft.is_file() {
                let Ok(md) = e.metadata() else { continue };
                out.insert(
                    key_of(&rel),
                    Found {
                        rel,
                        len: md.len(),
                        mtime: md.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                    },
                );
            }
        }
    }
    Ok((out, skipped))
}

/// `a` 是不是比 `b` 新。差在 [`MTIME_SLACK`] 以內回 `None`（當成一樣新）
fn newer(a: SystemTime, b: SystemTime) -> Option<bool> {
    match a.duration_since(b) {
        Ok(d) if d > MTIME_SLACK => Some(true),
        Ok(_) => None,
        Err(e) if e.duration() > MTIME_SLACK => Some(false),
        Err(_) => None,
    }
}

/// 比對兩個資料夾，算出要做哪些事（這一步不動到任何檔案）
pub fn plan(src: &Path, dst: &Path, recursive: bool, cancel: &AtomicBool) -> Result<Plan, String> {
    validate(src, dst)?;
    let (a, mut skipped) = scan(src, recursive, cancel)?;
    // 目的資料夾還沒建起來：當成空的，整批都是「新增」
    let (b, skipped_dst) = if dst.is_dir() {
        scan(dst, recursive, cancel)?
    } else {
        (HashMap::new(), Vec::new())
    };
    skipped.extend(skipped_dst);

    let mut actions: Vec<Action> = Vec::new();
    let mut same = 0usize;
    let mut newer_dst = 0usize;
    for (k, s) in &a {
        match b.get(k) {
            // 目的沒有：直接拷過去
            None => actions.push(Action {
                rel: s.rel.clone(),
                kind: Kind::Copy,
                bytes: s.len,
            }),
            Some(d) => match newer(s.mtime, d.mtime) {
                Some(true) => actions.push(Action {
                    rel: s.rel.clone(),
                    kind: Kind::Update,
                    bytes: s.len,
                }),
                // 目的那份比較新：只往目的寫，所以原封不動留著
                Some(false) => newer_dst += 1,
                // 一樣新、也一樣大：同一份，不必動
                None if s.len == d.len => same += 1,
                // 一樣新卻不一樣大：其中一份是上次拷到一半留下的。
                // 時間分不出勝負，備份就以來源那份為準補一次
                None => actions.push(Action {
                    rel: s.rel.clone(),
                    kind: Kind::Update,
                    bytes: s.len,
                }),
            },
        }
    }
    for (k, d) in &b {
        if !a.contains_key(k) {
            actions.push(Action {
                rel: d.rel.clone(),
                kind: Kind::Extra,
                bytes: d.len,
            });
        }
    }
    actions.sort_by(|x, y| {
        x.kind
            .order()
            .cmp(&y.kind.order())
            .then_with(|| x.rel.cmp(&y.rel))
    });
    Ok(Plan {
        actions,
        same,
        newer_dst,
        skipped,
    })
}

/// 做掉一件事。錯誤訊息裡帶相對路徑，整批跑完才一起顯示。
///
/// 寫的一律是目的那一邊——`src_root` 只拿來讀
pub fn apply(
    act: &Action,
    src_root: &Path,
    dst_root: &Path,
    keep_extra: bool,
) -> Result<(), String> {
    let in_src = src_root.join(&act.rel);
    let in_dst = dst_root.join(&act.rel);
    match act.kind {
        Kind::Copy | Kind::Update => copy_file(&in_src, &in_dst, &act.rel),
        Kind::Extra => {
            if keep_extra {
                return Ok(());
            }
            fs::remove_file(&in_dst).map_err(|e| format!("{}：刪除失敗（{e}）", act.rel.display()))
        }
    }
}

fn copy_file(from: &Path, to: &Path, rel: &Path) -> Result<(), String> {
    if let Some(dir) = to.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}：無法建立資料夾（{e}）", rel.display()))?;
    }
    // 舊檔帶唯讀屬性時 fs::copy 會失敗（從光碟或記憶卡拷出來的照片常帶著）。
    // 要覆蓋的既然是舊的那一份，先把旗標拿掉再蓋
    if let Ok(md) = fs::metadata(to) {
        let mut perm = md.permissions();
        if perm.readonly() {
            perm.set_readonly(false);
            let _ = fs::set_permissions(to, perm);
        }
    }
    fs::copy(from, to).map_err(|e| format!("{}：複製失敗（{e}）", rel.display()))?;
    // 修改時間一起帶過去，下次比對才認得出這兩份是同一個——不帶的話新拷的
    // 那份永遠比較新，每次備份都要再抄一遍。Windows
    // 的 CopyFileEx 本來就會帶，其他平台不會，統一補一次。補不成不算失敗，
    // 檔案內容已經對了
    if let Ok(t) = fs::metadata(from).and_then(|md| md.modified()) {
        if let Ok(f) = fs::File::options().write(true).open(to) {
            let _ = f.set_modified(t);
        }
    }
    Ok(())
}

/// 刪完多餘的檔案後，把目的資料夾裡「來源沒有、而且已經空了」的資料夾也
/// 收掉。只刪檔案會留下一地空殼，看起來像沒清乾淨
pub fn prune_empty_dirs(src_root: &Path, dst_root: &Path, cancel: &AtomicBool) {
    // 先把所有子資料夾收齊，再由深到淺刪——巢狀的空殼才會一層一層收掉
    let mut all: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![PathBuf::new()];
    while let Some(rel) = stack.pop() {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let Ok(rd) = fs::read_dir(dst_root.join(&rel)) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() && !ft.is_symlink() {
                let child = rel.join(e.file_name());
                stack.push(child.clone());
                all.push(child);
            }
        }
    }
    all.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for rel in all {
        // 來源也有這個資料夾就留著：它只是這次剛好沒東西，不是多出來的
        if src_root.join(&rel).is_dir() {
            continue;
        }
        // 裡面還有東西的話 remove_dir 自己會擋下來，失敗不必回報
        let _ = fs::remove_dir(dst_root.join(&rel));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// 在暫存區開一個乾淨的測試資料夾
    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("p2v_backup_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 寫一個檔案並指定修改時間（`secs_ago` 是相對於現在往前推幾秒）
    fn write_at(path: &Path, body: &str, secs_ago: u64) {
        if let Some(d) = path.parent() {
            fs::create_dir_all(d).unwrap();
        }
        let f = fs::File::create(path).unwrap();
        (&f).write_all(body.as_bytes()).unwrap();
        f.set_modified(SystemTime::now() - Duration::from_secs(secs_ago))
            .unwrap();
    }

    fn kinds(plan: &Plan) -> HashMap<String, Kind> {
        plan.actions
            .iter()
            .map(|a| (a.rel.to_string_lossy().replace('\\', "/"), a.kind))
            .collect()
    }

    #[test]
    fn only_the_destination_is_ever_written() {
        // 來源新就蓋過去、沒有的直接拷、目的多出來的另外列；
        // **目的比較新的那一份原封不動**——來源資料夾一個字都不會被改到
        let root = tmp("plan");
        let src = root.join("src");
        let dst = root.join("dst");
        write_at(&src.join("來源比較新.txt"), "new", 0);
        write_at(&dst.join("來源比較新.txt"), "old", 600);
        write_at(&src.join("目的比較新.txt"), "舊的留著", 600);
        write_at(&dst.join("目的比較新.txt"), "new", 0);
        write_at(&src.join("只有來源有.txt"), "a", 0);
        write_at(&dst.join("只有目的有.txt"), "b", 0);
        // 同時間、同內容＝同一份，不該被排進待辦
        write_at(&src.join("兩邊一樣.txt"), "same", 300);
        write_at(&dst.join("兩邊一樣.txt"), "same", 300);

        let p = plan(&src, &dst, true, &AtomicBool::new(false)).unwrap();
        let got = kinds(&p);
        assert_eq!(got.get("來源比較新.txt"), Some(&Kind::Update));
        assert_eq!(got.get("只有來源有.txt"), Some(&Kind::Copy));
        assert_eq!(got.get("只有目的有.txt"), Some(&Kind::Extra));
        assert_eq!(got.get("目的比較新.txt"), None, "目的比較新的不能排進待辦");
        assert_eq!(got.len(), 3, "兩邊一樣的那個也不該被排進待辦");
        assert_eq!(p.same, 1);
        assert_eq!(p.newer_dst, 1, "目的比較新的要另外算一份給畫面顯示");

        // 整批做完，來源那一份還是原來的內容
        for a in &p.actions {
            apply(a, &src, &dst, false).unwrap();
        }
        assert_eq!(
            fs::read_to_string(src.join("目的比較新.txt")).unwrap(),
            "舊的留著",
            "不能反過來動到來源"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn copy_keeps_mtime_so_the_next_run_has_nothing_to_do() {
        // 拷過去的那份若帶著「現在」的時間，下一輪比對它永遠比來源新，
        // 於是每一批都被算成「目的比較新」而整批跳過，備份看起來沒在做事。
        // 跑第二次應該是「一件都不用做」，而且是因為兩邊真的一樣
        let root = tmp("mtime");
        let src = root.join("src");
        let dst = root.join("dst");
        write_at(&src.join("子資料夾/照片.jpg"), "xxxx", 3600);
        fs::create_dir_all(&dst).unwrap();

        let cancel = AtomicBool::new(false);
        let p = plan(&src, &dst, true, &cancel).unwrap();
        assert_eq!(p.actions.len(), 1);
        apply(&p.actions[0], &src, &dst, true).unwrap();
        assert_eq!(
            fs::read_to_string(dst.join("子資料夾/照片.jpg")).unwrap(),
            "xxxx",
            "子資料夾要跟著建起來"
        );

        let p2 = plan(&src, &dst, true, &cancel).unwrap();
        assert!(p2.actions.is_empty(), "第二次應該一件都不用做");
        assert_eq!(p2.same, 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn extra_files_are_only_deleted_when_keep_is_off() {
        // 「保留目的資料夾多餘的檔案」是使用者唯一會踩到刪除的地方，
        // 勾著的時候一個檔案都不能少
        let root = tmp("extra");
        let src = root.join("src");
        let dst = root.join("dst");
        fs::create_dir_all(&src).unwrap();
        write_at(&dst.join("多出來的.txt"), "x", 0);
        let act = Action {
            rel: PathBuf::from("多出來的.txt"),
            kind: Kind::Extra,
            bytes: 1,
        };

        apply(&act, &src, &dst, true).unwrap();
        assert!(dst.join("多出來的.txt").exists(), "勾了保留就不能刪");

        apply(&act, &src, &dst, false).unwrap();
        assert!(!dst.join("多出來的.txt").exists(), "沒勾保留才刪掉");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_destination_is_created_under_the_picked_folder() {
        // 「備份 F:\2025_鳥_精選 到 O:\」講的是 O:\2025_鳥_精選。那個資料夾
        // 還不存在時不能當成錯誤擋下來——整批都是「新增」，做的時候建起來
        let root = tmp("target");
        let src = root.join("2025_鳥_精選");
        let picked = root.join("備份碟");
        write_at(&src.join("IMG_001.jpg"), "aaa", 0);
        fs::create_dir_all(&picked).unwrap();

        let dst = target(&src, &picked);
        assert_eq!(dst, picked.join("2025_鳥_精選"));
        assert!(!dst.exists(), "測試前提：那個資料夾還不存在");
        validate(&src, &dst).expect("還不存在的目的資料夾不該被擋下來");
        // 使用者直接挑到那一個時就用它，不能再往下多套一層同名資料夾
        assert_eq!(
            target(&src, &dst),
            dst,
            "挑的目的資料夾自己就叫這個名字，不該變成 …\\2025_鳥_精選\\2025_鳥_精選"
        );

        let p = plan(&src, &dst, true, &AtomicBool::new(false)).unwrap();
        assert_eq!(kinds(&p).get("IMG_001.jpg"), Some(&Kind::Copy));
        apply(&p.actions[0], &src, &dst, true).unwrap();
        assert_eq!(
            fs::read_to_string(dst.join("IMG_001.jpg")).unwrap(),
            "aaa",
            "資料夾要跟著建出來"
        );

        // 挑的位置本身不見了（隨身碟被拔掉）就要擋
        let _ = fs::remove_dir_all(&picked);
        assert!(validate(&src, &dst).is_err(), "上一層都不在了要擋");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn kept_extras_do_not_count_as_work_to_do() {
        // 「目的地多餘檔案 82（保留）、相同 82」＝一個檔案都不會動。
        // 這時「開始備份」要是暗的，按了才不會是比一次然後什麼也沒發生
        let root = tmp("todo");
        let src = root.join("src");
        let dst = root.join("dst");
        write_at(&src.join("兩邊都有.txt"), "x", 300);
        write_at(&dst.join("兩邊都有.txt"), "x", 300);
        write_at(&dst.join("只有目的有.txt"), "y", 0);

        let p = plan(&src, &dst, true, &AtomicBool::new(false)).unwrap();
        assert_eq!(p.count(Kind::Extra), 1);
        assert_eq!(p.same, 1);
        assert_eq!(p.todo(true), 0, "勾了保留就一件事都不用做");
        assert_eq!(p.todo(false), 1, "沒勾保留就要刪那一個");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn nested_folders_are_rejected_before_anything_is_copied() {
        // 目的在來源底下時，拷進去的東西下一輪又被當成來源的內容，
        // 會一路拷到磁碟滿。要在開始之前就擋掉
        let root = tmp("nest");
        let src = root.join("src");
        let inner = src.join("備份");
        fs::create_dir_all(&inner).unwrap();
        assert!(validate(&src, &inner).is_err(), "目的在來源裡面要擋");
        assert!(validate(&inner, &src).is_err(), "來源在目的裡面要擋");
        assert!(validate(&src, &src).is_err(), "同一個資料夾要擋");
        // 目的挑到來源的上一層時，實際寫入的位置就是來源自己——還沒建立的
        // 路徑一樣要算得出包含關係，不然這一條會漏掉
        assert!(
            validate(&src, &target(&src, &root)).is_err(),
            "目的算出來等於來源自己要擋"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn subfolders_are_skipped_when_recursion_is_off() {
        // 「含子資料夾」關掉時只比對這一層：子資料夾裡的東西既不拷過去，
        // 也不能被當成「目的多出來的」而刪掉
        let root = tmp("flat");
        let src = root.join("src");
        let dst = root.join("dst");
        write_at(&src.join("這一層.txt"), "a", 0);
        write_at(&src.join("裡面/深的.txt"), "b", 0);
        write_at(&dst.join("裡面/深的.txt"), "b", 0);

        let p = plan(&src, &dst, false, &AtomicBool::new(false)).unwrap();
        let got = kinds(&p);
        assert_eq!(got.get("這一層.txt"), Some(&Kind::Copy));
        assert_eq!(got.len(), 1, "只該看這一層");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pruning_only_removes_empty_folders_the_source_does_not_have() {
        // 刪掉多餘檔案後留下的空殼要收掉，但來源也有的那個資料夾要留著
        // ——它只是這次剛好沒東西，不是多出來的
        let root = tmp("prune");
        let src = root.join("src");
        let dst = root.join("dst");
        fs::create_dir_all(src.join("兩邊都有")).unwrap();
        fs::create_dir_all(dst.join("兩邊都有")).unwrap();
        fs::create_dir_all(dst.join("多出來的/更裡面")).unwrap();
        write_at(&dst.join("還有東西/檔案.txt"), "x", 0);

        prune_empty_dirs(&src, &dst, &AtomicBool::new(false));
        assert!(dst.join("兩邊都有").is_dir(), "來源也有的空資料夾要留著");
        assert!(!dst.join("多出來的").exists(), "巢狀的空殼要一路收掉");
        assert!(dst.join("還有東西").is_dir(), "裡面有檔案的不能刪");

        let _ = fs::remove_dir_all(&root);
    }
}
