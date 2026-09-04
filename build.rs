fn main() {
    // 把 assets/icon.ico 嵌入 Windows 執行檔（檔案總管與工作列圖示）
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        // 版本資訊字串。沒簽章的執行檔如果 CompanyName、LegalCopyright、
        // OriginalFilename 都空白，NOD32 這類啟發式掃描會把它當成可疑檔加權，
        // 拷到別台電腦常被誤判成病毒。winresource 只會自動帶 ProductName、
        // FileDescription 與版號，其餘在這裡補齊
        res.set("CompanyName", "Yellow Huang");
        res.set("FileDescription", "photo2video 照片轉影片、去煙霧與夜空後製工具");
        res.set("LegalCopyright", "Copyright © 2026 Yellow Huang");
        res.set("OriginalFilename", "photo2video.exe");
        res.set("InternalName", "photo2video");
        res.compile().expect("嵌入應用程式圖示失敗");
    }
    println!("cargo:rerun-if-changed=assets/icon.ico");
}
