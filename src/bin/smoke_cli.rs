//! 去煙演算法的參數實驗工具：
//! cargo run --release --bin smoke_cli -- <輸入> <輸出> [strength] [detail] [x0,y0,x1,y1]
//! 最後一個參數為只去煙的矩形區塊（0~1 相對座標），省略則整張處理。
//! 兩種漸層遮色片另以環境變數指定（可與矩形疊用，取聯集）：
//!   SMOKE_LINEAR=x0,y0,x1,y1      起點全效果 → 終點歸零
//!   SMOKE_RADIAL=cx,cy,rx,ry[,1]  橢圓內去煙，第五個值非 0 則反轉
//! 筆刷只有 GUI 畫得出來，這裡不提供。
//! SMOKE_AUTO=1 改用自動判出來的去除煙霧／細節／去除雲朵／範圍（GUI 開檔時做的就是這件事），
//! 想知道某張照片會被判成什麼值，跑這個最快。

#[path = "../dehaze.rs"]
mod dehaze;

fn tm_save(img: &image::RgbImage, out: &str) {
    let p = out.replace(".jpg", "_layer.jpg");
    img.save(&p).expect("寫出失敗");
    println!("煙霧層 → {p}");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("用法：smoke_cli <輸入> <輸出> [strength] [detail]");
        std::process::exit(2);
    }
    /// 逗號分隔的浮點數清單，長度不符就當作沒給
    fn nums(s: &str, n: usize) -> Option<Vec<f32>> {
        let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        (v.len() == n).then_some(v)
    }
    let mut shapes: Vec<dehaze::Shape> = Vec::new();
    if let Some(v) = a.get(5).and_then(|s| nums(s, 4)) {
        shapes.push(dehaze::Shape::Rect(dehaze::Region {
            x0: v[0],
            y0: v[1],
            x1: v[2],
            y1: v[3],
        }));
    }
    // 兩種漸層遮色片：SMOKE_LINEAR=x0,y0,x1,y1（起點全效果→終點歸零）、
    // SMOKE_RADIAL=cx,cy,rx,ry[,1 反轉]
    if let Some(v) = std::env::var("SMOKE_LINEAR").ok().and_then(|s| nums(&s, 4)) {
        shapes.push(dehaze::Shape::Linear(dehaze::Linear {
            x0: v[0],
            y0: v[1],
            x1: v[2],
            y1: v[3],
        }));
    }
    if let Ok(s) = std::env::var("SMOKE_RADIAL") {
        if let Some(v) = nums(&s, 4).or_else(|| nums(&s, 5)) {
            shapes.push(dehaze::Shape::Radial(dehaze::Radial {
                cx: v[0],
                cy: v[1],
                rx: v[2],
                ry: v[3],
                invert: v.get(4).is_some_and(|f| *f != 0.0),
            }));
        }
    }
    let mut p = dehaze::SmokeParams {
        strength: a.get(3).and_then(|s| s.parse().ok()).unwrap_or(80),
        detail: a.get(4).and_then(|s| s.parse().ok()).unwrap_or(60),
        shapes,
        tolerance: std::env::var("SMOKE_TOL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30),
        ..Default::default()
    };
    // 保護色以環境變數帶入：SMOKE_PROTECT=r,g,b[;r,g,b…]（0~255）、SMOKE_TOL=容差
    if let Ok(s) = std::env::var("SMOKE_PROTECT") {
        for one in s.split(';') {
            let v: Vec<u8> = one
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            if v.len() == 3 {
                p.add_protect([v[0], v[1], v[2]]);
            }
        }
    }
    // 天空：SMOKE_CLEAN=清雲強度、SMOKE_RANGE=判定範圍、
    // SMOKE_SKY=r,g,b 夜空色、SMOKE_SKYTINT=上色強度
    // SMOKE_SKYONLY=0：關掉「只處理天空」，整張一視同仁（預設是開著的）
    if let Ok(v) = std::env::var("SMOKE_SKYONLY") {
        p.sky_only = v != "0";
    }
    // SMOKE_RESTORE=0：關掉「補回煙裡的軌跡」（預設是開著的）
    if let Ok(v) = std::env::var("SMOKE_RESTORE") {
        p.restore_trails = v != "0";
    }
    // SMOKE_FAST=1：影片模組的「速度優先」——煙霧層用比較小的解析度估
    if let Ok(v) = std::env::var("SMOKE_FAST") {
        p.fast = v != "0";
    }
    // SMOKE_PREVIEW_OF=原圖長邊：把手上這張當成那張原圖的縮圖來處理
    // （GUI 畫預覽時做的就是這件事，見 dehaze::SmokeParams::preview_of）。
    // 想確認「預覽與成品是不是同一個結果」，拿它跑最快
    if let Ok(v) = std::env::var("SMOKE_PREVIEW_OF") {
        p.preview_of = v.parse().ok();
    }
    // SMOKE_DENSITY=0~100：遮色片濃度（畫到的地方最多用到幾成）
    if let Ok(v) = std::env::var("SMOKE_DENSITY") {
        p.mask_density = v.parse().unwrap_or(80);
    }
    if let Ok(v) = std::env::var("SMOKE_CLEAN") {
        p.sky_clean = v.parse().unwrap_or(0);
    }
    if let Ok(v) = std::env::var("SMOKE_RANGE") {
        p.sky_range = v.parse().unwrap_or(40);
    }
    if let Ok(s) = std::env::var("SMOKE_SKY") {
        let v: Vec<u8> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 3 {
            p.sky_color = Some([v[0], v[1], v[2]]);
        }
    }
    if let Ok(v) = std::env::var("SMOKE_SKYTINT") {
        p.sky_tint = v.parse().unwrap_or(60);
    }
    // 雲色（吸管指名亮到判定不出來的雲）：
    // SMOKE_CLOUD=r,g,b[;r,g,b…]、SMOKE_CLOUDRANGE=相近範圍
    if let Ok(s) = std::env::var("SMOKE_CLOUD") {
        for one in s.split(';') {
            let v: Vec<u8> = one
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            if v.len() == 3 {
                p.add_cloud([v[0], v[1], v[2]]);
            }
        }
    }
    if let Ok(v) = std::env::var("SMOKE_CLOUDRANGE") {
        p.cloud_range = v.parse().unwrap_or(35);
    }
    let t0 = std::time::Instant::now();
    let img = image::open(&a[1]).expect("讀取失敗").to_rgb8();
    // SMOKE_AUTO=1：改用自動判出來的四個值（GUI 開檔時做的就是這件事）
    if std::env::var("SMOKE_AUTO").is_ok() {
        let ta = std::time::Instant::now();
        let auto = dehaze::auto_params(&img, img.width().max(img.height()));
        println!("自動判參數 {auto:?}（{:?}）", ta.elapsed());
        auto.apply_to(&mut p);
    }
    println!("輸入 {}x{}，參數 {:?}", img.width(), img.height(), p);
    if std::env::var("SMOKE_DEBUG").is_ok() {
        let (layer, air) = dehaze::debug_smoke_layer(&img, &p);
        println!("煙霧最濃處（線性 RGB）＝{air:?}");
        tm_save(&layer, &a[2]);
    }
    if std::env::var("SMOKE_STREAK").is_ok() {
        let m = dehaze::debug_streak_map(&img);
        let path = a[2].replace(".jpg", "_streak.jpg");
        m.save(&path).expect("寫出失敗");
        println!("線條密度判據 → {path}");
    }
    if std::env::var("SMOKE_REGION").is_ok() {
        let m = dehaze::debug_sky_region(&img, &p);
        let path = a[2].replace(".jpg", "_region.jpg");
        m.save(&path).expect("寫出失敗");
        println!("去煙作用範圍 → {path}");
    }
    if std::env::var("SMOKE_SEED").is_ok() {
        let m = dehaze::debug_sky_seed(&img, &p);
        let path = a[2].replace(".jpg", "_seed.jpg");
        m.save(&path).expect("寫出失敗");
        println!("天空種子 → {path}");
    }
    if std::env::var("SMOKE_SKYMASK").is_ok() {
        let m = dehaze::debug_sky_mask(&img, &p);
        let path = a[2].replace(".jpg", "_sky.jpg");
        m.save(&path).expect("寫出失敗");
        println!("天空遮罩 → {path}");
    }
    let out = if std::env::var("SMOKE_MASK").is_ok() {
        // 遮色片檢視：紅色蓋住的地方不會被去煙
        dehaze::mask_overlay(&img, &p)
    } else {
        dehaze::remove_smoke(&img, &p)
    };
    out.save(&a[2]).expect("寫出失敗");
    println!("完成 → {} （{:?}）", a[2], t0.elapsed());

    // 局部檢視：SMOKE_CROP=x,y,w,h 把處理前後的同一塊剪出來另存，
    // 縮圖看不出來的殘留與硬邊，剪一塊原尺寸的下來才看得準
    if let Ok(s) = std::env::var("SMOKE_CROP") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 4 {
            let (x, y) = (v[0].min(img.width() - 1), v[1].min(img.height() - 1));
            let (w, h) = (v[2].min(img.width() - x), v[3].min(img.height() - y));
            for (label, src) in [("before", &img), ("after", &out)] {
                let c = image::imageops::crop_imm(src, x, y, w, h).to_image();
                let path = a[2].replace(".jpg", &format!("_{label}.jpg"));
                c.save(&path).expect("寫出失敗");
                println!("{label} 局部 → {path}");
            }
            // 這一塊的平均亮度：殘留多少用數字看，不必猜
            let mean = |i: &image::RgbImage| {
                let s: f64 = i
                    .pixels()
                    .map(|p| p.0.iter().map(|&v| v as f64).sum::<f64>())
                    .sum();
                s / (i.width() * i.height() * 3) as f64
            };
            let b = image::imageops::crop_imm(&img, x, y, w, h).to_image();
            let af = image::imageops::crop_imm(&out, x, y, w, h).to_image();
            println!(
                "這一塊平均亮度 {:.1} → {:.1}（留 {:.0}%）",
                mean(&b),
                mean(&af),
                mean(&af) / mean(&b) * 100.0
            );
            // 原本就過曝的像素（已經沒有細節）扣完剩多少：變灰的話這個數字會掉
            let hi = |i: &image::RgbImage, m: &image::RgbImage| {
                let (mut s, mut n) = (0f64, 0u64);
                for (p, q) in i.pixels().zip(m.pixels()) {
                    if q.0.iter().copied().max().unwrap() >= 250 {
                        s += p.0.iter().map(|&v| v as f64).sum::<f64>() / 3.0;
                        n += 1;
                    }
                }
                (s / n.max(1) as f64, n)
            };
            let (hb, n) = hi(&b, &b);
            let (ha, _) = hi(&af, &b);
            println!(
                "其中原本過曝的 {n} 個像素：{hb:.1} → {ha:.1}（留 {:.0}%）",
                ha / hb * 100.0
            );
        }
    }
}
