//! 從圖庫學來的鳥頭辨識：一個很小的卷積網路，判斷「這一點是不是使用者會標的
//! 那個位置」（側面是眼睛，正面、俯衝時是額頭、頭頂中央）。
//!
//! 為什麼不像五色鳥、八色鳥那樣寫規則：翠鳥的頭頂是深藍綠帶鮮綠橫紋，翅膀
//! 覆羽也是藍綠帶淺藍斑點，橘色的耳羽、胸口與紅腳又搶走「顏色雜」的分數；
//! 俯衝時頭朝下、出水時頭朝上，什麼角度都有。實測用顏色組成判斷（各色比例
//! 加上周圍一圈），在鳥附近找頭只對三成；拿圖庫的頭當範本（含旋轉、翻轉）
//! 比對也只對兩成半。這個網路用使用者框好的圖庫訓練，保留 21 段連拍不給它看
//! 當考題，已知頭多大時 118 張對了 104 張（落在半個框寬以內）。
//!
//! 網路的輸入固定是「頭寬 [`KS`] 像素」的彩色圖：照片先縮到那個比例再整片
//! 一起算，每 4 個像素得一個分數。頭多大事先不知道，就多試幾種縮放
//! （見 `track::net_heads`）。
//!
//! 權重放在 `assets/kingfisher_head.bin`（f32 小端序，照 [`L`] 的順序排）。
//! 重新訓練見本檔最後的 `train_from_project`。

use image::RgbImage;
use std::sync::OnceLock;

/// 網路假設的頭寬（像素）：照片縮到頭寬這麼大再算
pub const KS: f32 = 12.0;
/// 判斷一點時看的範圍（邊長，像素）：兩個頭寬
pub const NP: usize = 24;
/// 輸出圖每一格相隔幾個像素（兩層 2×2 池化）
pub const STRIDE: usize = 4;
/// 訓練時每一塊多帶的周邊（像素）：卷積看得到真正的鄰居，與整片一起算的結果才一致
const CTX: usize = 8;
const C1: usize = 12;
const C2: usize = 16;
const C3: usize = 24;
const FH: usize = 32;
const FLAT: usize = C3 * 6 * 6;

/// 各層權重在扁平陣列裡的起點
struct Lay {
    w1: usize,
    b1: usize,
    w2: usize,
    b2: usize,
    w3: usize,
    b3: usize,
    w4: usize,
    b4: usize,
    w5: usize,
    b5: usize,
    n: usize,
}

const L: Lay = {
    let w1 = 0;
    let b1 = w1 + C1 * 3 * 9;
    let w2 = b1 + C1;
    let b2 = w2 + C2 * C1 * 9;
    let w3 = b2 + C2;
    let b3 = w3 + C3 * C2 * 9;
    let w4 = b3 + C3;
    let b4 = w4 + FH * FLAT;
    let w5 = b4 + FH;
    let b5 = w5 + FH;
    Lay { w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, n: b5 + 1 }
};

static KINGFISHER: &[u8] = include_bytes!("../assets/kingfisher_head.bin");

/// 翠鳥的權重
pub fn kingfisher() -> &'static [f32] {
    static W: OnceLock<Vec<f32>> = OnceLock::new();
    W.get_or_init(|| {
        // 量測時可以換成別份權重（例如沒看過考題的那一份）
        #[cfg(test)]
        if let Ok(p) = std::env::var("P2V_HEAD_NET") {
            return std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        }
        let w: Vec<f32> = KINGFISHER.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(w.len(), L.n, "kingfisher_head.bin 的大小與網路不合");
        w
    })
}

/// 3×3 卷積（邊上補零），輸入 ci×h×w → 輸出 co×h×w
fn conv_f(x: &[f32], ci: usize, h: usize, w: usize, wt: &[f32], b: &[f32], co: usize) -> Vec<f32> {
    let hw = h * w;
    let mut y = vec![0f32; co * hw];
    for o in 0..co {
        let yo = &mut y[o * hw..(o + 1) * hw];
        yo.fill(b[o]);
        for i in 0..ci {
            let xi = &x[i * hw..(i + 1) * hw];
            for ky in 0..3 {
                for kx in 0..3 {
                    let wv = wt[((o * ci + i) * 3 + ky) * 3 + kx];
                    let (x0, x1) = (usize::from(kx == 0), if kx == 2 { w - 1 } else { w });
                    for yy in 0..h {
                        let sy = yy as isize + ky as isize - 1;
                        if sy < 0 || sy >= h as isize {
                            continue;
                        }
                        let row = &xi[sy as usize * w..];
                        let out = &mut yo[yy * w..];
                        for xx in x0..x1 {
                            out[xx] += wv * row[xx + kx - 1];
                        }
                    }
                }
            }
        }
    }
    y
}

/// 2×2 最大池化，順便記下每一格取的是哪個輸入（訓練時倒傳用）
fn pool_f(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, Vec<usize>) {
    let (gh, gw) = (h / 2, w / 2);
    let mut y = vec![0f32; c * gh * gw];
    let mut idx = vec![0usize; c * gh * gw];
    for ch in 0..c {
        for yy in 0..gh {
            for xx in 0..gw {
                let mut best = (f32::MIN, 0);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let i = ch * h * w + (2 * yy + dy) * w + 2 * xx + dx;
                    if x[i] > best.0 {
                        best = (x[i], i);
                    }
                }
                y[ch * gh * gw + yy * gw + xx] = best.0;
                idx[ch * gh * gw + yy * gw + xx] = best.1;
            }
        }
    }
    (y, idx)
}

fn relu(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

/// 三層卷積的中間結果（訓練倒傳要用）
struct Conv {
    a1: Vec<f32>,
    p1: Vec<f32>,
    i1: Vec<usize>,
    a2: Vec<f32>,
    p2: Vec<f32>,
    i2: Vec<usize>,
    a3: Vec<f32>,
}

/// 三層卷積：輸入 3×h×w → 輸出 C3×(h/4)×(w/4)
fn net_conv(p: &[f32], x: &[f32], h: usize, w: usize) -> Conv {
    let mut a1 = conv_f(x, 3, h, w, &p[L.w1..L.b1], &p[L.b1..L.w2], C1);
    relu(&mut a1);
    let (p1, i1) = pool_f(&a1, C1, h, w);
    let mut a2 = conv_f(&p1, C1, h / 2, w / 2, &p[L.w2..L.b2], &p[L.b2..L.w3], C2);
    relu(&mut a2);
    let (p2, i2) = pool_f(&a2, C2, h / 2, w / 2);
    let mut a3 = conv_f(&p2, C2, h / 4, w / 4, &p[L.w3..L.b3], &p[L.b3..L.w4], C3);
    relu(&mut a3);
    Conv { a1, p1, i1, a2, p2, i2, a3 }
}

/// 最後兩層（全連接）：a3 圖（gh×gw）上以 (u, v) 為左上角的 6×6 → 分數（logit）
fn net_head(p: &[f32], a3: &[f32], gh: usize, gw: usize, u: usize, v: usize) -> (Vec<f32>, Vec<f32>, f32) {
    let mut flat = vec![0f32; FLAT];
    for c in 0..C3 {
        for j in 0..6 {
            let src = c * gh * gw + (v + j) * gw + u;
            flat[(c * 6 + j) * 6..(c * 6 + j) * 6 + 6].copy_from_slice(&a3[src..src + 6]);
        }
    }
    let mut h4 = vec![0f32; FH];
    for j in 0..FH {
        let wr = &p[L.w4 + j * FLAT..L.w4 + (j + 1) * FLAT];
        h4[j] = p[L.b4 + j] + wr.iter().zip(&flat).map(|(a, b)| a * b).sum::<f32>();
    }
    relu(&mut h4);
    let z = p[L.b5] + (0..FH).map(|j| p[L.w5 + j] * h4[j]).sum::<f32>();
    (flat, h4, z)
}

/// 整張一起算的分數圖：第 (u, v) 格的判斷範圍中心在
/// (u·[`STRIDE`] + NP/2, v·STRIDE + NP/2)（輸入圖的像素）
pub struct ScoreMap {
    pub w: usize,
    pub h: usize,
    pub z: Vec<f32>,
}

/// 把一張彩色圖（已縮到頭寬約 [`KS`]）整片算一遍
pub fn score_map(p: &[f32], img: &RgbImage) -> ScoreMap {
    let (w, h) = ((img.width() as usize) / STRIDE * STRIDE, (img.height() as usize) / STRIDE * STRIDE);
    if w < NP || h < NP {
        return ScoreMap { w: 0, h: 0, z: Vec::new() };
    }
    let mut x = vec![0f32; 3 * w * h];
    for yy in 0..h {
        for xx in 0..w {
            let q = img.get_pixel(xx as u32, yy as u32);
            for c in 0..3 {
                x[c * w * h + yy * w + xx] = q[c] as f32 / 255.0 - 0.5;
            }
        }
    }
    let a3 = net_conv(p, &x, h, w).a3;
    let (gh, gw) = (h / STRIDE, w / STRIDE);
    let (mh, mw) = (gh - 5, gw - 5);
    let mut z = Vec::with_capacity(mh * mw);
    for v in 0..mh {
        for u in 0..mw {
            z.push(net_head(p, &a3, gh, gw, u, v).2);
        }
    }
    ScoreMap { w: mw, h: mh, z }
}

impl ScoreMap {
    /// 分數圖上的局部最高點（比周圍 8 格都高、至少 `min`），最高的在前，最多 `max` 個。
    /// 位置用左右、上下各三格配拋物線補到格子之間，回傳輸入圖的像素座標
    pub fn peaks(&self, min: f32, max: usize) -> Vec<(f32, f32, f32)> {
        let (w, h) = (self.w, self.h);
        let at = |u: usize, v: usize| self.z[v * w + u];
        let mut out = Vec::new();
        for v in 0..h {
            for u in 0..w {
                let c = at(u, v);
                if c < min {
                    continue;
                }
                let mut top = true;
                for dv in -1i32..=1 {
                    for du in -1i32..=1 {
                        let (uu, vv) = (u as i32 + du, v as i32 + dv);
                        if (du, dv) == (0, 0) || uu < 0 || vv < 0 || uu >= w as i32 || vv >= h as i32 {
                            continue;
                        }
                        let o = at(uu as usize, vv as usize);
                        // 平手時只認先掃到的那一格
                        if o > c || (o == c && (dv < 0 || (dv == 0 && du < 0))) {
                            top = false;
                        }
                    }
                }
                if !top {
                    continue;
                }
                let sub = |a: f32, b: f32| {
                    let d = a - 2.0 * c + b;
                    if d < 0.0 { (0.5 * (a - b) / d).clamp(-0.5, 0.5) } else { 0.0 }
                };
                let du = if u > 0 && u + 1 < w { sub(at(u - 1, v), at(u + 1, v)) } else { 0.0 };
                let dv = if v > 0 && v + 1 < h { sub(at(u, v - 1), at(u, v + 1)) } else { 0.0 };
                let half = NP as f32 / 2.0;
                out.push(((u as f32 + du) * STRIDE as f32 + half, (v as f32 + dv) * STRIDE as f32 + half, c));
            }
        }
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        out.truncate(max);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 整片一起算與「只算一塊」要一致（判斷範圍外圍帶 CTX 的周邊時）
    #[test]
    fn whole_map_matches_single_patch() {
        let p = kingfisher();
        let img = RgbImage::from_fn(64, 48, |x, y| image::Rgb([(x * 4) as u8, (y * 5) as u8, ((x * y) % 255) as u8]));
        let map = score_map(p, &img);
        // 第 (2, 2) 格：判斷範圍左上角在 (8, 8)，帶上周邊就是從 (0, 0) 起的 40×40
        let (u, v) = (2usize, 2usize);
        let n = NP + 2 * CTX;
        let (ox, oy) = (u * STRIDE - CTX, v * STRIDE - CTX);
        let mut x = vec![0f32; 3 * n * n];
        for j in 0..n {
            for i in 0..n {
                let q = img.get_pixel((ox + i) as u32, (oy + j) as u32);
                for c in 0..3 {
                    x[c * n * n + j * n + i] = q[c] as f32 / 255.0 - 0.5;
                }
            }
        }
        let a3 = net_conv(p, &x, n, n).a3;
        let z = net_head(p, &a3, n / 4, n / 4, CTX / 4, CTX / 4).2;
        let whole = map.z[v * map.w + u];
        assert!((z - whole).abs() < 1e-3, "{z} vs {whole}");
    }

    /// 一個樣本的前向結果（倒傳用）
    struct Fwd {
        x: Vec<f32>,
        conv: Conv,
        flat: Vec<f32>,
        h4: Vec<f32>,
        z: f32,
    }

    fn net_f(p: &[f32], x: Vec<f32>) -> Fwd {
        let n = NP + 2 * CTX;
        let conv = net_conv(p, &x, n, n);
        let (flat, h4, z) = net_head(p, &conv.a3, n / 4, n / 4, CTX / 4, CTX / 4);
        Fwd { x, conv, flat, h4, z }
    }

    /// 3×3 卷積的倒傳：累加權重與偏移的梯度，需要時回傳對輸入的梯度
    #[allow(clippy::too_many_arguments)]
    fn conv_b(x: &[f32], ci: usize, h: usize, w: usize, wt: &[f32], co: usize, dy: &[f32], dw: &mut [f32], db: &mut [f32], need_dx: bool) -> Vec<f32> {
        let hw = h * w;
        let mut dx = if need_dx { vec![0f32; ci * hw] } else { Vec::new() };
        for o in 0..co {
            let dyo = &dy[o * hw..(o + 1) * hw];
            db[o] += dyo.iter().sum::<f32>();
            for i in 0..ci {
                let xi = &x[i * hw..(i + 1) * hw];
                for ky in 0..3 {
                    for kx in 0..3 {
                        let k = ((o * ci + i) * 3 + ky) * 3 + kx;
                        let wv = wt[k];
                        let mut g = 0f32;
                        let (x0, x1) = (usize::from(kx == 0), if kx == 2 { w - 1 } else { w });
                        for yy in 0..h {
                            let sy = yy as isize + ky as isize - 1;
                            if sy < 0 || sy >= h as isize {
                                continue;
                            }
                            let sy = sy as usize;
                            for xx in x0..x1 {
                                let d = dyo[yy * w + xx];
                                g += d * xi[sy * w + xx + kx - 1];
                                if need_dx {
                                    dx[i * hw + sy * w + xx + kx - 1] += d * wv;
                                }
                            }
                        }
                        dw[k] += g;
                    }
                }
            }
        }
        dx
    }

    fn net_b(p: &[f32], f: &Fwd, dz: f32, g: &mut [f32]) {
        let n = NP + 2 * CTX;
        g[L.b5] += dz;
        let mut dh4 = vec![0f32; FH];
        for j in 0..FH {
            g[L.w5 + j] += dz * f.h4[j];
            dh4[j] = if f.h4[j] > 0.0 { dz * p[L.w5 + j] } else { 0.0 };
        }
        let mut dflat = vec![0f32; FLAT];
        for j in 0..FH {
            if dh4[j] == 0.0 {
                continue;
            }
            g[L.b4 + j] += dh4[j];
            for k in 0..FLAT {
                g[L.w4 + j * FLAT + k] += dh4[j] * f.flat[k];
                dflat[k] += dh4[j] * p[L.w4 + j * FLAT + k];
            }
        }
        let (gq, o) = (n / 4, CTX / 4);
        let c = &f.conv;
        let mut da3 = vec![0f32; c.a3.len()];
        for ch in 0..C3 {
            for j in 0..6 {
                for i in 0..6 {
                    let k = ch * gq * gq + (o + j) * gq + o + i;
                    if c.a3[k] > 0.0 {
                        da3[k] = dflat[(ch * 6 + j) * 6 + i];
                    }
                }
            }
        }
        let (gw3, rest) = g[L.w3..].split_at_mut(L.b3 - L.w3);
        let dp2 = conv_b(&c.p2, C2, gq, gq, &p[L.w3..L.b3], C3, &da3, gw3, &mut rest[..C3], true);
        let mut da2 = vec![0f32; c.a2.len()];
        for (k, &i) in c.i2.iter().enumerate() {
            da2[i] += dp2[k];
        }
        for k in 0..da2.len() {
            if c.a2[k] <= 0.0 {
                da2[k] = 0.0;
            }
        }
        let (gw2, rest) = g[L.w2..].split_at_mut(L.b2 - L.w2);
        let dp1 = conv_b(&c.p1, C1, n / 2, n / 2, &p[L.w2..L.b2], C2, &da2, gw2, &mut rest[..C2], true);
        let mut da1 = vec![0f32; c.a1.len()];
        for (k, &i) in c.i1.iter().enumerate() {
            da1[i] += dp1[k];
        }
        for k in 0..da1.len() {
            if c.a1[k] <= 0.0 {
                da1[k] = 0.0;
            }
        }
        let (gw1, rest) = g[L.w1..].split_at_mut(L.b1 - L.w1);
        conv_b(&f.x, 3, n, n, &p[L.w1..L.b1], C1, &da1, gw1, &mut rest[..C1], false);
    }

    /// 以 (cx, cy) 為中心、每格 step 像素、轉 ang、可左右翻，取 n×n 的彩色小塊
    #[allow(clippy::too_many_arguments)]
    fn patch(img: &RgbImage, cx: f32, cy: f32, n: usize, step: f32, ang: f32, mirror: bool, gain: [f32; 3]) -> Vec<f32> {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let (ca, sa) = (ang.cos(), ang.sin());
        let mut out = vec![0f32; 3 * n * n];
        let half = n as f32 / 2.0;
        for j in 0..n {
            for i in 0..n {
                let ii = if mirror { n - 1 - i } else { i };
                let u = (ii as f32 + 0.5 - half) * step;
                let v = (j as f32 + 0.5 - half) * step;
                let x = (cx + u * ca - v * sa - 0.5).clamp(0.0, w - 1.001);
                let y = (cy + u * sa + v * ca - 0.5).clamp(0.0, h - 1.001);
                let (x0, y0) = (x as u32, y as u32);
                let (fx, fy) = (x - x0 as f32, y - y0 as f32);
                let (x1, y1) = ((x0 + 1).min(img.width() - 1), (y0 + 1).min(img.height() - 1));
                let (p00, p10, p01, p11) = (img.get_pixel(x0, y0), img.get_pixel(x1, y0), img.get_pixel(x0, y1), img.get_pixel(x1, y1));
                for c in 0..3 {
                    let a = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                    let b = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                    out[c * n * n + j * n + i] = ((a * (1.0 - fy) + b * fy) * gain[c] / 255.0).min(1.0) - 0.5;
                }
            }
        }
        out
    }

    /// 一張訓練照片：縮到頭寬 KS 的一層，加上 2、4、8 倍大的幾層
    /// （取「比頭小很多」的反例時要從清楚的那層取，才跟實際執行時一樣清楚）
    struct Shot {
        lv: Vec<RgbImage>,
        gx: f32,
        gy: f32,
    }

    impl Shot {
        #[allow(clippy::too_many_arguments)]
        fn patch(&self, x: f32, y: f32, n: usize, step: f32, ang: f32, mirror: bool, gain: [f32; 3]) -> Vec<f32> {
            let mut k = 0;
            while k + 1 < self.lv.len() && step * ((1u32 << k) as f32) < 1.0 {
                k += 1;
            }
            let m = (1u32 << k) as f32;
            patch(&self.lv[k], x * m, y * m, n, step * m, ang, mirror, gain)
        }
    }

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn n(&mut self) -> f32 {
            let (a, b) = (self.u().max(1e-7), self.u());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
    }

    /// 重新訓練：拿一個專案裡使用者手動框的照片當圖庫（框中心＝要學的位置，
    /// 框寬當作頭寬）。
    ///
    /// `P2V_HEAD_P2V=專案檔 P2V_HEAD_OUT=輸出.bin cargo test --release
    /// train_from_project -- --ignored --nocapture`，再把輸出檔換掉
    /// `assets/` 裡對應的那一個。
    ///
    /// 每張每輪取 8 個正樣本（框中心附近、隨意旋轉翻轉、明暗與色偏、大小差兩成內）
    /// 與 28 個反例：一部分在頭附近、一部分整張隨意，還有大半是「大小不對」——
    /// 不教它這個，多種縮放一起找時，羽毛上的小斑點也會被當成很小的頭
    #[test]
    #[ignore]
    fn train_from_project() {
        let proj = std::env::var("P2V_HEAD_P2V").expect("P2V_HEAD_P2V");
        let out = std::env::var("P2V_HEAD_OUT").expect("P2V_HEAD_OUT");
        let epochs: usize = std::env::var("P2V_HEAD_EPOCHS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
        let pf: crate::ProjectFile = serde_json::from_str(&std::fs::read_to_string(&proj).unwrap()).unwrap();
        let labels: Vec<_> = pf.track_boxes.iter().filter(|b| b.2 == 0).cloned().collect();
        let shots: Vec<Shot> = std::thread::scope(|sc| {
            let hs: Vec<_> = labels
                .chunks(labels.len().div_ceil(8).max(1))
                .map(|part| {
                    sc.spawn(move || {
                        part.iter()
                            .filter_map(|(path, r, _, _)| {
                                let img = image::open(path).ok()?.to_rgb8();
                                let k = KS / ((r[2] - r[0]) * img.width() as f32);
                                let (w, h) = ((img.width() as f32 * k).round() as u32, (img.height() as f32 * k).round() as u32);
                                let lv = (0..4)
                                    .map(|s| image::imageops::resize(&img, w << s, h << s, image::imageops::FilterType::Triangle))
                                    .collect();
                                Some(Shot { lv, gx: (r[0] + r[2]) / 2.0 * w as f32, gy: (r[1] + r[3]) / 2.0 * h as f32 })
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        println!("{} 張", shots.len());
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let mut p = vec![0f32; L.n];
        for (a, b, fan) in [(L.w1, L.b1, 27), (L.w2, L.b2, C1 * 9), (L.w3, L.b3, C2 * 9), (L.w4, L.b4, FLAT), (L.w5, L.b5, FH)] {
            let sd = (2.0 / fan as f32).sqrt();
            for v in &mut p[a..b] {
                *v = rng.n() * sd;
            }
        }
        let (mut m1, mut m2) = (vec![0f32; L.n], vec![0f32; L.n]);
        let mut t = 0i32;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let n_in = NP + 2 * CTX;
        for ep in 0..epochs {
            // （第幾張, x, y, 角度, 翻轉, 大小倍數, 增益, 標籤）
            let mut samples: Vec<(usize, f32, f32, f32, bool, f32, [f32; 3], f32)> = Vec::new();
            for (k, sh) in shots.iter().enumerate() {
                let (w, h) = (sh.lv[0].width() as f32, sh.lv[0].height() as f32);
                for q in 0..36 {
                    let pos = q < 8;
                    let wrong_scale = !pos && q % 3 != 2;
                    let scale = if wrong_scale {
                        if rng.u() < 0.6 {
                            (0.1f32.ln() + rng.u() * 6.0f32.ln()).exp()
                        } else {
                            (1.6f32.ln() + rng.u() * 2.5f32.ln()).exp()
                        }
                    } else {
                        0.8 + rng.u() * 0.45
                    };
                    let (x, y) = if pos || (wrong_scale && q % 2 == 0) {
                        (sh.gx + (rng.u() - 0.5) * 0.25 * KS, sh.gy + (rng.u() - 0.5) * 0.25 * KS)
                    } else {
                        loop {
                            let (x, y) = if q % 3 == 0 {
                                (rng.u() * w, rng.u() * h)
                            } else {
                                let (a, d) = (rng.u() * std::f32::consts::TAU, (0.55 + rng.u() * 2.5) * KS);
                                (sh.gx + a.cos() * d, sh.gy + a.sin() * d)
                            };
                            if (x - sh.gx).hypot(y - sh.gy) >= 0.55 * KS && x >= 0.0 && y >= 0.0 && x < w && y < h {
                                break (x, y);
                            }
                        }
                    };
                    let gain = 0.7 + rng.u() * 0.6;
                    let g = [gain * (0.92 + rng.u() * 0.16), gain * (0.92 + rng.u() * 0.16), gain * (0.92 + rng.u() * 0.16)];
                    samples.push((k, x, y, rng.u() * std::f32::consts::TAU, rng.u() < 0.5, scale, g, if pos { 1.0 } else { 0.0 }));
                }
            }
            for i in (1..samples.len()).rev() {
                let j = (rng.u() * (i + 1) as f32) as usize % (i + 1);
                samples.swap(i, j);
            }
            let lr = 0.002 * if ep >= epochs * 3 / 4 { 0.25 } else { 1.0 };
            let mut loss = 0f64;
            for batch in samples.chunks(64) {
                let grads: Vec<(Vec<f32>, f64)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = batch
                        .chunks(batch.len().div_ceil(threads))
                        .map(|part| {
                            let (p, shots) = (&p, &shots);
                            sc.spawn(move || {
                                let mut g = vec![0f32; L.n];
                                let mut ls = 0f64;
                                for &(k, x, y, a, mi, scale, gain, lab) in part {
                                    let f = net_f(p, shots[k].patch(x, y, n_in, 2.0 * KS * scale / NP as f32, a, mi, gain));
                                    let pr = 1.0 / (1.0 + (-f.z).exp());
                                    let wt = if lab > 0.5 { 3.0 } else { 1.0 };
                                    ls -= (wt * if lab > 0.5 { pr.max(1e-6).ln() } else { (1.0 - pr).max(1e-6).ln() }) as f64;
                                    net_b(p, &f, (pr - lab) * wt, &mut g);
                                }
                                (g, ls)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let mut g = vec![0f32; L.n];
                for (gg, ls) in &grads {
                    for i in 0..L.n {
                        g[i] += gg[i];
                    }
                    loss += ls;
                }
                // Adam，加一點權重衰減
                t += 1;
                let bn = batch.len() as f32;
                for i in 0..L.n {
                    let gi = g[i] / bn + 1e-4 * p[i];
                    m1[i] = 0.9 * m1[i] + 0.1 * gi;
                    m2[i] = 0.999 * m2[i] + 0.001 * gi * gi;
                    let mh = m1[i] / (1.0 - 0.9f32.powi(t));
                    let vh = m2[i] / (1.0 - 0.999f32.powi(t));
                    p[i] -= lr * mh / (vh.sqrt() + 1e-8);
                }
            }
            println!("第 {ep} 輪 loss {:.4}", loss / samples.len() as f64);
        }
        let bytes: Vec<u8> = p.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out, bytes).unwrap();
    }
}
