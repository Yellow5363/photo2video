#!/usr/bin/env bash
# 煙霧測試：用 test_images 以 CLI 模式實際轉出影片，驗證核心轉檔沒有回歸。
# 發版前可跑一次確認輸出的時長／解析度／格式正確。
#
# 需求：已建置的 release exe（會自動 cargo build --release）、dist/ffmpeg.exe
# 用法：bash scripts/smoke-test.sh
set -u

cd "$(dirname "$0")/.." || exit 1

EXE="./target/release/photo2video.exe"
FF="./dist/ffmpeg.exe"
IMAGES="test_images"
TMP="${TEMP:-/tmp}/p2v_smoke"
FAIL=0

[ -f "$FF" ] || { echo "✗ 找不到 $FF（驗證需要）"; exit 1; }
[ -d "$IMAGES" ] || { echo "✗ 找不到 $IMAGES"; exit 1; }

echo "→ 建置 release…"
# 注意：不能寫成 `cargo build … | tail -1 || …`——管線的退出碼來自 tail
# （永遠 0），build 真的失敗也偵測不到，會拿舊 exe 給出假通過。用
# PIPESTATUS 取 cargo 自己的退出碼。
cargo build --release 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "✗ 建置失敗"; exit 1; }

mkdir -p "$TMP"
N=$(ls "$IMAGES" | wc -l | tr -d ' ')

# 取影片時長（秒，兩位小數）。解析完整 HH:MM:SS.ss 再換算成秒：
# 不能像舊版用 `00:00:0*` 硬吃前導零——它會把「00.50」的整數位 0 也吃掉、
# 回傳「.50」而非「0.50」，任何不足一秒的影片（如單張照片）都會誤判失敗。
duration() {
  local d; d=$("$FF" -i "$1" 2>&1 | sed -n 's/.*Duration: \([0-9:.]*\),.*/\1/p')
  awk -F: 'NF==3{printf "%.2f", $1*3600+$2*60+$3}' <<<"$d"
}

# 一個案例：格式、fps、期望時長
case_run() {
  local label="$1" ext="$2" fps="$3" want="$4"
  local out="$TMP/smoke.$ext"
  rm -f "$out"
  "$EXE" --cli "$IMAGES" "$fps" "$out" >/dev/null 2>&1
  if [ ! -f "$out" ]; then echo "✗ $label：沒有產出檔案"; FAIL=1; return; fi
  local got; got=$(duration "$out")
  local res; res=$("$FF" -i "$out" 2>&1 | sed -n 's/.*, \([0-9]*x[0-9]*\).*/\1/p' | head -1)
  if [ "$got" = "$want" ] && [ "$res" = "1920x1080" ]; then
    echo "✓ $label：時長 ${got}s、解析度 $res"
  else
    echo "✗ $label：時長 ${got}s（期望 ${want}s）、解析度 $res（期望 1920x1080）"; FAIL=1
  fi
  rm -f "$out"
}

# 錯誤案例：以指定引數執行，期望非 0 退出碼且錯誤訊息含某片段。
# 用法：error_case "標題" "訊息片段" <cli 引數…>
error_case() {
  local label="$1" want="$2"; shift 2
  local out; out=$("$EXE" --cli "$@" 2>&1); local code=$?
  if [ "$code" != 0 ] && printf '%s' "$out" | grep -q "$want"; then
    echo "✓ $label：正確拒絕（含「$want」）"
  else
    echo "✗ $label：退出碼 $code、輸出「$out」（期望非 0 且含「$want」）"; FAIL=1
  fi
}

# 各容器格式，N 張、fps → 時長 = N/fps
case_run "MP4 (H.264) fps=2" mp4  2 "$(awk "BEGIN{printf \"%.2f\", $N/2}")"
case_run "MKV (H.264) fps=2" mkv  2 "$(awk "BEGIN{printf \"%.2f\", $N/2}")"
case_run "MOV (H.264) fps=2" mov  2 "$(awk "BEGIN{printf \"%.2f\", $N/2}")"
case_run "AVI (H.264) fps=2" avi  2 "$(awk "BEGIN{printf \"%.2f\", $N/2}")"
case_run "WebM (VP9)  fps=3" webm 3 "$(awk "BEGIN{printf \"%.2f\", $N/3}")"
# 極端 fps 邊界：最慢（每張 1 秒）與最快（每張 1/60 秒，考驗 concat 清單對
# 極小 duration 的精度；不足一秒也順帶考驗 duration() 解析）
case_run "MP4 fps=1（慢）"  mp4 1  "$(awk "BEGIN{printf \"%.2f\", $N/1}")"
case_run "MP4 fps=60（快）" mp4 60 "$(awk "BEGIN{printf \"%.2f\", $N/60}")"

# 單張照片：concat demuxer「最後一張要再列一次」的慣例在 N=1 時是唯一輸入
# 又是結尾張，最容易出邊界問題；且時長不足一秒，順帶驗證 duration() 的解析。
SINGLE="$TMP/single"
rm -rf "$SINGLE"; mkdir -p "$SINGLE"
"$FF" -f lavfi -i color=purple:s=1000x800:d=1 -frames:v 1 -y "$SINGLE/only.png" >/dev/null 2>&1
single_out="$TMP/single.mp4"; rm -f "$single_out"
"$EXE" --cli "$SINGLE" 2 "$single_out" >/dev/null 2>&1   # 1 張 / fps 2 → 0.50s
single_res=$("$FF" -i "$single_out" 2>&1 | sed -n 's/.*, \([0-9]*x[0-9]*\).*/\1/p' | head -1)
single_dur=$(duration "$single_out")
if [ "$single_dur" = "0.50" ] && [ "$single_res" = "1920x1080" ]; then
  echo "✓ 單張照片 fps=2：時長 ${single_dur}s、解析度 $single_res"
else
  echo "✗ 單張照片 fps=2：時長 ${single_dur}s（期望 0.50）、解析度 $single_res（期望 1920x1080）"; FAIL=1
fi
rm -rf "$SINGLE" "$single_out"

# 副檔名處理：輸出無副檔名時應自動補成所選格式（預設 mp4）
ext_out="$TMP/smoke_noext"
rm -f "$ext_out" "$ext_out.mp4"
"$EXE" --cli "$IMAGES" 2 "$ext_out" >/dev/null 2>&1
if [ -f "$ext_out.mp4" ]; then
  echo "✓ 無副檔名輸出：自動補成 .mp4"
else
  echo "✗ 無副檔名輸出：預期補成 .mp4，未產出"; FAIL=1
fi
rm -f "$ext_out" "$ext_out.mp4"

# 大寫副檔名須不分大小寫辨識格式：OUT.WEBM 要選 VP9，否則會落到預設
# mp4(H264)、被 ffmpeg 依 .WEBM 寫成 H264-in-WebM 這種不相容檔案。
up_out="$TMP/UP.WEBM"; rm -f "$up_out"
"$EXE" --cli "$IMAGES" 3 "$up_out" >/dev/null 2>&1
up_codec=$("$FF" -i "$up_out" 2>&1 | sed -n 's/.*Video: \([a-z0-9]*\).*/\1/p' | head -1)
if [ "$up_codec" = "vp9" ]; then
  echo "✓ 大寫副檔名 .WEBM：正確選用 VP9 編碼"
else
  echo "✗ 大寫副檔名 .WEBM：編碼為 $up_codec（期望 vp9；大小寫辨識失敗會產生不相容檔）"; FAIL=1
fi
rm -f "$up_out"

# 混合尺寸縮放補邊：橫向／直向／小圖／4K 都應等比例縮放置中、補黑邊成 1920x1080。
# （其餘案例都用同尺寸的 test_images，這條 scale/pad 路徑靠這裡覆蓋。）
MIX="$TMP/mixed"
rm -rf "$MIX"; mkdir -p "$MIX"
"$FF" -f lavfi -i color=red:s=1920x1080:d=1    -frames:v 1 -y "$MIX/1_landscape.png" >/dev/null 2>&1
"$FF" -f lavfi -i color=blue:s=1080x1920:d=1   -frames:v 1 -y "$MIX/2_portrait.png"  >/dev/null 2>&1
"$FF" -f lavfi -i color=green:s=400x300:d=1    -frames:v 1 -y "$MIX/3_small.png"      >/dev/null 2>&1
"$FF" -f lavfi -i color=yellow:s=3840x2160:d=1 -frames:v 1 -y "$MIX/4_4k.png"         >/dev/null 2>&1
mix_out="$TMP/mixed.mp4"; rm -f "$mix_out"
"$EXE" --cli "$MIX" 2 "$mix_out" >/dev/null 2>&1
mix_res=$("$FF" -i "$mix_out" 2>&1 | sed -n 's/.*, \([0-9]*x[0-9]*\).*/\1/p' | head -1)
mix_dur=$(duration "$mix_out")
if [ "$mix_res" = "1920x1080" ] && [ "$mix_dur" = "2.00" ]; then
  echo "✓ 混合尺寸（橫/直/小/4K）：都輸出 1920x1080、時長 ${mix_dur}s"
else
  echo "✗ 混合尺寸：解析度 $mix_res（期望 1920x1080）、時長 ${mix_dur}s（期望 2.00）"; FAIL=1
fi
rm -rf "$MIX" "$mix_out"

# 0 位元組空檔：下載中斷、雲端同步佔位、存檔失敗常留下 0 位元組的圖片檔。
# 混進 concat 清單時 demuxer 讀到它就會中止、把其後所有照片一起靜默丟掉，
# 成品少照片卻回報成功。程式須在收集照片時就排除空檔。
ZERODIR="$TMP/zero"
rm -rf "$ZERODIR"; mkdir -p "$ZERODIR"
"$FF" -f lavfi -i color=red:s=640x480:d=1  -frames:v 1 -y "$ZERODIR/1.png" >/dev/null 2>&1
: > "$ZERODIR/2.png"   # 0 位元組空檔夾在中間
"$FF" -f lavfi -i color=blue:s=640x480:d=1 -frames:v 1 -y "$ZERODIR/3.png" >/dev/null 2>&1
zero_out="$TMP/zero.mp4"; rm -f "$zero_out"
"$EXE" --cli "$ZERODIR" 2 "$zero_out" >/dev/null 2>&1
zero_dur=$(duration "$zero_out")
if [ "$zero_dur" = "1.00" ]; then
  echo "✓ 0 位元組空檔：被排除，兩張正常照片都保留（時長 ${zero_dur}s）"
else
  echo "✗ 0 位元組空檔：時長 ${zero_dur}s（期望 1.00；短少代表空檔害後面照片被丟）"; FAIL=1
fi
rm -rf "$ZERODIR" "$zero_out"

# 只收頂層、不遞迴子資料夾：使用者選資料夾時只取該層照片（子資料夾可能是
# 縮圖/備份等不想收的內容）。頂層 2 張＋子資料夾 3 張，只應收頂層 2 張。
SUBDIR="$TMP/toplevel"
rm -rf "$SUBDIR"; mkdir -p "$SUBDIR/子相簿"
"$FF" -f lavfi -i color=red:s=640x480:d=1  -frames:v 1 -y "$SUBDIR/a.png" >/dev/null 2>&1
"$FF" -f lavfi -i color=blue:s=640x480:d=1 -frames:v 1 -y "$SUBDIR/b.png" >/dev/null 2>&1
for i in 1 2 3; do "$FF" -f lavfi -i color=green:s=640x480:d=1 -frames:v 1 -y "$SUBDIR/子相簿/$i.png" >/dev/null 2>&1; done
sub_out="$TMP/toplevel.mp4"; rm -f "$sub_out"
"$EXE" --cli "$SUBDIR" 2 "$sub_out" >/dev/null 2>&1
sub_dur=$(duration "$sub_out")
if [ "$sub_dur" = "1.00" ]; then
  echo "✓ 只收頂層：忽略子資料夾，只轉頂層 2 張（時長 ${sub_dur}s）"
else
  echo "✗ 只收頂層：時長 ${sub_dur}s（期望 1.00＝2 張；若為 2.50 代表誤把子資料夾一起收了）"; FAIL=1
fi
rm -rf "$SUBDIR" "$sub_out"

# 混合格式：一個資料夾內同時有 jpg/png/bmp/webp/tif（手機照片＋截圖＋下載圖
# 很常見）。concat demuxer 要求編碼一致，程式須先把非 PNG 的照片正規化，否則
# 會靜默丟格（成品少照片）甚至整個失敗。以 6 種格式各一張、fps=2 驗證時長 3.00s。
FMTDIR="$TMP/fmts"
rm -rf "$FMTDIR"; mkdir -p "$FMTDIR"
n=1
for f in jpg png bmp webp tif tiff; do
  "$FF" -f lavfi -i "color=red:s=640x480:d=1" -frames:v 1 -y "$FMTDIR/$n.$f" >/dev/null 2>&1
  n=$((n+1))
done
fmt_out="$TMP/fmts.mp4"; rm -f "$fmt_out"
"$EXE" --cli "$FMTDIR" 2 "$fmt_out" >/dev/null 2>&1
fmt_res=$("$FF" -i "$fmt_out" 2>&1 | sed -n 's/.*, \([0-9]*x[0-9]*\).*/\1/p' | head -1)
fmt_dur=$(duration "$fmt_out")
if [ "$fmt_dur" = "3.00" ] && [ "$fmt_res" = "1920x1080" ]; then
  echo "✓ 混合格式（jpg/png/bmp/webp/tif/tiff）：6 張都在，時長 ${fmt_dur}s"
else
  echo "✗ 混合格式：時長 ${fmt_dur}s（期望 3.00，短少代表有照片被丟格）、解析度 $fmt_res"; FAIL=1
fi
rm -rf "$FMTDIR" "$fmt_out"

# 副檔名與實際內容不符：使用者常把 png 改名成 .jpg。若只看副檔名判斷格式會漏
# 掉這種混用、照樣丟格，故格式偵測須依檔頭實際內容。三張都叫 .jpg 但內容為
# jpeg/png/jpeg，應被正規化、3 張全保留（fps=2 → 1.50s）。
LIEDIR="$TMP/extlie"
rm -rf "$LIEDIR"; mkdir -p "$LIEDIR"
"$FF" -f lavfi -i color=red:s=640x480:d=1   -frames:v 1 -c:v mjpeg -y "$LIEDIR/1.jpg" >/dev/null 2>&1
"$FF" -f lavfi -i color=green:s=640x480:d=1 -frames:v 1 -c:v png   -y "$LIEDIR/2.jpg" >/dev/null 2>&1
"$FF" -f lavfi -i color=blue:s=640x480:d=1  -frames:v 1 -c:v mjpeg -y "$LIEDIR/3.jpg" >/dev/null 2>&1
lie_out="$TMP/extlie.mp4"; rm -f "$lie_out"
"$EXE" --cli "$LIEDIR" 2 "$lie_out" >/dev/null 2>&1
lie_dur=$(duration "$lie_out")
if [ "$lie_dur" = "1.50" ]; then
  echo "✓ 副檔名不符實際內容（png 改名 .jpg）：依內容正規化，3 張全保留（${lie_dur}s）"
else
  echo "✗ 副檔名不符實際內容：時長 ${lie_dur}s（期望 1.50；短少代表偵測被副檔名騙過而丟格）"; FAIL=1
fi
rm -rf "$LIEDIR" "$lie_out"

# 特殊字元路徑：資料夾與檔名含單引號、空格、中文（都是 Windows 合法檔名）。
# concat 清單以單引號包住每個路徑，escape 沒處理好單引號會讓整條清單解析失敗、
# 轉檔中止。這條路徑（concat_escape）先前無自動化覆蓋。
SPDIR="$TMP/引號 don't test"
rm -rf "$SPDIR"; mkdir -p "$SPDIR"
"$FF" -f lavfi -i color=red:s=640x480:d=1   -frames:v 1 -y "$SPDIR/don't 1.png"       >/dev/null 2>&1
"$FF" -f lavfi -i color=blue:s=640x480:d=1  -frames:v 1 -y "$SPDIR/我的 2.png"         >/dev/null 2>&1
"$FF" -f lavfi -i color=green:s=640x480:d=1 -frames:v 1 -y "$SPDIR/it's a test 10.png" >/dev/null 2>&1
sp_out="$TMP/引號 out.mp4"; rm -f "$sp_out"
"$EXE" --cli "$SPDIR" 2 "$sp_out" >/dev/null 2>&1
sp_res=$("$FF" -i "$sp_out" 2>&1 | sed -n 's/.*, \([0-9]*x[0-9]*\).*/\1/p' | head -1)
sp_dur=$(duration "$sp_out")
if [ "$sp_res" = "1920x1080" ] && [ "$sp_dur" = "1.50" ]; then
  echo "✓ 特殊字元路徑（單引號/空格/中文）：時長 ${sp_dur}s、解析度 $sp_res"
else
  echo "✗ 特殊字元路徑：時長 ${sp_dur}s（期望 1.50）、解析度 $sp_res（期望 1920x1080）"; FAIL=1
fi
rm -rf "$SPDIR" "$sp_out"

# EXIF 方向：手機直拍照片常以「未旋轉像素＋方向標記」儲存，轉檔必須依 EXIF
# 自動轉正（與縮圖、預覽、原始像素解析度的判斷一致），否則直拍照片會躺著輸出。
# 需 python3 + Pillow 產生帶 EXIF orientation 的測試圖；沒有就略過這個案例。
if python -c "import PIL" >/dev/null 2>&1; then
  EXIFDIR="$TMP/exif"; rm -rf "$EXIFDIR"; mkdir -p "$EXIFDIR"
  python - "$EXIFDIR" <<'PY'
from PIL import Image
import sys
d = sys.argv[1]
# 800x200 明顯橫向、亮灰（好與黑邊區分）；orientation=6 表示顯示時要轉成 200x800 直向
img = Image.new("RGB", (800, 200), (200, 200, 200))
ex = img.getexif(); ex[0x0112] = 6
img.save(f"{d}/wide.jpg", exif=ex, quality=95)
PY
  exif_out="$TMP/exif.mp4"; rm -f "$exif_out"
  "$EXE" --cli "$EXIFDIR" 2 "$exif_out" >/dev/null 2>&1
  exif_frame="$TMP/exif_frame.png"; rm -f "$exif_frame"
  "$FF" -i "$exif_out" -frames:v 1 -y "$exif_frame" >/dev/null 2>&1
  # 量出畫面中非黑內容的邊界框：已轉正→直向（高>寬），沒轉正→橫向（寬>高）
  verdict=$(python - "$exif_frame" <<'PY'
from PIL import Image
import sys
im = Image.open(sys.argv[1]).convert("RGB"); W, H = im.size; px = im.load()
xs = []; ys = []
for y in range(0, H, 4):
    for x in range(0, W, 4):
        r, g, b = px[x, y]
        if r + g + b > 150:
            xs.append(x); ys.append(y)
print("portrait" if xs and (max(ys) - min(ys)) > (max(xs) - min(xs)) else "landscape")
PY
)
  if [ "$verdict" = portrait ]; then
    echo "✓ EXIF 方向：orientation=6 直拍照片自動轉正為直向輸出"
  else
    echo "✗ EXIF 方向：未依 EXIF 轉正（輸出為 $verdict），直拍照片會躺著"; FAIL=1
  fi
  rm -rf "$EXIFDIR" "$exif_out" "$exif_frame"

  # 自然排序實際套用：不同位數編號的照片須依「數值」而非字典序排列，否則
  # 影片裡照片會亂序（img10 跑到 img2 前面）。時長／解析度檢查抓不到順序錯誤，
  # 這裡逐格比對顏色確認 img1→img2→img10→img20 的實際播放順序正確。
  ORDDIR="$TMP/order"; rm -rf "$ORDDIR"; mkdir -p "$ORDDIR"
  "$FF" -f lavfi -i color=0xFF0000:s=320x240:d=1 -frames:v 1 -y "$ORDDIR/img1.png"  >/dev/null 2>&1
  "$FF" -f lavfi -i color=0x00FF00:s=320x240:d=1 -frames:v 1 -y "$ORDDIR/img2.png"  >/dev/null 2>&1
  "$FF" -f lavfi -i color=0x0000FF:s=320x240:d=1 -frames:v 1 -y "$ORDDIR/img10.png" >/dev/null 2>&1
  "$FF" -f lavfi -i color=0xFFFF00:s=320x240:d=1 -frames:v 1 -y "$ORDDIR/img20.png" >/dev/null 2>&1
  ord_out="$TMP/order.mp4"; rm -f "$ord_out"
  "$EXE" --cli "$ORDDIR" 1 "$ord_out" >/dev/null 2>&1
  ord_fr="$TMP/order_frames"; rm -rf "$ord_fr"; mkdir -p "$ord_fr"
  "$FF" -v error -i "$ord_out" -vf fps=1 "$ord_fr/%02d.png" >/dev/null 2>&1
  ord_res=$(PYTHONIOENCODING=utf-8 python - "$ord_fr" <<'PY'
import sys, glob, os
from PIL import Image
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.png")))
expect = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]  # img1,img2,img10,img20
ok = len(files) == 4
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB"); W, H = im.size
    r, g, b = im.getpixel((W//2, H//2)); er, eg, eb = expect[i]
    ok &= max(abs(r-er), abs(g-eg), abs(b-eb)) < 50
print("ok" if ok else "bad")
PY
)
  if [ "$ord_res" = ok ]; then
    echo "✓ 自然排序套用：img1→img2→img10→img20 依數值順序播放"
  else
    echo "✗ 自然排序：影片內照片順序錯誤（可能退回字典序）"; FAIL=1
  fi
  rm -rf "$ORDDIR" "$ord_out" "$ord_fr"

  # 色階（full-range）保留：JPEG 多為 full-range（0~255）。轉檔若把它壓成
  # limited-range（16~235）或標錯 range 旗標，純黑會發灰、純白會變暗——照片
  # 明顯褪色。以左黑右白的 JPEG 轉檔後驗證黑仍是 0、白仍是 255。
  RNGDIR="$TMP/range"; rm -rf "$RNGDIR"; mkdir -p "$RNGDIR"
  python - "$RNGDIR" <<'PY'
from PIL import Image
import sys
img = Image.new("RGB", (640, 480), (0, 0, 0))
for y in range(480):
    for x in range(320, 640):
        img.putpixel((x, y), (255, 255, 255))
img.save(f"{sys.argv[1]}/1.jpg", quality=98)
img.save(f"{sys.argv[1]}/2.jpg", quality=98)
PY
  rng_out="$TMP/range.mp4"; rm -f "$rng_out"
  "$EXE" --cli "$RNGDIR" 2 "$rng_out" >/dev/null 2>&1
  rng_frame="$TMP/range_frame.png"; rm -f "$rng_frame"
  "$FF" -v error -i "$rng_out" -frames:v 1 -y "$rng_frame" >/dev/null 2>&1
  rng_res=$(PYTHONIOENCODING=utf-8 python - "$rng_frame" <<'PY'
from PIL import Image
import sys
im = Image.open(sys.argv[1]).convert("RGB"); W, H = im.size
black = im.getpixel((W // 2 - 300, H // 2))
white = im.getpixel((W // 2 + 300, H // 2))
print("ok" if black[0] < 20 and white[0] > 235 else f"bad(黑={black[0]},白={white[0]})")
PY
)
  if [ "$rng_res" = ok ]; then
    echo "✓ 色階保留：full-range JPEG 的純黑(0)/純白(255)未被壓縮或標錯"
  else
    echo "✗ 色階：$rng_res（黑發灰或白變暗＝range 被壓成 limited 或旗標錯）"; FAIL=1
  fi
  rm -rf "$RNGDIR" "$rng_out" "$rng_frame"

  # EXIF 方向 × 混合格式：EXIF 直拍照片被混合格式正規化（轉成暫存 PNG）時，
  # 正規化那步（pre_adjust_photo）也必須依 EXIF 轉正，否則直拍照片會躺著。
  # 這條路徑串起「EXIF 自動轉正」與「混合格式正規化」兩個功能的交互作用。
  EXMIX="$TMP/exmix"; rm -rf "$EXMIX"; mkdir -p "$EXMIX"
  python - "$EXMIX" <<'PY'
from PIL import Image
import sys
img = Image.new("RGB", (800, 200), (200, 200, 200))   # 橫向、亮灰
ex = img.getexif(); ex[0x0112] = 6                     # 顯示應轉成直向
img.save(f"{sys.argv[1]}/1.jpg", exif=ex, quality=95)  # jpg：混合格式時會被正規化
PY
  "$FF" -f lavfi -i color=red:s=640x480:d=1 -frames:v 1 -c:v png -y "$EXMIX/2.png" >/dev/null 2>&1
  exmix_out="$TMP/exmix.mp4"; rm -f "$exmix_out"
  "$EXE" --cli "$EXMIX" 1 "$exmix_out" >/dev/null 2>&1   # fps=1 好逐格抽
  exmix_fr="$TMP/exmix_fr"; rm -rf "$exmix_fr"; mkdir -p "$exmix_fr"
  "$FF" -v error -i "$exmix_out" -vf fps=1 "$exmix_fr/%02d.png" >/dev/null 2>&1
  exmix_res=$(PYTHONIOENCODING=utf-8 python - "$exmix_fr" <<'PY'
import sys, glob, os
from PIL import Image
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.png")))
im = Image.open(files[0]).convert("RGB"); W, H = im.size; px = im.load()  # 第 1 張＝EXIF jpg
xs = []; ys = []
for y in range(0, H, 4):
    for x in range(0, W, 4):
        r, g, b = px[x, y]
        if r + g + b > 150: xs.append(x); ys.append(y)
print("portrait" if xs and (max(ys)-min(ys)) > (max(xs)-min(xs)) else "landscape")
PY
)
  if [ "$exmix_res" = portrait ]; then
    echo "✓ EXIF × 混合格式：直拍照片經正規化仍正確轉正為直向"
  else
    echo "✗ EXIF × 混合格式：正規化後 EXIF 方向遺失（輸出 $exmix_res）"; FAIL=1
  fi
  rm -rf "$EXMIX" "$exmix_out" "$exmix_fr"

  # EXIF 方向＝3（180°）：180° 旋轉「不」交換寬高，橫向照片轉正後仍是橫向。
  # 上面已驗 orientation=6（90°→換成直向）；這裡補「不換寬高」那條路徑，
  # 防止日後把 180° 誤當成需要交換寬高（會把橫向照片壓成直向）。
  EX180="$TMP/ex180"; rm -rf "$EX180"; mkdir -p "$EX180"
  python - "$EX180" <<'PY'
from PIL import Image
import sys
img = Image.new("RGB", (800, 200), (200, 200, 200))   # 橫向、亮灰
ex = img.getexif(); ex[0x0112] = 3                     # 180°：轉正後仍應是橫向
img.save(f"{sys.argv[1]}/1.jpg", exif=ex, quality=95)
PY
  ex180_out="$TMP/ex180.mp4"; rm -f "$ex180_out"
  "$EXE" --cli "$EX180" 2 "$ex180_out" >/dev/null 2>&1
  ex180_fr="$TMP/ex180_fr.png"; rm -f "$ex180_fr"
  "$FF" -v error -i "$ex180_out" -frames:v 1 -y "$ex180_fr" >/dev/null 2>&1
  ex180_res=$(PYTHONIOENCODING=utf-8 python - "$ex180_fr" <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB"); W, H = im.size; px = im.load()
xs = []; ys = []
for y in range(0, H, 4):
    for x in range(0, W, 4):
        r, g, b = px[x, y]
        if r + g + b > 150: xs.append(x); ys.append(y)
print("landscape" if xs and (max(xs)-min(xs)) > (max(ys)-min(ys)) else "portrait")
PY
)
  if [ "$ex180_res" = landscape ]; then
    echo "✓ EXIF 方向＝3（180°）：橫向照片轉正後仍為橫向（未誤交換寬高）"
  else
    echo "✗ EXIF 方向＝3：橫向照片被壓成 $ex180_res（180° 誤當成需交換寬高）"; FAIL=1
  fi
  rm -rf "$EX180" "$ex180_out" "$ex180_fr"
else
  echo "↷ 略過 EXIF 方向與排序順序測試（環境未安裝 python3 + Pillow）"
fi

# 錯誤路徑：確認各種不合法輸入都被明確拒絕（非 0 退出碼＋易懂訊息）
mkdir -p "$TMP/empty"
error_case "空資料夾"   "沒有圖片"        "$TMP/empty" 2 "$TMP/x.mp4"
error_case "資料夾不存在" "找不到資料夾"    "$TMP/nope_xyz" 2 "$TMP/x.mp4"
error_case "fps=0"      "介於 1 到 60"    "$IMAGES" 0 "$TMP/x.mp4"
error_case "fps 非數字"  "必須是正整數"    "$IMAGES" abc "$TMP/x.mp4"
rmdir "$TMP/empty" 2>/dev/null; rm -f "$TMP/x.mp4"

rmdir "$TMP" 2>/dev/null
if [ "$FAIL" = 0 ]; then echo "全部通過。"; else echo "有案例失敗。"; exit 1; fi
