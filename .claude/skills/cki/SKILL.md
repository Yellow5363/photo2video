---
name: cki
description: Check issue — 查詢 GitHub 上 photo2video 的最新問題回報並開始解決。使用者輸入 /cki 時觸發。
---

# /cki — 檢查並處理 GitHub 問題回報

## 步驟

1. **列出未處理的 issue**（repo 固定為 `yellow5363/photo2video`）：

   ```
   gh issue list --repo yellow5363/photo2video --state open --limit 20
   ```

   - 沒有任何 open issue 時，直接回報「目前沒有新的問題回報」並結束。

2. **逐一閱讀 issue 內容**（由最新的開始）：

   ```
   gh issue view <編號> --repo yellow5363/photo2video --comments
   ```

   - 使用者的回報多半來自程式內建的「回報問題」按鈕，內文會帶版本號與錯誤訊息／閃退紀錄（panic 訊息與原始碼位置）。

3. **向使用者簡短彙報**目前有哪些問題（編號、標題、重點錯誤訊息），然後**直接開始解問題**，不用等使用者確認：
   - 依錯誤訊息或 panic 位置（file:line）定位程式碼，分析根因。
   - 修好後 `cargo check` 與 `cargo clippy` 驗證，並依專案慣例自動 commit（訊息用繁體中文）。

4. **處理完成後回覆 issue**：

   ```
   gh issue comment <編號> --repo yellow5363/photo2video --body "<修正說明（繁體中文、白話）>"
   ```

   - 已修正的問題先留言說明修法與預計發版版本；**要等修正實際發版後**（使用者說「發版」）才用 `gh issue close` 關閉。
   - 若 issue 資訊不足無法重現，留言向回報者詢問細節，不要關閉。

## 注意

- 一次處理多個 issue 時，逐一修、逐一 commit，不要混在同一個 commit。
- 若問題出在使用者環境（例如缺 ffmpeg、磁碟空間不足），程式端能防呆就順手加上防呆再回覆。
