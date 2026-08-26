# Parameter Origin Notes（中文版）

English version: `docs/PARAMETER_ORIGIN_NOTES.md`

## 用途

這份文件是用來交代：後來 O256 主線裡幾個重要設定不是憑感覺亂選的。

它不是正式證明文件，而是連接前史探索與後來 StageE protocol 的歷史備註。

## 為什麼最後固定到 256x256？

較早期的 pixel-bridge 分支曾明確測試：

- 如果把輸入解析度放大到 256x256
- 對應 latent grid 也變大
- 是否能把較長 message 的 learnability 從失敗區拉回來

關鍵歷史文件是：

- `Pixel_bridge_256_msglen32_msgloss20_identity/md/CAPACITY_SCALING_256_MSG32_IDENTITY.md`

這條分支存在的原因，就是因為更早的低容量設定顯示 watermark signal 太弱，
所以必須確認問題是不是 partly capacity-driven，而不只是 bridge 本身的錯。

後來 paper-side 分支便把 256x256 標準化，並把它當作較乾淨、較穩定的研究尺度。

## 為什麼最後固定到 32-bit message？

較早期文字封存裡有明確的 payload 診斷：

- 64-bit baseline
- 32-bit diagnostic
- 16-bit diagnostic
- 8-bit diagnostic

這些分支的目的就是確認：原本 payload 是否對 latent watermark signal 來說太重。

後來形成的邏輯是：

- 64 bits 對早期設定太重
- 8 bits 和 16 bits 很適合做診斷，但太小，不夠代表預期的實際目標
- 當容量與表示方式改善後，32 bits 成為較實用的中間點

因此後來的 paper-side 與 O256 線，才把 32 bits 當成 practical target payload，
而不是只把它當成玩具型診斷設定。

## 為什麼會用 `message_map_mode: bitplane`？

bitplane 不是隨便加的，而是因為早期觀察到：

- 舊的 linear message map 會太早把 32 個 bit 混在一起
- 某些 bit position 會反覆變成 weak bits

關鍵歷史文件是：

- `Pixel_bridge_256_msglen32_r30_msgmap_bitplane_identity/md/MESSAGE_MAP_BITPLANE_ABLATION.md`

它背後的想法是：

- linear map 讓所有 bit 一開始就被混合
- 某些 bit 比較容易在這個過程裡被稀釋或變不穩
- bitplane 則能先保留 bit 身分，再讓 CNN 學怎麼混

所以 bitplane 應該被理解成有歷史根據的結構選擇，而不是小參數微調。

## 為什麼 `message_loss_weight = 30`？

paper-side 清理過的分支已明確把最佳當時設計記錄成：

- image size 256
- message length 32
- message loss ratio `r=30`
- bitplane message map
- non-geometry attack profile

這點出現在：

- `paper/Pixel_bridge_256_msglen32_paper_bitplane_nongeom/README_PAPER_PROJECT.md`

從歷史角度來看，後來固定在 `r=30`，可以理解成在 256 / 32 / bitplane 這條線
部分跨過容量瓶頸後，為了維持較強 watermark pressure 而留下來的設定。

## 為什麼會用 `identity_gn_sp_webp_bc`？

這個 non-geometric attack profile 不是因為它列出越多攻擊越好，而是有明確原因：

- bitplane identity-only 已經顯示 clean decoding 可以很強
- 但部分非幾何攻擊仍偏弱
- Crop / Rotate 則更像 geometry / alignment 問題
- 所以應先把 non-geometry robustness 單獨處理，而不是一次混進 geometry 問題

關鍵歷史文件是：

- `Pixel_bridge_256_msglen32_r30_bitplane_gn_sp_webp_bc_identity/md/BITPLANE_NONGEOM_PROFILE.md`

這個思路後來也呼應了 O256 主線中更細緻的 attack scheduling 設計。

## 為什麼早期 paper-side 工作仍然值得保留？

後來的 StageE protocol 並不是直接複製早期 paper-side 分支。

但這些較早分支仍然很重要，因為它們幫助解釋：

- 為什麼最後固定 256x256
- 為什麼 32 bits 成為實用目標
- 為什麼某些分支偏好 bitplane 而不是舊 linear map
- 為什麼非幾何 robustness 往往先被單獨處理

因此即使不保留它們的 checkpoint，它們仍然應該作為前史備註被保留在交接文件中。
