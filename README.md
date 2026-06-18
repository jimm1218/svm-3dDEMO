# SVM Kernel Trick 互動教學專案
**連結：https://svm-3ddemo-nqmra2ma3grxvyrby5ryid.streamlit.app/**

這是一個以 Streamlit、Plotly、Scikit-learn 與 Manim 製作的 SVM 支援向量機互動教學專案。教學頁會先用 Manim 影片說明 Kernel Trick 的直覺，再透過互動式 2D/3D 圖表展示不同資料集、Z 軸公式與 SVM kernel 對分類邊界的影響。
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e83d87d4-d446-4ca8-a777-4c955fed694d" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/09540f35-c216-431f-a4ba-26e94a7a164b" />




主要入口只有一個：

```text
svm_tutorial.py
```

Manim 動畫腳本則保留為：

```text
svm_animation.py
```

## 功能特色

- 顯示 Manim 產生的 Kernel Trick 概念影片。
- 支援互動選擇資料集：同心圓、半月形、高斯團。
- 支援互動選擇 Z 軸特徵公式：`x² + y²`、`x × y`、`x² - y²`、`x + y`、`sin(πx) + cos(πy)`。
- 支援選擇 SVM kernel：`linear`、`rbf`、`poly`、`sigmoid`。
- 可調整模型參數：`C`、`Gamma`、`degree`、`coef0`、Noise、是否標準化。
- 顯示 2D 決策邊界。
- 顯示 3D 特徵空間。
- 顯示 support vectors。
- 使用 `linear` kernel 時，3D 圖會額外顯示分類超平面。

## 資料夾架構

以下架構以 `D:\data\practice\svm` 為根目錄：

```text
D:\data\practice\svm
├── README.md
├── prompt.md
├── svm_tutorial.py
├── svm_animation.py
└── media/
    └── videos/
        └── svm_animation/
            └── 480p15/
                └── SvmKernelAnimation.mp4
```

補充：如果你從 `D:\data` 執行 Manim，影片可能會產生在下面這個位置，`svm_tutorial.py` 也會自動嘗試讀取：

```text
D:\data\media\videos\svm_animation\480p15\SvmKernelAnimation.mp4
```

## 檔案說明

| 檔案 | 說明 |
| --- | --- |
| `svm_tutorial.py` | 主要 Streamlit 教學網頁，整合影片、文字說明、互動參數、2D/3D 圖表與 kernel 比較。 |
| `svm_animation.py` | Manim 動畫腳本，用來產生 Kernel Trick 概念影片。 |
| `README.md` | 專案介紹、資料夾架構與執行方式。 |
| `prompt.md` | 專案需求、開發提示詞與維護方向。 |
| `media/videos/.../SvmKernelAnimation.mp4` | Manim render 後的影片檔，教學頁會自動讀取。 |

## 環境需求

建議使用 Python 3.10 以上。需要的主要套件：

- `streamlit`
- `numpy`
- `plotly`
- `scikit-learn`
- `manim`

安裝指令：

```powershell
python -m pip install streamlit numpy plotly scikit-learn manim
```

## 執行方式

請先切換到專案資料夾：

```powershell
cd D:\data\practice\svm
```

### 1. 產生 Manim 動畫

```powershell
python -m manim -ql svm_animation.py SvmKernelAnimation
```

常用畫質參數：

```text
-ql  low quality，適合快速預覽
-qm  medium quality
-qh  high quality
```

### 2. 啟動 Streamlit 教學網頁

```powershell
python -m streamlit run svm_tutorial.py --server.port 8501
```

啟動後打開：

```text
http://localhost:8501
```

## 從 D:\data 執行的替代指令

如果你的終端機目前在 `D:\data`，也可以使用：

```powershell
python -m manim -ql practice\svm\svm_animation.py SvmKernelAnimation
python -m streamlit run practice\svm\svm_tutorial.py --server.port 8501
```

## 建議教學流程

1. 先觀看頁面上方的 Manim 動畫，理解 2D 資料升維到 3D 的概念。
2. 選擇 `同心圓 (Circles)`。
3. 將 Z 軸公式設為 `x² + y²`。
4. 將 kernel 設為 `linear`，觀察 3D 超平面如何切開資料。
5. 切換到 `rbf`，觀察 2D 決策邊界如何變得更彈性。
6. 調整 `Gamma` 與 `C`，觀察 underfitting 與 overfitting 的變化。

## 核心概念

SVM 會尋找能最大化分類間隔的決策邊界。當資料在原始 2D 平面中無法用直線分開時，可以透過特徵轉換把資料映射到更高維度。

例如同心圓資料在 2D 中很難用直線切開，但透過：

```text
z = x² + y²
```

內圈與外圈會被抬到不同高度。此時在 3D 空間中，SVM 可能只需要一個平面就能完成分類。

RBF kernel 則不需要手動建立所有高維特徵，而是透過 kernel function 計算資料點在高維空間中的相似度，因此能形成非線性決策邊界。

## 常見問題

### 開啟 http://localhost:8501 是空白

請確認你執行的是：

```powershell
python -m streamlit run svm_tutorial.py --server.port 8501
```

不要用 Streamlit 執行 `svm_animation.py`，因為它是 Manim 腳本，不是網頁。

### 看不到影片

請先執行：

```powershell
python -m manim -ql svm_animation.py SvmKernelAnimation
```

如果你是從 `D:\data` 執行，則用：

```powershell
python -m manim -ql practice\svm\svm_animation.py SvmKernelAnimation
```

### manim 指令無法辨識

請改用：

```powershell
python -m manim -ql svm_animation.py SvmKernelAnimation
```

這種寫法不需要 `manim.exe` 在 PATH 裡。

## 後續擴充方向

- 新增 spiral dataset。
- 新增 confusion matrix、precision、recall、F1-score。
- 新增參數組合匯出功能。
- 新增 `requirements.txt`。
- 新增教學步驟導覽模式。
