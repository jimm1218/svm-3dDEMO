# Prompt 與開發需求紀錄

專案路徑：

```text
D:\data\practice\svm
```

本文件記錄 SVM Kernel Trick 互動教學專案的需求、開發提示詞與維護方向。未來若要請 AI 或開發者擴充專案，可以直接參考本文件。

## 專案目標

建立一個專業、清楚、可互動的 SVM Kernel Trick 教學網頁。使用者可以先透過 Manim 動畫理解 2D 資料如何升維到 3D，再透過互動圖表觀察不同資料集、Z 軸公式、kernel 與模型參數如何影響分類邊界。

## 最終整合需求

- 主要教學入口只保留 `svm_tutorial.py`。
- Manim 動畫腳本保留 `svm_animation.py`。
- 文件保留 `README.md` 與 `prompt.md`。
- 不需要獨立的 `svm_tutorial_advanced.py` 或 `svm_animation_app.py`。

`svm_tutorial.py` 必須包含：

- Manim 影片展示區。
- SVM Kernel Trick 文字說明。
- 資料集選擇。
- Z 軸公式選擇。
- SVM kernel 選擇。
- 模型參數調整。
- 2D 決策邊界圖。
- 3D 特徵空間圖。
- Support vectors 顯示。

## 技術選型

| 技術 | 用途 |
| --- | --- |
| Streamlit | 建立互動式教學網頁。 |
| Plotly | 繪製 2D contour 與 3D scatter/surface 圖。 |
| Scikit-learn | 產生資料集並訓練 SVM 模型。 |
| Manim | 製作 Kernel Trick 概念動畫。 |
| NumPy | 數值運算與網格資料產生。 |

## 互動控制需求

### 資料集

- `同心圓 (Circles)`
- `半月形 (Moons)`
- `高斯團 (Blobs)`

### Z 軸公式

- `x² + y²`
- `x × y`
- `x² - y²`
- `x + y`
- `sin(πx) + cos(πy)`

### SVM Kernel

- `linear`
- `rbf`
- `poly`
- `sigmoid`

### 模型參數

- `C`
- `Gamma`
- `degree`
- `coef0`
- `Noise`
- 是否標準化特徵

## 執行命令

請先切換到專案資料夾：

```powershell
cd D:\data\practice\svm
```

產生動畫：

```powershell
python -m manim -ql svm_animation.py SvmKernelAnimation
```

啟動教學網頁：

```powershell
python -m streamlit run svm_tutorial.py --server.port 8501
```

開啟網址：

```text
http://localhost:8501
```

## 可重用開發 Prompt

```text
請協助我維護 D:\data\practice\svm 內的 SVM Kernel Trick 互動教學專案。

專案使用 Streamlit、Plotly、Scikit-learn、NumPy 與 Manim。
主要教學入口是 svm_tutorial.py，Manim 動畫腳本是 svm_animation.py。

請保持專案結構簡潔，不要新增不必要的入口檔。
svm_tutorial.py 必須包含：
1. Manim 影片展示區。
2. SVM Kernel Trick 文字說明。
3. 資料集選擇：同心圓、半月形、高斯團。
4. Z 軸公式選擇：x² + y²、x × y、x² - y²、x + y、sin(πx) + cos(πy)。
5. Kernel 選擇：linear、rbf、poly、sigmoid。
6. 可調參數：C、Gamma、degree、coef0、Noise、標準化。
7. 2D 決策邊界圖。
8. 3D 特徵空間圖。
9. Support vectors 顯示。

請優先保持教學清楚、互動穩定、圖表易讀。
如果要修改檔案，請先檢查目前程式碼，避免刪除既有功能。
```

## 未來擴充 Prompt

```text
請在現有 SVM 互動教學頁中新增模型評估區塊。

需求：
- 顯示 accuracy、precision、recall、F1-score。
- 顯示 confusion matrix。
- 評估區塊放在 2D/3D 圖表下方。
- 不要移除現有的影片、Z 軸公式選擇、kernel 選擇與互動圖表。
- 保持 Streamlit 頁面清楚、適合教學展示。
```

```text
請在現有 SVM 互動教學頁中新增 spiral dataset。

需求：
- 在資料集選單中新增 Spiral。
- 使用 NumPy 產生兩類螺旋資料。
- Noise slider 仍需有效。
- 2D 決策邊界與 3D 特徵空間都要能正常顯示。
- 不要新增新的入口檔，請直接修改 svm_tutorial.py。
```

## 維護注意事項

- `svm_animation.py` 是 Manim 腳本，不應用 Streamlit 直接執行。
- `svm_tutorial.py` 是唯一教學網頁入口。
- 如果影片不存在，頁面應顯示清楚的產生影片指令。
- 若要新增套件，建議同步建立或更新 `requirements.txt`。
- 修改互動圖表時，需確認 `linear`、`rbf`、`poly`、`sigmoid` 四種 kernel 都能正常運作。
