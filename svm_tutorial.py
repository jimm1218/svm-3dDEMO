from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATHS = [
    Path(__file__).resolve().parent / "media" / "videos" / "svm_animation" / "480p15" / "SvmKernelAnimation.mp4",
    ROOT / "media" / "videos" / "svm_animation" / "480p15" / "SvmKernelAnimation.mp4",
    Path(__file__).resolve().parent / "SvmKernelAnimation.mp4",
    ROOT / "SvmKernelAnimation.mp4",
]


st.set_page_config(page_title="SVM Pro 互動實驗室", layout="wide", initial_sidebar_state="expanded")

# 注入自訂 CSS，打造 v2.0 Pro 專業科技、毛玻璃與進階動畫風格
st.markdown(
    """
    <style>
        /* 匯入現代化字體 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Noto Sans TC', sans-serif;
        }
        /* 主區塊上方間距微調，讓畫面更緊湊 */
        .block-container {padding-top: 2.5rem;}
        /* 科技感 Metrics (儀表板數字微發光) */
        [data-testid="stMetricValue"] {
            color: #0ea5e9;
            font-weight: 700;
            text-shadow: 0px 2px 4px rgba(14, 165, 233, 0.2);
        }
        /* 全局淡入動畫 */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stTabs [data-baseweb="tab-panel"] {
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        /* 毛玻璃與懸浮特效 (Glassmorphism & Hover) */
        div[data-testid="stVerticalBlock"] > div[style*="border"] {
            background: rgba(15, 23, 42, 0.4) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
            border-radius: 16px !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px 0 rgba(14, 165, 233, 0.25) !important;
        }
        /* Banner 樣式 */
        @keyframes gradientFade {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero-container {
            padding: 2.5rem;
            border-radius: 20px;
            background: linear-gradient(-45deg, rgba(6,182,212,0.1), rgba(139,92,246,0.15), rgba(236,72,153,0.1));
            background-size: 400% 400%;
            animation: gradientFade 10s ease infinite;
            border: 1px solid rgba(139,92,246,0.3);
            box-shadow: 0 10px 30px -10px rgba(139,92,246,0.2);
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero-title {
            background: linear-gradient(to right, #0ea5e9, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        /* 美化 Tabs 標籤頁 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.2rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 渲染 Animated Hero Banner
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">🚀 SVM Kernel Trick Pro 實驗室</h1>
        <p style="font-size: 1.2rem; color: #94a3b8; margin-top: 0;">電影級 3D 視覺化 × 全方位模型評估，探索支援向量機的升維魔法！✨</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 建立主要導覽 Tabs
tab_video, tab_lab, tab_guide = st.tabs(["🎬 概念動畫解析", "🎛️ 核心互動實驗室", "💡 理論與學習指南"])

with tab_video:
    video_path = next((path for path in VIDEO_PATHS if path.exists()), None)

    if video_path:
        # 使用 columns 來縮小影片在網頁上的視窗寬度比例
        _, center_col, _ = st.columns([1, 2, 1])
        center_col.video(str(video_path))
    else:
        st.warning("找不到 Manim 影片。請先在終端機產生影片，再重新整理頁面。")
        st.code(
            "python -m manim -ql svm_animation.py SvmKernelAnimation",
            language="powershell",
        )
    
    with st.expander("影片重點回顧", expanded=True):
        st.markdown(
            """
            - **核心概念**：原本在 2D 平面中無法用直線分開的資料（如同心圓），可以透過一個數學轉換（例如 `z = x² + y²`）被「抬」到 3D 空間。
            - **神奇之處**：在更高維度的空間中，這些資料點可能就變得可以用一個簡單的平面（稱為「超平面」）來分開了。
            - **這就是 Kernel Trick**：它讓我們能夠在原始空間中得到非線性的決策邊界，但計算上卻像是在高維空間中做線性分類一樣有效率。
            """
        )

st.sidebar.header("⚙️ 互動參數")

with st.sidebar.container(border=True):
    dataset_type = st.selectbox(
        "① 選擇資料集",
        ["同心圓 (Circles)", "半月形 (Moons)", "螺旋形 (Spiral)", "高斯團 (Blobs)"],
    )

    z_transform = st.selectbox(
        "② 選擇 Z 軸轉換公式 (用於 3D 視覺化)",
        [
            "x² + y²",
            "x × y",
            "x² - y²",
            "x + y",
            "sin(πx) + cos(πy)",
        ],
    )

    kernel = st.selectbox(
        "③ 選擇 SVM Kernel",
        ["linear", "rbf", "poly", "sigmoid"],
        help="linear 適合觀察升維後的超平面；rbf/poly/sigmoid 可以形成更彈性的非線性邊界。",
    )

    noise_level = st.slider("資料雜訊 Noise", 0.0, 0.5, 0.08, 0.01)
    c_param = st.slider("C 正規化參數", 0.01, 20.0, 1.0, 0.1)
    standardize = st.checkbox("標準化特徵", value=True)

gamma_param = "scale"
degree_param = 3
coef0_param = 0.0

if kernel in {"rbf", "poly", "sigmoid"}:
    with st.sidebar.expander("🛠️ 進階 Kernel 專屬參數", expanded=True):
        gamma_param = st.select_slider(
            "Gamma",
            options=["scale", "auto", 0.01, 0.1, 1.0, 10.0, 50.0],
            value="scale",
            help="Gamma 越大，單一資料點影響範圍越小，邊界通常越彎曲 (易 overfitting)。",
        )

        if kernel == "poly":
            degree_param = st.slider("Polynomial degree", 2, 6, 3, 1)
            coef0_param = st.slider("coef0", -2.0, 2.0, 0.0, 0.1)
        elif kernel == "sigmoid":
            coef0_param = st.slider("coef0", -2.0, 2.0, 0.0, 0.1)


@st.cache_data
def make_dataset(kind, noise):
    if kind == "同心圓 (Circles)":
        return make_circles(n_samples=400, factor=0.35, noise=noise, random_state=42)
    if kind == "半月形 (Moons)":
        return make_moons(n_samples=400, noise=noise, random_state=42)
    if kind == "螺旋形 (Spiral)":
        n_points = 200
        n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.randn(n_points, 1) * noise * 2.5
        d1y = np.sin(n) * n + np.random.randn(n_points, 1) * noise * 2.5
        d2x = np.cos(n) * n + np.random.randn(n_points, 1) * noise * 2.5
        d2y = -np.sin(n) * n + np.random.randn(n_points, 1) * noise * 2.5
        # 組合並縮放，使其範圍與其他資料集相近
        X = np.vstack((np.hstack((d1x, d1y)), np.hstack((d2x, d2y)))) / 5.0
        y = np.hstack((np.zeros(n_points), np.ones(n_points)))
        return X, y.astype(int)
    return make_blobs(n_samples=400, centers=2, n_features=2, cluster_std=1.35, random_state=42)


def transform_z(points, formula):
    x = points[:, 0]
    y = points[:, 1]

    if formula == "x² + y²":
        return x**2 + y**2, "z = x² + y²"
    if formula == "x × y":
        return x * y, "z = x × y"
    if formula == "x² - y²":
        return x**2 - y**2, "z = x² - y²"
    if formula == "x + y":
        return x + y, "z = x + y"
    return np.sin(np.pi * x) + np.cos(np.pi * y), "z = sin(πx) + cos(πy)"


X_raw, y = make_dataset(dataset_type, noise_level)
z_raw, z_label = transform_z(X_raw, z_transform)
X_3d_raw = np.column_stack([X_raw, z_raw])

if standardize:
    scaler = StandardScaler()
    X_model = scaler.fit_transform(X_3d_raw)
else:
    scaler = None
    X_model = X_3d_raw

clf = SVC(
    kernel=kernel,
    C=c_param,
    gamma=gamma_param,
    degree=degree_param,
    coef0=coef0_param,
)
clf.fit(X_model, y)
preds = clf.predict(X_model)
accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds, average='macro', zero_division=0)
recall = recall_score(y, preds, average='macro', zero_division=0)
f1 = f1_score(y, preds, average='macro', zero_division=0)

with tab_lab:
    st.markdown("#### 📟 模型效能觀測站")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Accuracy (準確率)", f"{accuracy * 100:.1f}%")
    metric_cols[1].metric("Precision (精確率)", f"{precision * 100:.1f}%")
    metric_cols[2].metric("Recall (召回率)", f"{recall * 100:.1f}%")
    metric_cols[3].metric("F1-Score", f"{f1 * 100:.1f}%")
    metric_cols[4].metric("Support Vectors", len(clf.support_))
    metric_cols[5].metric("Z 特徵公式", z_label.replace("z = ", ""))

    with st.expander("🔍 展開詳細評估分析 (Confusion Matrix)", expanded=False):
        st.markdown("下方矩陣顯示了模型對於每個類別的預測分佈情況，對角線上的數字代表預測正確的數量。")
        cm = confusion_matrix(y, preds)
        cm_labels = ['類別 0', '類別 1']
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=cm_labels,
            y=cm_labels,
            colorscale=[[0.0, '#0f172a'], [1.0, '#3b82f6']],
            showscale=False
        )
        fig_cm.update_layout(
            height=250,
            margin=dict(l=10, r=10, b=20, t=20),
            xaxis_title="預測標籤",
            yaxis_title="真實標籤",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    plot_tab_2d, plot_tab_3d = st.tabs(["📊 2D 決策邊界", "🧊 3D 特徵空間"])

    with plot_tab_2d:
        st.markdown("此處為模型在原始 x-y 平面上的決策邊界。可觀察不同 Kernel 如何形成非線性切分。")
        x_min, x_max = X_raw[:, 0].min() - 0.45, X_raw[:, 0].max() + 0.45
        y_min, y_max = X_raw[:, 1].min() - 0.45, X_raw[:, 1].max() + 0.45
        grid_size = 180
        grid_x = np.linspace(x_min, x_max, grid_size)
        grid_y = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
        grid_z, _ = transform_z(grid_2d, z_transform)
        grid_3d_raw = np.column_stack([grid_2d, grid_z])
        grid_model = scaler.transform(grid_3d_raw) if scaler is not None else grid_3d_raw
        decision = clf.decision_function(grid_model).reshape(xx.shape)

        # 確保決策邊界等高線的中心點 (0) 能完美對齊透明色，避免泛白
        z_bound = max(abs(decision.min()), abs(decision.max()))
        if z_bound == 0:
            z_bound = 1.0  # 防止邊界皆為 0 時引發繪圖錯誤
        fig_2d = go.Figure()
        fig_2d.add_trace(
            go.Contour(
                x=grid_x, y=grid_y, z=decision,
                zmin=-z_bound, zmax=z_bound,
                colorscale=[[0.0, '#3b82f6'], [0.5, 'rgba(255,255,255,0)'], [1.0, '#f43f5e']],
                opacity=0.5, contours=dict(showlines=False), showscale=False, name="Decision score",
            )
        )
        fig_2d.add_trace(
            go.Contour(
                x=grid_x, y=grid_y, z=decision,
                contours=dict(start=0, end=0, size=1, coloring="lines"),
                line=dict(width=3, color="#111827"), showscale=False, name="Decision boundary",
            )
        )
        fig_2d.add_trace(
            go.Scatter(
                x=X_raw[:, 0], y=X_raw[:, 1], mode="markers",
                marker=dict(size=8, color=y, colorscale=[[0.0, '#3b82f6'], [1.0, '#f43f5e']], line=dict(width=1, color="white")), name="Data",
            )
        )
        fig_2d.add_trace(
            go.Scatter(
                x=X_raw[clf.support_, 0], y=X_raw[clf.support_, 1], mode="markers",
                marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2.5, color="#fde047")), name="Support vectors",
            )
        )
        fig_2d.update_layout(
            height=600, margin=dict(l=10, r=10, b=10, t=10),
            xaxis_title="x", yaxis_title="y",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
        fig_2d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
        st.plotly_chart(fig_2d, use_container_width=True)

    with plot_tab_3d:
        st.markdown(f"此圖僅為視覺化輔助，將資料點依 `{z_label}` 抬升。`linear` kernel 會額外顯示超平面與 Margin 切面。")
        fig_3d = go.Figure()
        
        # 增加底部投影特效 (projection)
        fig_3d.add_trace(
            go.Scatter3d(
                x=X_raw[:, 0], y=X_raw[:, 1], z=z_raw, mode="markers",
                marker=dict(size=6, color=y, colorscale=[[0.0, '#3b82f6'], [1.0, '#f43f5e']], opacity=0.95, line=dict(width=0.5, color="white")),
                name="3D transformed data", projection=dict(z=dict(show=True, opacity=0.1))
            )
        )
        # 支持向量發光特效 (加大光暈)
        fig_3d.add_trace(
            go.Scatter3d(
                x=X_raw[clf.support_, 0], y=X_raw[clf.support_, 1], z=z_raw[clf.support_], mode="markers",
                marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=4, color="#fde047")), name="Support vectors",
            )
        )

        if kernel == "linear":
            if scaler is None:
                w = clf.coef_[0]
                b = clf.intercept_[0]
            else:
                w_scaled = clf.coef_[0]
                scale = scaler.scale_
                mean = scaler.mean_
                w = w_scaled / scale
                b = clf.intercept_[0] - np.sum(w_scaled * mean / scale)

            if abs(w[2]) > 1e-8:
                x_min, x_max = X_raw[:, 0].min() - 0.45, X_raw[:, 0].max() + 0.45
                y_min, y_max = X_raw[:, 1].min() - 0.45, X_raw[:, 1].max() + 0.45
                plane_x, plane_y = np.meshgrid(np.linspace(x_min, x_max, 32), np.linspace(y_min, y_max, 32))
                
                # 主決策面與 Margin 平面 (+1, -1)
                plane_z = (-w[0] * plane_x - w[1] * plane_y - b) / w[2]
                plane_z_plus = (-w[0] * plane_x - w[1] * plane_y - b + 1) / w[2]
                plane_z_minus = (-w[0] * plane_x - w[1] * plane_y - b - 1) / w[2]
                
                z_pad = max(0.5, (z_raw.max() - z_raw.min()) * 0.25)
                plane_z = np.clip(plane_z, z_raw.min() - z_pad, z_raw.max() + z_pad)
                plane_z_plus = np.clip(plane_z_plus, z_raw.min() - z_pad, z_raw.max() + z_pad)
                plane_z_minus = np.clip(plane_z_minus, z_raw.min() - z_pad, z_raw.max() + z_pad)
                
                # 繪製主決策面 (降低反光以避免被反白，並改用深色漸層)
                fig_3d.add_trace(
                    go.Surface(
                        x=plane_x, y=plane_y, z=plane_z, opacity=0.7,
                        colorscale=[[0.0, '#3b82f6'], [1.0, '#f43f5e']], showscale=False, name="Hyperplane",
                        contours=dict(
                            x=dict(show=True, color='rgba(255,255,255,0.1)', width=1),
                            y=dict(show=True, color='rgba(255,255,255,0.1)', width=1)
                        ),
                        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.9, specular=0.05, fresnel=0.1)
                    )
                )
                # 繪製上下 Margin 切面 (改用對應的類別色彩，避免原本的灰色太像反白)
                fig_3d.add_trace(
                    go.Surface(
                        x=plane_x, y=plane_y, z=plane_z_plus, opacity=0.12,
                        colorscale=[[0.0, '#3b82f6'], [1.0, '#3b82f6']], showscale=False, name="+1 Margin"
                    )
                )
                fig_3d.add_trace(
                    go.Surface(
                        x=plane_x, y=plane_y, z=plane_z_minus, opacity=0.12,
                        colorscale=[[0.0, '#f43f5e'], [1.0, '#f43f5e']], showscale=False, name="-1 Margin"
                    )
                )
            else:
                st.info("目前 linear 模型的超平面幾乎不依賴 Z 軸，所以 3D 平面不容易顯示。")

        # 解決 3D 偏離中心與按鈕點擊後字體反白問題：
        # 1. 將按鈕方向設為水平 (left)，並移至畫面正上方居中 (x=0.5, y=1.05)，釋放左右空間讓 3D 圖自然置中
        # 2. 設定 activebgcolor 為亮藍色，避免預設的白色底色吃掉白色字體
        fig_3d.update_layout(
            height=700, margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title=z_label,
                camera=dict(eye=dict(x=1.35, y=1.35, z=0.8)), # 稍微拉近一點視角
                aspectratio=dict(x=1, y=1, z=0.8),
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                zaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            updatemenus=[
                dict(
                    type="buttons", direction="left", 
                    x=0.5, xanchor="center", y=1.05, yanchor="bottom", 
                    showactive=True,
                    buttons=[
                        dict(label="🎥 預設視角", method="relayout", args=[{"scene.camera.eye": dict(x=1.45, y=1.45, z=0.85)}]),
                        dict(label="🛸 俯視投影", method="relayout", args=[{"scene.camera.eye": dict(x=0, y=0, z=2.2)}]),
                        dict(label="👁️ 側視剖面", method="relayout", args=[{"scene.camera.eye": dict(x=2.2, y=0, z=0)}]),
                    ],
                    bgcolor="rgba(15, 23, 42, 0.8)",
                    font=dict(color="white"),
                    bordercolor="#334155"
                )
            ]
        )
        st.plotly_chart(fig_3d, use_container_width=True)

with tab_guide:
    st.markdown("### 💡 理論與觀察方向")
    st.markdown("在機器學習中，許多分類器擅長處理「線性可分」的資料。但真實世界的資料往往是複雜、非線性的。**Kernel Trick** 就是 SVM 能處理這些複雜資料的關鍵。")
    
    st.info("💡 **建議實驗流程**：\n"
            "1. 選擇 `同心圓 (Circles)` 資料集。\n"
            "2. 將 Z 軸公式設為 `x² + y²`。\n"
            "3. 將 kernel 設為 `linear`，觀察 **3D 圖**中的分類超平面如何漂亮地切開資料。\n"
            "4. 接著將 kernel 切換到 `rbf`，比較 **2D 圖**中的決策邊界如何變得更彈性，並適應了資料形狀。\n"
            "5. 嘗試調整 `Gamma` 與 `C` 參數，觀察 2D 邊界如何從平滑 (underfitting) 變為過度扭曲 (overfitting)。"
    )
    
    st.markdown("#### 各 Kernel 特性比較")
    st.markdown(
        """
        - 📏 **`linear` kernel**: 最適合用來觀察「升維後用一個平面切開資料」的概念。
        - 🌐 **`rbf` kernel**: 最常用也最有彈性的非線性 kernel，通常能產生平滑的邊界。
        - 📈 **`poly` kernel**: 多項式 kernel，可以透過 `degree` 參數控制邊界的彎曲複雜度。
        - 🧠 **`sigmoid` kernel**: 形狀類似神經網路啟發的 kernel，在某些參數下可能比較不穩定。
        """
    )
