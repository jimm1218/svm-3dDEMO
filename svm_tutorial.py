from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATHS = [
    Path(__file__).resolve().parent / "media" / "videos" / "svm_animation" / "480p15" / "SvmKernelAnimation.mp4",
    ROOT / "media" / "videos" / "svm_animation" / "480p15" / "SvmKernelAnimation.mp4",
    Path(__file__).resolve().parent / "SvmKernelAnimation.mp4",
    ROOT / "SvmKernelAnimation.mp4",
]


st.set_page_config(page_title="SVM 互動教學", layout="wide")

st.title("SVM 支援向量機互動教學")
st.markdown(
    """
這份教學把 Manim 概念動畫、Z 軸升維公式、以及不同 SVM kernel 的互動比較整合在同一個頁面。
先看影片理解 kernel trick 的直覺，再用下方圖表調參數觀察決策邊界如何改變。
"""
)

st.markdown("## 1. Kernel Trick 概念動畫")
video_path = next((path for path in VIDEO_PATHS if path.exists()), None)

if video_path:
    # 使用 columns 來縮小影片在網頁上的視窗寬度比例 (左右留白)
    _, center_col, _ = st.columns([1, 2, 1])
    center_col.video(str(video_path))
else:
    st.warning("找不到 Manim 影片。請先在終端機產生影片，再重新整理頁面。")
    st.code(
        "python -m manim -ql practice\\svm\\svm_animation.py SvmKernelAnimation",
        language="powershell",
    )

st.markdown(
    """
影片中的重點是：原本在 2D 平面中不好用直線分開的資料，可以透過特徵轉換被放到 3D 空間。
在高維空間中，SVM 可能只需要一個平面就能把不同類別分開。
"""
)

st.markdown("---")
st.markdown("## 2. 互動實驗")

st.sidebar.header("互動參數")

dataset_type = st.sidebar.selectbox(
    "資料集形狀",
    ["同心圓 (Circles)", "半月形 (Moons)", "高斯團 (Blobs)"],
)

z_transform = st.sidebar.selectbox(
    "Z 軸特徵轉換公式",
    [
        "x² + y²",
        "x × y",
        "x² - y²",
        "x + y",
        "sin(πx) + cos(πy)",
    ],
)

kernel = st.sidebar.selectbox(
    "SVM Kernel",
    ["linear", "rbf", "poly", "sigmoid"],
    help="linear 適合觀察升維後的超平面；rbf/poly/sigmoid 可以形成更彈性的非線性邊界。",
)

noise_level = st.sidebar.slider("資料雜訊 Noise", 0.0, 0.5, 0.08, 0.01)
c_param = st.sidebar.slider("C 正規化參數", 0.01, 20.0, 1.0, 0.1)
standardize = st.sidebar.checkbox("標準化特徵", value=True)

gamma_param = "scale"
degree_param = 3
coef0_param = 0.0

if kernel in {"rbf", "poly", "sigmoid"}:
    gamma_param = st.sidebar.select_slider(
        "Gamma",
        options=["scale", 0.01, 0.1, 1.0, 10.0],
        value="scale",
        help="Gamma 越大，單一資料點影響範圍越小，邊界通常越彎曲。",
    )

if kernel == "poly":
    degree_param = st.sidebar.slider("Polynomial degree", 2, 6, 3, 1)
    coef0_param = st.sidebar.slider("coef0", -2.0, 2.0, 0.0, 0.1)
elif kernel == "sigmoid":
    coef0_param = st.sidebar.slider("coef0", -2.0, 2.0, 0.0, 0.1)


@st.cache_data
def make_dataset(kind, noise):
    if kind == "同心圓 (Circles)":
        return make_circles(n_samples=400, factor=0.35, noise=noise, random_state=42)
    if kind == "半月形 (Moons)":
        return make_moons(n_samples=400, noise=noise, random_state=42)
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

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{accuracy * 100:.1f}%")
metric_cols[1].metric("Kernel", kernel)
metric_cols[2].metric("Z 公式", z_label.replace("z = ", ""))
metric_cols[3].metric("Support Vectors", len(clf.support_))

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

col1, col2 = st.columns([1, 1.25])

with col1:
    st.markdown("### 2D 原始空間與決策邊界")
    st.markdown("這裡顯示模型投影回原始 x-y 平面後的分類邊界。切換 kernel 會直接改變邊界形狀。")

    fig_2d = go.Figure()
    fig_2d.add_trace(
        go.Contour(
            x=grid_x,
            y=grid_y,
            z=decision,
            colorscale="RdBu",
            opacity=0.35,
            contours=dict(showlines=False),
            showscale=False,
            name="Decision score",
        )
    )
    fig_2d.add_trace(
        go.Contour(
            x=grid_x,
            y=grid_y,
            z=decision,
            contours=dict(start=0, end=0, size=1, coloring="lines"),
            line=dict(width=3, color="#111827"),
            showscale=False,
            name="Decision boundary",
        )
    )
    fig_2d.add_trace(
        go.Scatter(
            x=X_raw[:, 0],
            y=X_raw[:, 1],
            mode="markers",
            marker=dict(size=8, color=y, colorscale="RdBu", line=dict(width=1, color="white")),
            name="Data",
        )
    )
    fig_2d.add_trace(
        go.Scatter(
            x=X_raw[clf.support_, 0],
            y=X_raw[clf.support_, 1],
            mode="markers",
            marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(width=2, color="#f59e0b")),
            name="Support vectors",
        )
    )
    fig_2d.update_layout(
        height=560,
        margin=dict(l=0, r=0, b=0, t=30),
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_2d, use_container_width=True)

with col2:
    st.markdown("### 3D 特徵空間")
    st.markdown(
        f"每個資料點都依照 `{z_label}` 被抬到 3D。`linear` kernel 會額外顯示超平面。"
    )

    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=X_raw[:, 0],
            y=X_raw[:, 1],
            z=z_raw,
            mode="markers",
            marker=dict(size=5, color=y, colorscale="RdBu", opacity=0.9, line=dict(width=0.5, color="white")),
            name="3D transformed data",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=X_raw[clf.support_, 0],
            y=X_raw[clf.support_, 1],
            z=z_raw[clf.support_],
            mode="markers",
            marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(width=3, color="#f59e0b")),
            name="Support vectors",
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
            plane_x, plane_y = np.meshgrid(
                np.linspace(x_min, x_max, 32),
                np.linspace(y_min, y_max, 32),
            )
            plane_z = (-w[0] * plane_x - w[1] * plane_y - b) / w[2]
            z_pad = max(0.5, (z_raw.max() - z_raw.min()) * 0.25)
            plane_z = np.clip(plane_z, z_raw.min() - z_pad, z_raw.max() + z_pad)
            fig_3d.add_trace(
                go.Surface(
                    x=plane_x,
                    y=plane_y,
                    z=plane_z,
                    opacity=0.45,
                    colorscale="Greys",
                    showscale=False,
                    name="Linear hyperplane",
                )
            )
        else:
            st.info("目前 linear 模型的超平面幾乎不依賴 Z 軸，所以 3D 平面不容易顯示。")

    fig_3d.update_layout(
        height=620,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.85)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")
st.markdown(
    """
### 觀察方向

- `linear`：最適合看「升維後用一個平面切開資料」的概念。
- `rbf`：常用的非線性 kernel，邊界通常最有彈性。
- `poly`：多項式 kernel，可以透過 degree 控制彎曲程度。
- `sigmoid`：形狀類似神經網路啟發的 kernel，某些參數下可能比較不穩定。

建議先選 `同心圓 + x² + y² + linear`，再切到 `rbf` 比較真正 kernel trick 的非線性決策邊界。
"""
)
