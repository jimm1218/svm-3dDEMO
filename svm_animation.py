from manim import *
import numpy as np

# 調整 Manim 影片輸出解析度 (實體視窗大小)
# 將解析度調整為較小的 640x360 (比例維持 16:9)，讓渲染出來的視窗變小
config.pixel_width = 640
config.pixel_height = 360

class SvmKernelAnimation(ThreeDScene):
    def construct(self):
        # --- SCENE SETUP ---
        # 如果覺得「畫面裡的 3D 座標與物件」佔據版面太多，可以將 zoom 調小 (例如原本是 1.2，這裡改為 0.9)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES, zoom=0.9)
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[0, 4, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        self.add(axes)

        # --- 1. 顯示 2D 資料點 ---
        self.begin_ambient_camera_rotation(rate=0.1)
        
        text_2d = Text("1. 原始 2D 資料 (線性不可分)", font="Microsoft JhengHei").to_corner(UL).scale(0.5)
        self.add_fixed_in_frame_mobjects(text_2d)

        # 產生同心圓資料點
        inner_circle = VGroup(*[Dot(axes.c2p(0.5 * np.cos(a), 0.5 * np.sin(a), 0), color=BLUE) for a in np.linspace(0, TAU, 15)])
        outer_circle = VGroup(*[Dot(axes.c2p(1.5 * np.cos(a), 1.5 * np.sin(a), 0), color=RED) for a in np.linspace(0, TAU, 25)])
        
        self.play(Create(inner_circle), Create(outer_circle), run_time=2)
        self.wait(2)

        # --- 2. 引入核函數轉換 ---
        self.play(FadeOut(text_2d))
        # 改用 Text 搭配系統字體，避免 LaTeX 編譯中文失敗
        text_transform = Text("2. 透過核函數轉換: z = x² + y²", font="Microsoft JhengHei", t2c={"z = x² + y²": YELLOW}).to_corner(UL).scale(0.5)
        self.add_fixed_in_frame_mobjects(text_transform)

        # 定義轉換函數
        def kernel_transform(point):
            x, y, z = point
            return np.array([x, y, x**2 + y**2])

        # 動畫：將點提升到 3D 空間
        self.play(
            ApplyPointwiseFunction(kernel_transform, inner_circle),
            ApplyPointwiseFunction(kernel_transform, outer_circle),
            run_time=3
        )
        self.wait(2)

        # --- 3. 在 3D 空間中用平面切割 ---
        self.play(FadeOut(text_transform))
        text_3d_cut = Text("3. 在 3D 空間中，資料變為線性可分", font="Microsoft JhengHei").to_corner(UL).scale(0.5)
        self.add_fixed_in_frame_mobjects(text_3d_cut)

        # 創建切割平面
        plane = Surface(
            lambda u, v: axes.c2p(u, v, 1.5), # z=1.5 的平面
            u_range=[-2, 2],
            v_range=[-2, 2],
            checkerboard_colors=[GRAY_C, GRAY_D],
            fill_opacity=0.5
        )

        self.play(Create(plane), run_time=2)
        self.wait(4)