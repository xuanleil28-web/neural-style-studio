# SPDX-License-Identifier: MIT
"""Streamlit 界面：上传内容图/风格图，拖动滑块调节色彩与笔触。"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from engine_cache import get_engine
from style_transfer import StyleParams, __version__, open_rgb_image
from ui_video import render_video_page

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
DEMO_CONTENT = ASSETS_DIR / "demo_content.png"
DEMO_STYLE = ASSETS_DIR / "demo_style.png"

DEFAULTS = {
    "style_strength": 1.0,
    "color_strength": 1.0,
    "brush_strength": 1.0,
    "brush_scale": 0.55,
    "content_strength": 0.5,
    "smoothness": 0.2,
    "num_steps": 40,
    "output_size": 512,
    "seed": 42,
}

PRESETS = {
    "默认": dict(DEFAULTS),
    "梵高·厚涂": {
        "style_strength": 1.0,
        "color_strength": 1.10,
        "brush_strength": 1.35,
        "brush_scale": 0.68,
        "content_strength": 0.50,
        "smoothness": 0.14,
        "num_steps": 40,
        "output_size": 512,
        "seed": 42,
    },
    "色彩冲击": {
        "style_strength": 1.20,
        "color_strength": 1.80,
        "brush_strength": 0.85,
        "brush_scale": 0.50,
        "content_strength": 0.55,
        "smoothness": 0.20,
        "num_steps": 40,
        "output_size": 512,
        "seed": 42,
    },
    "细腻笔触": {
        "style_strength": 0.85,
        "color_strength": 0.95,
        "brush_strength": 0.80,
        "brush_scale": 0.15,
        "content_strength": 0.70,
        "smoothness": 0.30,
        "num_steps": 25,
        "output_size": 512,
        "seed": 42,
    },
    "快速预览": {
        "style_strength": 1.0,
        "color_strength": 1.0,
        "brush_strength": 1.0,
        "brush_scale": 0.55,
        "content_strength": 0.5,
        "smoothness": 0.2,
        "num_steps": 15,
        "output_size": 384,
        "seed": 42,
    },
}

CSS = """
<style>
    :root {
        --ink: #23212b;
        --muted: #6d6677;
        --accent: #c2410c;
        --line: rgba(35, 33, 43, .10);
        --card: rgba(255, 255, 255, .78);
    }
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(255, 237, 213, .85) 0%, rgba(247, 243, 236, 0) 42%),
            radial-gradient(circle at 92% 8%, rgba(214, 232, 222, .55) 0%, rgba(247, 243, 236, 0) 38%),
            #f7f3ec;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -.02em;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 2rem 2.2rem 1.9rem;
        margin-bottom: 1.7rem;
        background: linear-gradient(120deg, rgba(255, 255, 255, .96), rgba(255, 248, 238, .90));
        box-shadow: 0 20px 55px rgba(80, 58, 35, .08);
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -70px;
        top: -95px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle at 40% 40%, rgba(194, 65, 12, .16), rgba(194, 65, 12, 0) 68%);
        pointer-events: none;
    }
    .kicker {
        color: var(--accent);
        font-size: .70rem;
        font-weight: 800;
        letter-spacing: .18em;
    }
    .hero h1 {
        margin: .45rem 0 .55rem;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.12;
        font-weight: 800;
    }
    .hero p {
        margin: 0;
        max-width: 780px;
        color: #5d5665;
        font-size: .98rem;
        line-height: 1.75;
    }
    .hero .tags {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
        margin-top: 1rem;
    }
    .hero .tag {
        font-size: .72rem;
        color: #7b4a2f;
        background: rgba(194, 65, 12, .08);
        border: 1px solid rgba(194, 65, 12, .14);
        border-radius: 999px;
        padding: .22rem .62rem;
    }
    .panel-head {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin: .2rem 0 .65rem;
        color: var(--ink);
        font-weight: 800;
        font-size: 1.02rem;
    }
    .panel-head .num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 999px;
        background: #23212b;
        color: #fff8ee;
        font-size: .68rem;
        letter-spacing: 0;
    }
    .panel-head .en {
        color: #a49cab;
        font-size: .62rem;
        letter-spacing: .16em;
        font-weight: 700;
        margin-left: .1rem;
    }
    div[data-testid="stImage"] img {
        max-height: 310px;
        width: auto !important;
        height: auto !important;
        max-width: 100%;
        display: block;
        margin: 0 auto;
        object-fit: contain;
        border-radius: 14px;
        box-shadow: 0 12px 30px rgba(49, 35, 22, .14);
    }
    .st-key-result_frame div[data-testid="stImage"] img {
        max-height: 560px;
        box-shadow: 0 22px 55px rgba(49, 35, 22, .20);
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, .55);
        border-radius: 14px;
        padding: .15rem .2rem;
    }
    div[data-testid="stSidebar"] {
        background: rgba(255, 253, 249, .82);
        border-right: 1px solid var(--line);
    }
    div[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }
    .side-label {
        margin: 1.15rem 0 .35rem;
        color: #8b8394;
        font-size: .66rem;
        font-weight: 800;
        letter-spacing: .17em;
        text-transform: uppercase;
    }
    .stButton > button[kind="primary"] {
        border: 0;
        border-radius: 13px;
        padding: .72rem 1rem;
        font-weight: 800;
        letter-spacing: .03em;
        color: #fffaf3;
        background: linear-gradient(135deg, #c2410c, #9a3412);
        box-shadow: 0 12px 26px rgba(194, 65, 12, .22);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #b93d0b, #7c2d12);
        box-shadow: 0 14px 30px rgba(194, 65, 12, .28);
    }
    .result-badge {
        display: inline-block;
        color: #166534;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        padding: .2rem .65rem;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .10em;
    }
    .result-title {
        margin: .5rem 0 .35rem;
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -.02em;
    }
    .muted {
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.65;
    }
    .run-note {
        margin: .4rem 0 .7rem;
        color: var(--muted);
        font-size: .82rem;
    }
    code {
        color: #9a3412;
        background: rgba(194, 65, 12, .08);
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.35rem;
    }
    @media (max-width: 720px) {
        .hero { padding: 1.4rem 1.25rem; }
        .hero h1 { font-size: 1.9rem; }
        div[data-testid="stImage"] img { max-height: 240px; }
    }
</style>
"""


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    st.session_state.update(PRESETS[name])
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_params", None)


def clear_result() -> None:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_params", None)


@st.cache_data(show_spinner=False, max_entries=12)
def load_image_bytes(data: bytes) -> Image.Image:
    return open_rgb_image(io.BytesIO(data))


@st.cache_data(show_spinner=False, max_entries=4)
def load_demo_image(path: str, modified_at: float) -> Image.Image:
    del modified_at  # 仅用于让缓存随文件更新
    return open_rgb_image(path)


def image_caption(image: Image.Image, is_demo: bool, label: str) -> str:
    source = "内置示例" if is_demo else "已上传"
    return f"{label} · {image.width} × {image.height} px · {source}"


def render_image_page() -> None:
    init_state()

    # ---------------- 侧边栏：参数控制 ----------------
    with st.sidebar:
        st.markdown("## 参数控制")
        st.caption("滑块直接改变损失函数权重、风格图输入尺度与优化轮数。")

        st.markdown('<div class="side-label">快速预设</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.button(
            "梵高·厚涂",
            on_click=apply_preset,
            args=("梵高·厚涂",),
            use_container_width=True,
            help="加强深层笔触权重，风格图缩小以放大笔刷颗粒。",
        )
        p2.button(
            "色彩冲击",
            on_click=apply_preset,
            args=("色彩冲击",),
            use_container_width=True,
            help="提高浅层色彩权重与整体风格强度。",
        )
        p3, p4 = st.columns(2)
        p3.button(
            "细腻笔触",
            on_click=apply_preset,
            args=("细腻笔触",),
            use_container_width=True,
            help="使用更大的风格图，保留细密笔触。",
        )
        p4.button(
            "快速预览",
            on_click=apply_preset,
            args=("快速预览",),
            use_container_width=True,
            help="15 轮 + 384px，适合先看大致效果。",
        )
        st.button(
            "恢复默认参数",
            on_click=apply_preset,
            args=("默认",),
            use_container_width=True,
        )

        st.markdown('<div class="side-label">风格与色彩</div>', unsafe_allow_html=True)
        style_strength = st.slider(
            "风格强度",
            min_value=0.10,
            max_value=2.00,
            step=0.05,
            key="style_strength",
            help="全局风格权重倍率。基准 style_weight = 3×10⁶，越大风格越浓。",
        )
        st.caption(f"style_weight ≈ {3.0e6 * style_strength:,.0f}")

        color_strength = st.slider(
            "色彩强度",
            min_value=0.00,
            max_value=2.00,
            step=0.05,
            key="color_strength",
            help="放大 VGG 浅层 relu1_1 / relu2_1 的权重，控制色调、冷暖与色彩丰富度。",
        )

        brush_strength = st.slider(
            "笔触强度",
            min_value=0.00,
            max_value=2.00,
            step=0.05,
            key="brush_strength",
            help="放大深层 relu4_1 / relu5_1 的权重，让笔触、漩涡纹理更明显。",
        )

        brush_scale = st.slider(
            "笔触尺度",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            key="brush_scale",
            help="改变风格图输入长边：0 为 512px（细笔触），1 为 224px（粗笔触）。",
        )
        brush_pixels = int(round(512 - 288 * brush_scale))
        st.caption(f"风格图输入长边 ≈ {brush_pixels}px · {'粗' if brush_scale > .66 else '中' if brush_scale > .33 else '细'}笔触")

        st.markdown('<div class="side-label">内容与画面</div>', unsafe_allow_html=True)
        content_strength = st.slider(
            "内容保留",
            min_value=0.05,
            max_value=2.00,
            step=0.05,
            key="content_strength",
            help="对应 content_weight。越大，山体、道路等原始结构保留得越清楚。",
        )
        smoothness = st.slider(
            "画面平滑",
            min_value=0.00,
            max_value=1.00,
            step=0.02,
            key="smoothness",
            help="对应 TV 权重。越大画面越平滑；越小颗粒感和噪点越多。",
        )

        st.markdown('<div class="side-label">运行设置</div>', unsafe_allow_html=True)
        num_steps = st.slider(
            "迭代轮数",
            min_value=10,
            max_value=60,
            step=5,
            key="num_steps",
            help="每轮内部执行 10 次 L-BFGS 迭代，轮数越多细节越充分，耗时也越长。",
        )
        st.caption(f"共 {num_steps * 10} 次 L-BFGS 内部迭代")

        output_size = st.select_slider(
            "输出长边（px）",
            options=[384, 512, 640],
            key="output_size",
            help="输出图片长边分辨率。显存不足时建议降到 384。",
        )
        seed = st.number_input(
            "随机种子",
            min_value=0,
            max_value=999999,
            step=1,
            key="seed",
            help="固定种子可复现相近结果；改变种子会改变噪声初始化和细节。",
        )

        st.divider()
        if torch.cuda.is_available():
            st.caption(f"⚡ 加速设备：{torch.cuda.get_device_name(0)}")
        else:
            st.warning("未检测到 CUDA，将在 CPU 上运行，40 轮可能需要数分钟。")

    params = StyleParams(
        style_weight=3.0e6 * float(style_strength),
        content_weight=float(content_strength),
        color_strength=float(color_strength),
        brush_strength=float(brush_strength),
        brush_scale=float(brush_scale),
        tv_weight=float(smoothness),
        num_steps=int(num_steps),
        output_size=int(output_size),
        seed=int(seed),
    )

    # ---------------- 主区域：上传与预览 ----------------
    st.markdown(
        """
        <div class="hero">
            <div class="kicker">NEURAL STYLE TRANSFER · VGG19 / GRAM MATRIX / L-BFGS</div>
            <h1>风格迁移工作台</h1>
            <p>上传一张内容图和一张风格图，把内容图的轮廓与构图，染上风格图的色彩与笔触。
            拖动左侧滑块即可改变损失函数权重与风格图输入尺度，实时比较不同效果。</p>
            <div class="tags">
                <span class="tag">色彩强度</span>
                <span class="tag">笔触强度</span>
                <span class="tag">笔触尺度</span>
                <span class="tag">内容保留</span>
                <span class="tag">画面平滑</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="panel-head"><span class="num">01</span> 内容图 <span class="en">CONTENT</span></div>',
            unsafe_allow_html=True,
        )
        content_file = st.file_uploader(
            "上传内容图",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="content_upload",
            label_visibility="collapsed",
            on_change=clear_result,
        )
        if content_file is not None:
            try:
                content_img = load_image_bytes(content_file.getvalue())
                content_is_demo = False
            except Exception as exc:
                st.error(f"内容图读取失败：{exc}")
                st.stop()
        else:
            try:
                content_img = load_demo_image(str(DEMO_CONTENT), DEMO_CONTENT.stat().st_mtime)
                content_is_demo = True
            except Exception as exc:
                st.error(f"内置示例内容图读取失败：{exc}")
                st.stop()

        st.image(content_img, use_container_width=True)
        st.caption(image_caption(content_img, content_is_demo, "内容图"))
        if content_is_demo:
            st.caption("未上传文件时自动使用内置示例，可直接运行体验。")

    with right:
        st.markdown(
            '<div class="panel-head"><span class="num">02</span> 风格图 <span class="en">STYLE</span></div>',
            unsafe_allow_html=True,
        )
        style_file = st.file_uploader(
            "上传风格图",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="style_upload",
            label_visibility="collapsed",
            on_change=clear_result,
        )
        if style_file is not None:
            try:
                style_img = load_image_bytes(style_file.getvalue())
                style_is_demo = False
            except Exception as exc:
                st.error(f"风格图读取失败：{exc}")
                st.stop()
        else:
            try:
                style_img = load_demo_image(str(DEMO_STYLE), DEMO_STYLE.stat().st_mtime)
                style_is_demo = True
            except Exception as exc:
                st.error(f"内置示例风格图读取失败：{exc}")
                st.stop()

        st.image(style_img, use_container_width=True)
        st.caption(image_caption(style_img, style_is_demo, "风格图"))
        if style_is_demo:
            st.caption("换成梵高、莫奈、浮世绘等风格图，笔触和配色会随之改变。")

    if min(content_img.size) < 24 or min(style_img.size) < 24:
        st.error("图片尺寸过小，请上传短边至少 24 像素的图片。")
        st.stop()

    st.markdown(
        f'<div class="run-note">当前配置：{num_steps} 轮 × 10 次迭代 · 输出长边 {output_size}px · '
        f'风格图 {brush_pixels}px · 预计 GPU 耗时约 {max(5, int(num_steps * 0.45))}–{max(8, int(num_steps * 0.8))} 秒</div>',
        unsafe_allow_html=True,
    )
    run_clicked = st.button("开始风格化", type="primary", use_container_width=True)

    if run_clicked:
        progress_bar = st.progress(0.0, text="准备模型…")
        preview_slot = st.empty()

        def on_progress(done: int, total: int, preview: Image.Image | None) -> None:
            progress_bar.progress(
                min(1.0, done / max(1, total)),
                text=f"正在优化笔触… 第 {done} / {total} 轮",
            )
            if preview is not None:
                preview_slot.image(
                    preview,
                    caption=f"第 {done} 轮中间结果",
                    width=300,
                )

        try:
            with st.spinner("首次运行需要加载 VGG19 权重，请稍候…"):
                engine = get_engine()
            result = engine.stylize(content_img, style_img, params, progress=on_progress)
        except torch.cuda.OutOfMemoryError:
            progress_bar.empty()
            preview_slot.empty()
            torch.cuda.empty_cache()
            st.error("显存不足：请把「输出长边」降到 384px，或减少迭代轮数后重试。")
        except Exception as exc:
            progress_bar.empty()
            preview_slot.empty()
            st.error(f"风格化失败：{exc}")
            with st.expander("查看错误详情"):
                st.exception(exc)
        else:
            progress_bar.empty()
            preview_slot.empty()
            st.session_state["last_result"] = result
            st.session_state["last_params"] = params
            st.toast("风格化完成", icon="✅")

    # ---------------- 结果展示 ----------------
    stored = st.session_state.get("last_result")
    if stored is None:
        st.markdown(
            '<p class="muted" style="margin-top:1rem">结果会显示在这里。建议先用「快速预览」试参数，满意后再提高轮数。</p>',
            unsafe_allow_html=True,
        )
        render_footer()
        return

    st.divider()
    st.markdown('<span class="result-badge">STYLE TRANSFER COMPLETE</span>', unsafe_allow_html=True)
    st.markdown('<div class="result-title">风格化结果</div>', unsafe_allow_html=True)

    last_params = st.session_state.get("last_params")
    if last_params is not None and last_params.signature() != params.signature():
        st.info("滑块参数已改变，下方仍是上一次生成的结果。点击上方「开始风格化」应用新参数。")

    result_left, result_right = st.columns([1.12, 1.0], gap="large")

    with result_left:
        with st.container(key="result_frame"):
            st.image(stored.image, use_container_width=True)
        st.caption(f"输出 {stored.image.width} × {stored.image.height} px · 内容输入 {stored.content_shape[1]} × {stored.content_shape[0]} px")

    with result_right:
        final = stored.history[-1] if stored.history else {"total": 0.0, "content": 0.0, "style": 0.0, "tv": 0.0}
        m1, m2 = st.columns(2)
        m1.metric("总耗时", f"{stored.elapsed:.1f} 秒")
        m2.metric("最终总损失", f"{final['total']:.2f}")
        st.caption(
            f"内容损失 {final['content']:.4g} · 风格损失 {final['style']:.4g} · "
            f"平滑损失 {final['tv']:.4g}"
        )
        st.caption(f"运行设备：{stored.device_name} · 风格图输入 {stored.style_shape[1]} × {stored.style_shape[0]} px")
        st.caption("风格层权重（relu1_1 → relu5_1）：" + " / ".join(f"{w:.2f}" for w in stored.layer_weights))

        buffer = io.BytesIO()
        stored.image.save(buffer, format="PNG")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "下载风格化图片（PNG）",
            data=buffer.getvalue(),
            file_name=f"stylized_{stamp}.png",
            mime="image/png",
            use_container_width=True,
        )
        st.caption("图片只在本地处理，不会上传到任何服务器。")

        if stored.history:
            chart_data = pd.DataFrame(stored.history).set_index("step")[["total"]]
            st.caption("总损失曲线（每轮记录一次）")
            st.line_chart(chart_data, height=170, color="#c2410c")

    render_footer()


def render_footer() -> None:
    st.divider()
    with st.expander("这些滑块分别改了什么？"):
        st.markdown(
            """
| 滑块 | 内部参数 | 作用 |
| --- | --- | --- |
| 风格强度 | `style_weight`（基准 3×10⁶） | 全局放大风格损失；越大风格越浓，过大可能压过内容结构 |
| 色彩强度 | relu1_1 / relu2_1 层权重倍率 | 浅层特征主要携带颜色与色调，提高后画面更接近风格图的配色 |
| 笔触强度 | relu4_1 / relu5_1 层权重倍率 | 深层特征对应大尺度纹理，提高后漩涡、厚涂笔触更明显 |
| 笔触尺度 | 风格图输入长边（512 → 224px） | 风格图越小，单个笔刷覆盖内容图的相对面积越大，笔触越粗犷 |
| 内容保留 | `content_weight` | 约束 relu4_2 内容特征；越大山体、道路等结构越清晰 |
| 画面平滑 | `tv_weight` | 全变分损失；越大画面越平滑，越小颗粒与噪点越多 |
| 迭代轮数 | `num_steps` | 每轮 10 次 L-BFGS 内部迭代，越多细节越充分 |

算法沿用 `StyleTransfer.ipynb`：用在 ImageNet 上预训练的 VGG19 提取内容与风格特征，用 Gram 矩阵表示风格，
在归一化图像空间中以内容图 + 30% 噪声为初始值，通过 L-BFGS 最小化「内容损失 + 风格损失 + TV 损失」。
本程序把池化层换成平均池化、ReLU 改为非原地操作，并按滑块重新分配五个风格层的权重。
            """
        )
    st.caption(f"v{__version__} · 算法来源：StyleTransfer.ipynb · 实现：PyTorch + torchvision + Streamlit · 完全本地运行")
    st.caption("小提示：总损失中风格项通常占主导，其绝对值大小不代表画质，重点看画面效果。")


def main() -> None:
    st.set_page_config(
        page_title="风格迁移工作台",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    navigation = st.navigation(
        [
            st.Page(
                render_image_page,
                title="图片风格化",
                icon="🖼️",
                url_path="image",
                default=True,
            ),
            st.Page(
                render_video_page,
                title="视频风格化",
                icon="🎬",
                url_path="video",
            ),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
