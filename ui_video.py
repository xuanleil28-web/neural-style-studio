# SPDX-License-Identifier: MIT
"""Streamlit 视频风格化页面：光流对齐 + 时间一致性损失。"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from engine_cache import get_engine
from style_transfer import (
    StyleParams,
    VideoInfo,
    VideoParams,
    __version__,
    open_rgb_image,
    probe_video,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEMO_STYLE = ASSETS_DIR / "demo_style.png"

VIDEO_DEFAULTS = {
    "v_style_strength": 1.0,
    "v_color_strength": 1.0,
    "v_brush_strength": 1.0,
    "v_brush_scale": 0.55,
    "v_content_strength": 0.5,
    "v_smoothness": 0.2,
    "v_first_steps": 35,
    "v_frame_steps": 8,
    "v_temporal": 200,
    "v_max_frames": 36,
    "v_video_size": 480,
    "v_flow_preset": "balanced",
    "v_occlusion": 0.20,
    "v_seed": 42,
}

VIDEO_PRESETS = {
    "稳定优先": {"v_temporal": 300, "v_frame_steps": 10, "v_max_frames": 36},
    "速度优先": {
        "v_temporal": 150,
        "v_frame_steps": 5,
        "v_max_frames": 24,
        "v_video_size": 360,
    },
    "质量优先": {
        "v_temporal": 250,
        "v_frame_steps": 12,
        "v_max_frames": 48,
        "v_video_size": 640,
    },
}


def init_video_state() -> None:
    for key, value in VIDEO_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def clear_video_outputs() -> None:
    st.session_state.pop("last_video_result", None)
    st.session_state.pop("last_video_preview", None)
    st.session_state.pop("last_video_params", None)
    st.session_state.pop("last_preview_params", None)


def apply_video_preset(name: str) -> None:
    st.session_state.update(VIDEO_PRESETS[name])
    clear_video_outputs()


@st.cache_data(show_spinner=False, max_entries=8)
def load_image_bytes(data: bytes) -> Image.Image:
    return open_rgb_image(io.BytesIO(data))


@st.cache_data(show_spinner=False, max_entries=2)
def load_demo_style(modified_at: float) -> Image.Image:
    del modified_at  # 仅用于让缓存随文件更新
    return open_rgb_image(str(DEMO_STYLE))


@st.cache_data(show_spinner=False, max_entries=2)
def inspect_video(data: bytes, max_size: int) -> VideoInfo:
    path = write_temp_video(data)
    try:
        return probe_video(path, max_size)
    finally:
        path.unlink(missing_ok=True)


def write_temp_video(data: bytes, suffix: str = ".mp4") -> Path:
    fd, name = tempfile.mkstemp(prefix="nss_upload_", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return Path(name)


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "未知"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}:{rest:02d}"


def estimated_seconds(first_steps: int, frame_steps: int, max_frames: int, size: int) -> int:
    # 实测 480p：首帧约 0.5s/轮，后续帧约 0.3s/轮（RTX 5060 Laptop）。
    base = first_steps * 0.5 + max(0, max_frames - 1) * frame_steps * 0.3
    return max(5, int(base * (size / 480.0) ** 2))


def image_caption(image: Image.Image, is_demo: bool, label: str) -> str:
    source = "内置示例" if is_demo else "已上传"
    return f"{label} · {image.width} × {image.height} px · {source}"


def render_video_sidebar() -> tuple[VideoParams, int]:
    with st.sidebar:
        st.markdown("## 视频参数")
        st.caption("光流对齐 + 时间一致性损失，减少逐帧风格化带来的闪烁。")

        st.markdown('<div class="side-label">视频预设</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.button(
            "稳定",
            on_click=apply_video_preset,
            args=("稳定优先",),
            use_container_width=True,
            help="提高时间一致性，减少闪烁，但可能增加拖影。",
        )
        p2.button(
            "速度",
            on_click=apply_video_preset,
            args=("速度优先",),
            use_container_width=True,
            help="更少轮数、更小分辨率，适合快速试片。",
        )
        p3.button(
            "质量",
            on_click=apply_video_preset,
            args=("质量优先",),
            use_container_width=True,
            help="更高分辨率与更多后续帧轮数，耗时明显增加。",
        )

        st.markdown('<div class="side-label">风格与色彩</div>', unsafe_allow_html=True)
        style_strength = st.slider(
            "风格强度",
            min_value=0.10,
            max_value=2.00,
            step=0.05,
            key="v_style_strength",
            help="全局 style_weight 倍率，基准为 3×10⁶。",
        )
        st.caption(f"style_weight ≈ {3.0e6 * style_strength:,.0f}")
        color_strength = st.slider(
            "色彩强度",
            min_value=0.00,
            max_value=2.00,
            step=0.05,
            key="v_color_strength",
            help="放大浅层 relu1_1 / relu2_1 的色彩权重。",
        )
        brush_strength = st.slider(
            "笔触强度",
            min_value=0.00,
            max_value=2.00,
            step=0.05,
            key="v_brush_strength",
            help="放大深层 relu4_1 / relu5_1 的笔触权重。",
        )
        brush_scale = st.slider(
            "笔触尺度",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            key="v_brush_scale",
            help="风格图输入长边 512 → 224px；越大笔触越粗。",
        )
        brush_pixels = int(round(512 - 288 * brush_scale))
        st.caption(f"风格图输入长边 ≈ {brush_pixels}px")

        st.markdown('<div class="side-label">内容与画面</div>', unsafe_allow_html=True)
        content_strength = st.slider(
            "内容保留",
            min_value=0.05,
            max_value=2.00,
            step=0.05,
            key="v_content_strength",
            help="对应 content_weight；越大结构越清晰。",
        )
        smoothness = st.slider(
            "画面平滑",
            min_value=0.00,
            max_value=1.00,
            step=0.02,
            key="v_smoothness",
            help="对应 tv_weight；越大越平滑。",
        )

        st.markdown('<div class="side-label">时序与光流</div>', unsafe_allow_html=True)
        first_steps = st.slider(
            "首帧轮数",
            min_value=10,
            max_value=50,
            step=5,
            key="v_first_steps",
            help="首帧没有历史参考，用更多轮数建立整体笔触基调。",
        )
        frame_steps = st.slider(
            "后续帧轮数",
            min_value=2,
            max_value=15,
            step=1,
            key="v_frame_steps",
            help="后续帧用光流热启动，通常 5–10 轮即可。",
        )
        temporal_weight = st.slider(
            "时间一致性",
            min_value=0.0,
            max_value=500.0,
            step=10.0,
            key="v_temporal",
            help="temporal_weight。越大越稳定，过大可能产生拖影。",
        )
        max_frames = st.slider(
            "最大帧数",
            min_value=8,
            max_value=120,
            step=4,
            key="v_max_frames",
            help="只处理视频前 N 帧；长视频建议分段处理。",
        )
        video_size = st.select_slider(
            "视频长边（px）",
            options=[360, 480, 640],
            key="v_video_size",
            help="视频分辨率。显存不足或想更快时选 360。",
        )
        flow_preset = st.selectbox(
            "光流精度",
            options=["fast", "balanced", "quality"],
            format_func=lambda value: {
                "fast": "快速（2 层 / 11px 窗口）",
                "balanced": "均衡（3 层 / 15px 窗口）",
                "quality": "精细（4 层 / 21px 窗口）",
            }[value],
            key="v_flow_preset",
            help="Farneback 光流的金字塔层数与搜索窗口。",
        )
        with st.expander("高级光流设置"):
            occlusion = st.slider(
                "遮挡敏感度",
                min_value=0.05,
                max_value=0.60,
                step=0.05,
                key="v_occlusion",
                help="越大，前后向光流误差大时越快放弃时序锁定。",
            )
        seed = st.number_input(
            "随机种子",
            min_value=0,
            max_value=999999,
            step=1,
            key="v_seed",
        )

        st.divider()
        estimate = estimated_seconds(first_steps, frame_steps, max_frames, video_size)
        st.caption(f"预计耗时约 {estimate // 60} 分 {estimate % 60} 秒（{max_frames} 帧，未计编码）")
        if torch.cuda.is_available():
            st.caption(f"⚡ 加速设备：{torch.cuda.get_device_name(0)}")
        else:
            st.warning("未检测到 CUDA：视频模式会非常慢，建议先用首帧预览并降低分辨率。")

    style_params = StyleParams(
        style_weight=3.0e6 * float(style_strength),
        content_weight=float(content_strength),
        color_strength=float(color_strength),
        brush_strength=float(brush_strength),
        brush_scale=float(brush_scale),
        tv_weight=float(smoothness),
        num_steps=int(first_steps),
        output_size=int(video_size),
        seed=int(seed),
    )
    video_params = VideoParams(
        style=style_params,
        max_frames=int(max_frames),
        video_size=int(video_size),
        first_frame_steps=int(first_steps),
        frame_steps=int(frame_steps),
        temporal_weight=float(temporal_weight),
        flow_preset=str(flow_preset),
        occlusion_sensitivity=float(occlusion),
        preview_every=6,
    )
    return video_params, int(max_frames)


def render_video_page() -> None:
    init_video_state()
    video_params, max_frames = render_video_sidebar()

    st.markdown(
        """
        <div class="hero">
            <div class="kicker">VIDEO STYLE TRANSFER · OPTICAL FLOW / TEMPORAL LOSS</div>
            <h1>视频风格化</h1>
            <p>对视频逐帧进行 VGG19 风格迁移，并用 Farneback 光流把上一帧的笔触推到当前帧，
            再通过时间一致性损失抑制闪烁。首帧用更多轮数建立笔触基调，后续帧用光流热启动加速。</p>
            <div class="tags">
                <span class="tag">光流对齐</span>
                <span class="tag">前后向一致性掩码</span>
                <span class="tag">时间一致性损失</span>
                <span class="tag">首帧预览</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="panel-head"><span class="num">01</span> 视频 <span class="en">VIDEO</span></div>',
            unsafe_allow_html=True,
        )
        video_file = st.file_uploader(
            "上传视频",
            type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
            key="video_upload",
            label_visibility="collapsed",
            on_change=clear_video_outputs,
        )
        if video_file is None:
            st.info("上传一段视频后，这里会显示首帧预览和视频信息。")
            video_info = None
            video_bytes = None
        else:
            video_bytes = video_file.getvalue()
            try:
                with st.spinner("正在读取视频信息…"):
                    video_info = inspect_video(video_bytes, int(video_params.video_size))
            except Exception as exc:
                st.error(f"视频读取失败：{exc}")
                video_info = None
            if video_info is not None:
                st.image(video_info.first_frame, use_container_width=True)
                st.caption(
                    f"首帧预览 · 原片 {video_info.width}×{video_info.height} · "
                    f"{video_info.fps:.1f} fps · {format_duration(video_info.duration)}"
                )
                st.caption(
                    f"将按 {video_info.first_frame.width}×{video_info.first_frame.height} 处理；"
                    f"最多 {max_frames} 帧。"
                )

    with right:
        st.markdown(
            '<div class="panel-head"><span class="num">02</span> 风格图 <span class="en">STYLE</span></div>',
            unsafe_allow_html=True,
        )
        style_file = st.file_uploader(
            "上传风格图",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="video_style_upload",
            label_visibility="collapsed",
            on_change=clear_video_outputs,
        )
        if style_file is not None:
            try:
                style_image = load_image_bytes(style_file.getvalue())
                style_is_demo = False
            except Exception as exc:
                st.error(f"风格图读取失败：{exc}")
                st.stop()
        else:
            try:
                style_image = load_demo_style(DEMO_STYLE.stat().st_mtime)
                style_is_demo = True
            except Exception as exc:
                st.error(f"内置风格图读取失败：{exc}")
                st.stop()
        st.image(style_image, use_container_width=True)
        st.caption(image_caption(style_image, style_is_demo, "风格图"))

    if video_file is None or video_info is None:
        st.markdown(
            '<p class="muted">视频模式需要上传视频；风格图不传时会使用内置梵高示例。'
            "建议先用「预览首帧」确认风格，再生成完整视频。</p>",
            unsafe_allow_html=True,
        )
        render_video_footer()
        return

    estimate = estimated_seconds(
        video_params.first_frame_steps,
        video_params.frame_steps,
        max_frames,
        video_params.video_size,
    )
    st.markdown(
        f'<div class="run-note">当前配置：首帧 {video_params.first_frame_steps} 轮 + 后续 '
        f"{video_params.frame_steps} 轮/帧 · 最多 {max_frames} 帧 · 长边 {video_params.video_size}px · "
        f"时间一致性 {video_params.temporal_weight:.0f} · 预计约 {estimate // 60} 分 {estimate % 60} 秒</div>",
        unsafe_allow_html=True,
    )

    button_left, button_right = st.columns(2)
    preview_clicked = button_left.button("预览首帧", use_container_width=True)
    generate_clicked = button_right.button(
        "生成风格化视频", type="primary", use_container_width=True
    )

    if preview_clicked or generate_clicked:
        temp_path = write_temp_video(video_bytes or b"")
        try:
            with st.spinner("正在加载 VGG19 模型…"):
                engine = get_engine()
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            st.error(f"模型加载失败：{exc}")
            render_video_footer()
            return

        if preview_clicked:
            progress_bar = st.progress(0.0, text="首帧优化中…")
            preview_slot = st.empty()

            def on_preview_progress(done: int, total: int, image: Image.Image | None) -> None:
                progress_bar.progress(
                    min(1.0, done / max(1, total)),
                    text=f"首帧优化中… 第 {done}/{total} 轮",
                )
                if image is not None:
                    preview_slot.image(image, caption="首帧中间预览", width=300)

            try:
                with st.spinner("正在渲染首帧…"):
                    result = engine.preview_video_first_frame(
                        temp_path, style_image, video_params, progress=on_preview_progress
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                st.error("显存不足：请把视频长边降到 360px，或减少首帧轮数。")
            except Exception as exc:
                st.error(f"首帧预览失败：{exc}")
                with st.expander("查看错误详情"):
                    st.exception(exc)
            else:
                st.session_state["last_video_preview"] = result
                st.session_state["last_preview_params"] = video_params
                st.toast("首帧预览完成", icon="✅")
            finally:
                progress_bar.empty()
                preview_slot.empty()
                temp_path.unlink(missing_ok=True)

        else:
            progress_bar = st.progress(0.0, text="准备视频处理…")
            status_slot = st.empty()
            frame_slot = st.empty()

            def on_video_progress(
                stage: str, done: int, total: int, image: Image.Image | None
            ) -> None:
                if stage == "first_frame":
                    status_slot.markdown("**首帧：建立整体笔触基调…**")
                    progress_bar.progress(0.0, text="首帧优化中…")
                elif stage == "encode":
                    status_slot.markdown("**正在编码 H.264 视频…**")
                    progress_bar.progress(1.0, text="编码中…")
                else:
                    progress_bar.progress(
                        min(1.0, done / max(1, total)),
                        text=f"处理帧 {done}/{total}",
                    )
                    status_slot.markdown(
                        f"**第 {done}/{total} 帧** · 光流对齐 + {video_params.frame_steps} 轮热启动"
                    )
                    if image is not None:
                        frame_slot.image(image, caption=f"第 {done} 帧输出预览", width=300)

            try:
                with st.spinner("正在生成视频，请保持页面打开…"):
                    result = engine.process_video(
                        temp_path, style_image, video_params, progress=on_video_progress
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                st.error("显存不足：请把视频长边降到 360px、减少最大帧数，或降低后续帧轮数。")
            except Exception as exc:
                st.error(f"视频生成失败：{exc}")
                with st.expander("查看错误详情"):
                    st.exception(exc)
            else:
                st.session_state["last_video_result"] = result
                st.session_state["last_video_params"] = video_params
                st.session_state.pop("last_video_preview", None)
                st.toast("视频生成完成", icon="✅")
            finally:
                progress_bar.empty()
                status_slot.empty()
                frame_slot.empty()
                temp_path.unlink(missing_ok=True)

    # ---------------- 首帧预览 ----------------
    preview = st.session_state.get("last_video_preview")
    if preview is not None:
        preview_params = st.session_state.get("last_preview_params", video_params)
        st.divider()
        st.markdown('<span class="result-badge">FIRST FRAME PREVIEW</span>', unsafe_allow_html=True)
        st.markdown('<div class="result-title">首帧预览</div>', unsafe_allow_html=True)
        if preview_params.signature() != video_params.signature():
            st.info("视频参数已改变，下方仍是上一次的首帧预览。")
        preview_left, preview_right = st.columns([1.12, 1.0], gap="large")
        with preview_left:
            st.image(preview.image, use_container_width=True)
            st.caption(
                f"输出 {preview.image.width} × {preview.image.height} px · "
                f"首帧 {preview_params.first_frame_steps} 轮"
            )
        with preview_right:
            st.metric("首帧耗时", f"{preview.elapsed:.1f} 秒")
            st.caption(
                "正式生成时会以这张首帧建立笔触基调，后续帧使用光流变形 + 时间一致性损失。"
            )
            if preview.history:
                st.caption("首帧总损失曲线")
                chart = pd.DataFrame(preview.history).set_index("step")[["total"]]
                st.line_chart(chart, height=160, color="#c2410c")

    # ---------------- 视频结果 ----------------
    result = st.session_state.get("last_video_result")
    if result is None:
        render_video_footer()
        return

    st.divider()
    st.markdown('<span class="result-badge">VIDEO COMPLETE</span>', unsafe_allow_html=True)
    st.markdown('<div class="result-title">风格化视频</div>', unsafe_allow_html=True)

    last_params = st.session_state.get("last_video_params")
    if last_params is not None and last_params.signature() != video_params.signature():
        st.info("视频参数已改变，下方仍是上一次生成的结果。点击「生成风格化视频」应用新参数。")

    video_left, video_right = st.columns([1.12, 1.0], gap="large")
    with video_left:
        st.video(result.video_bytes, format="video/mp4")
        st.caption(
            f"输出 {result.width} × {result.height} px · {result.fps:.1f} fps · "
            f"{result.frame_count} 帧 · 编码 {result.codec}"
        )
        if result.codec != "H.264":
            st.warning("当前环境未使用 H.264 编码，部分浏览器可能无法内嵌播放，请下载后用播放器查看。")

    with video_right:
        m1, m2 = st.columns(2)
        m1.metric("总耗时", f"{result.elapsed / 60:.1f} 分钟")
        m2.metric("处理帧数", f"{result.frame_count}")
        st.caption(
            f"时间一致性权重 {result.temporal_weight:.0f} · 源视频帧数 {result.source_frame_count or '未知'}"
        )
        st.caption(f"运行设备：{result.device_name} · 输出不含音轨")

        buffer = io.BytesIO()
        # VideoResult already stores bytes; keep one copy for the download button.
        buffer.write(result.video_bytes)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "下载风格化视频（MP4）",
            data=buffer.getvalue(),
            file_name=f"stylized_video_{stamp}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        st.caption("视频只在本地处理；长视频建议分段生成后自行拼接。")

        if result.history:
            history = pd.DataFrame(result.history)
            per_frame = history.groupby("frame", as_index=False).tail(1)
            st.caption("每帧最终总损失")
            st.line_chart(
                per_frame.set_index("frame")[["total"]],
                height=170,
                color="#c2410c",
            )

    with st.expander("首帧输出", expanded=False):
        st.image(result.first_frame, width=360)
        st.caption("正式视频的第一帧；可用它和预览结果对比。")

    render_video_footer()


def render_video_footer() -> None:
    st.divider()
    with st.expander("视频模式参数说明"):
        st.markdown(
            """
| 参数 | 内部设置 | 作用 |
| --- | --- | --- |
| 首帧轮数 | `first_frame_steps` | 首帧没有历史参考，用更多轮数建立整体风格基调 |
| 后续帧轮数 | `frame_steps` | 每帧用光流热启动后只做少量优化，控制速度与稳定 |
| 时间一致性 | `temporal_weight` | 约束非遮挡区与光流变形后的上一帧一致，抑制闪烁 |
| 最大帧数 | `max_frames` | 只处理前 N 帧，长视频建议分段 |
| 视频长边 | `video_size` | 输出分辨率，360/480/640；越大越慢、越占显存 |
| 光流精度 | `flow_preset` | Farneback 金字塔层数与窗口大小 |
| 遮挡敏感度 | `occlusion_sensitivity` | 前后向光流误差的指数衰减系数 |

视频处理流程：首帧按普通图片风格迁移（默认 35 轮）→ 对后续帧计算
当前帧到上一帧的反向 Farneback 光流 → 用前后向一致性生成遮挡掩码 → 将上一帧风格图
`remap` 到当前帧 → 在非遮挡区用上一帧笔触、遮挡区用当前内容帧做热启动 → 加入
`temporal_loss = mean(mask × (X - warped_prev)²)` 后做少量 L-BFGS 优化。
            """
        )
    st.caption(f"v{__version__} · 视频实现：OpenCV Farneback 光流 + 时间一致性损失 + H.264 编码")
    st.caption("提示：视频模式计算量远大于图片模式，建议先用「预览首帧」确认风格，再生成完整视频。")
