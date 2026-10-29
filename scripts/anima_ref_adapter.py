from __future__ import annotations

import base64
import io
import math
import os
import sys
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

from backend import memory_management
from backend.args import dynamic_args
from lib_anima_ref import (
    HF_FILES,
    HF_REPO,
    PILOT_LORA_FILENAME,
    V2_LORA_FILENAME,
    AnimaRefAdapterConfig,
    AnimaSafePAGConfig,
    _Adapter,
    _is_valid_safetensors,
    build_asym_reference_image,
    download_adapter_lora,
    download_pilot_lora,
    encode_multi_reference_latents,
    encode_reference_latents,
    find_lora_path,
    is_anima_adapter,
    letterbox_image,
    list_lora_files,
    patch_unet_for_anima_ref,
    patch_unet_for_safe_pag,
)
from lib_anima_ref.adapter import logger
from modules import scripts, shared
from modules.ui_components import FormRow, InputAccordion, ToolButton


def _is_lobe_theme_active() -> bool:
    try:
        from modules import extensions

        for ext in extensions.active() or []:
            name = (getattr(ext, "name", "") or "").lower()
            canon = (getattr(ext, "canonical_name", "") or "").lower()
            if "lobe" in name or "lobe" in canon:
                return True
        return False
    except Exception:
        return False


def _default_lora_choice(choices: list[str]) -> Optional[str]:
    for c in choices:
        if os.path.basename(c).lower() == V2_LORA_FILENAME.lower():
            return c
    for c in choices:
        if os.path.basename(c).lower() == PILOT_LORA_FILENAME.lower():
            return c
    for c in choices:
        lower = os.path.basename(c).lower()
        if "anima" in lower and "ref" in lower:
            return c
    return choices[0] if choices else None


def _coerce_pil_image(img) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        if os.path.isfile(img):
            try:
                return Image.open(img)
            except Exception:
                pass
        try:
            return Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[-1])))
        except Exception:
            return None
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, dict) and "image" in img:
        return _coerce_pil_image(img["image"])
    return None


def _calc_anima_1mp_resolution(img: Image.Image) -> tuple[int, int]:
    w, h = img.size
    if w <= 0 or h <= 0:
        return 1024, 1024
    aspect = w / h
    target_area = 1024.0 * 1024.0
    raw_w = math.sqrt(target_area * aspect)
    raw_h = math.sqrt(target_area / aspect)
    snap_w = max(512, min(2048, int(round(raw_w / 64.0) * 64)))
    snap_h = max(512, min(2048, int(round(raw_h / 64.0) * 64)))
    return snap_w, snap_h


class AnimaRefAdapterForForge(scripts.Script):
    sorting_priority = 530

    def __init__(self):
        super().__init__()
        self.cache: dict = {}
        self.adapter: Optional[_Adapter] = None
        self._last_letterboxed_ref: Optional[Image.Image] = None
        self._last_letterboxed_list: list[Image.Image] = []
        self._tab_components: dict[str, dict[str, gr.components.Component]] = {
            "txt2img": {},
            "img2img": {},
        }
        self._pending_tab_wirings: dict[str, dict] = {}

    def title(self):
        return "Anima Ref Adapter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        elem_id = getattr(component, "elem_id", None) or kwargs.get("elem_id")
        if not elem_id:
            return
        for tab in ("txt2img", "img2img"):
            for key in ("sampling", "scheduler", "steps", "cfg_scale", "width", "height"):
                if elem_id == f"{tab}_{key}":
                    self._tab_components[tab][key] = component
                    self._try_bind_tab_actions(tab)

    def _try_bind_tab_actions(self, tab: str):
        wiring = self._pending_tab_wirings.get(tab)
        if not wiring:
            return
        comps = self._tab_components.get(tab, {})

        if not wiring.get("bound_sampler") and all(k in comps for k in ("sampling", "scheduler", "steps", "cfg_scale")):
            wiring["bound_sampler"] = True
            action_feedback = wiring.get("action_feedback")
            outputs = [
                comps["sampling"],
                comps["scheduler"],
                comps["steps"],
                comps["cfg_scale"],
            ]
            if action_feedback is not None:
                outputs.append(action_feedback)
                fn_sampler = lambda: (
                    gr.update(value="Euler"),
                    gr.update(value="Simple"),
                    gr.update(value=28),
                    gr.update(value=4.5),
                    gr.update(value="✓ Applied: Euler / Simple / 28 steps / CFG 4.5"),
                )
            else:
                fn_sampler = lambda: (
                    gr.update(value="Euler"),
                    gr.update(value="Simple"),
                    gr.update(value=28),
                    gr.update(value=4.5),
                )
            wiring["apply_defaults_btn"].click(
                fn=fn_sampler,
                inputs=[],
                outputs=outputs,
                queue=False,
                show_progress=False,
            )

        if not wiring.get("bound_dims") and all(k in comps for k in ("width", "height")):
            wiring["bound_dims"] = True
            match_ar_btn = wiring["match_ar_btn"]
            ref_images = wiring["ref_images"]
            fill = wiring["fill"]
            preview_toggle = wiring["preview_toggle"]
            letterbox_preview = wiring["letterbox_preview"]
            action_feedback = wiring.get("action_feedback")

            outputs = [comps["width"], comps["height"], letterbox_preview]
            if action_feedback is not None:
                outputs.append(action_feedback)

            def _match_aspect(*args):
                imgs = args[:-2]
                fill_val = args[-2]
                show_prev = args[-1]

                pil_img = None
                for candidate in imgs:
                    p = _coerce_pil_image(candidate)
                    if p is not None:
                        pil_img = p
                        break

                if pil_img is None:
                    if action_feedback is not None:
                        return gr.skip(), gr.skip(), gr.skip(), gr.update(value="⚠️ Upload a reference image first")
                    return gr.skip(), gr.skip(), gr.skip()
                w, h = _calc_anima_1mp_resolution(pil_img)
                prev_update = gr.skip()
                if show_prev:
                    _, lb_pil = letterbox_image(pil_img, w, h, int(fill_val))
                    prev_update = gr.update(value=lb_pil, visible=True)
                mp = (w * h) / 1_000_000.0
                if action_feedback is not None:
                    return gr.update(value=w), gr.update(value=h), prev_update, gr.update(value=f"✓ Canvas set to {w} × {h} (~{mp:.2f} MP)")
                return gr.update(value=w), gr.update(value=h), prev_update

            match_ar_btn.click(
                fn=_match_aspect,
                inputs=[*ref_images, fill, preview_toggle],
                outputs=outputs,
                queue=False,
                show_progress=False,
            )

            def _update_letterbox_preview(*args):
                imgs = args[:-4]
                show_prev = args[-4]
                w = args[-3]
                h = args[-2]
                fill_val = args[-1]

                if not show_prev:
                    return gr.update(visible=False)
                pil_img = None
                for candidate in imgs:
                    p = _coerce_pil_image(candidate)
                    if p is not None:
                        pil_img = p
                        break
                if pil_img is None:
                    return gr.update(value=None, visible=True)
                _, lb_pil = letterbox_image(pil_img, int(w or 1024), int(h or 1024), int(fill_val))
                return gr.update(value=lb_pil, visible=True)

            preview_inputs = [*ref_images, preview_toggle, comps["width"], comps["height"], fill]
            for trig in [preview_toggle.change, fill.release] + [img.change for img in ref_images]:
                trig(
                    fn=_update_letterbox_preview,
                    inputs=preview_inputs,
                    outputs=[letterbox_preview],
                    queue=False,
                    show_progress=False,
                )

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        lobe_active = _is_lobe_theme_active()
        theme_class = "anima-ui-lobe" if lobe_active else "anima-ui-classic"

        lora_choices = list(list_lora_files().keys())
        default_lora = _default_lora_choice(lora_choices)
        active_lora_path = find_lora_path(default_lora or V2_LORA_FILENAME)
        lora_ok = bool(active_lora_path and _is_valid_safetensors(active_lora_path) and is_anima_adapter(active_lora_path))

        with InputAccordion(
            False,
            label=self.title(),
            elem_id=self.elem_id("enable"),
            elem_classes=["anima-ref-accordion", theme_class],
        ) as enable:
            with gr.Group(
                elem_id=self.elem_id("root"),
                elem_classes=["anima-ref-root", theme_class],
            ):
                with gr.Row(elem_classes=["anima-workspace-row"], equal_height=False):
                    with gr.Column(scale=5, elem_classes=["anima-col-ref"]):
                        ref_images = []
                        ref_weights = []
                        with gr.Tabs(elem_classes=["anima-ref-tabs"]):
                            for idx in range(1, 5):
                                tab_title = "Ref 1 (Left / Main)" if idx == 1 else f"Ref {idx} (Right)" if idx == 2 else f"Ref {idx}"
                                with gr.TabItem(tab_title, elem_id=self.elem_id(f"tab_ref{idx}")):
                                    img_comp = gr.Image(
                                        label=f"Reference {idx}",
                                        show_label=False,
                                        type="pil",
                                        sources=["upload", "clipboard"],
                                        height=210 if lobe_active else 230,
                                        show_download_button=False,
                                        show_share_button=False,
                                        elem_id=self.elem_id(f"ref_image_{idx}"),
                                        elem_classes=["anima-ref-image"],
                                    )
                                    ref_images.append(img_comp)
                                    with gr.Row(elem_classes=["anima-weight-row"]):
                                        w_comp = gr.Slider(
                                            minimum=0.0,
                                            maximum=2.0,
                                            value=1.0,
                                            step=0.05,
                                            label=f"Weight",
                                            elem_id=self.elem_id(f"ref_weight_{idx}"),
                                            elem_classes=["anima-ref-weight-slider"],
                                        )
                                        ref_weights.append(w_comp)

                        with gr.Row(elem_classes=["anima-multi-mode-row"]):
                            combine_mode = gr.Radio(
                                choices=["stitch", "attention", "blend"],
                                value="stitch",
                                label="Multi-ref mode",
                                elem_id=self.elem_id("combine_mode"),
                                elem_classes=["anima-segmented-radio"],
                                tooltip=(
                                    "stitch: glue references side-by-side (Ref 1 on the left, Ref 2 on the right). Best for 2 characters!\n"
                                    "attention: each reference is its own frame (pilot LoRA).\n"
                                    "blend: weighted latent blend."
                                ),
                            )

                        ref_info = gr.Markdown(
                            value="*No reference image loaded*",
                            elem_id=self.elem_id("ref_info"),
                            elem_classes=["anima-ref-info"],
                        )
                        with gr.Row(elem_classes=["anima-util-row"]):
                            apply_defaults_btn = gr.Button(
                                "⚡ Optimal Sampler",
                                size="sm",
                                elem_id=self.elem_id("apply_defaults"),
                                tooltip="Set Euler sampler, Simple schedule, 28 steps, CFG 4.5",
                            )
                            match_ar_btn = gr.Button(
                                "📐 Match Aspect (~1MP)",
                                size="sm",
                                elem_id=self.elem_id("match_ar"),
                                tooltip="Adjust canvas dimensions to match primary reference aspect ratio at ~1 megapixel",
                            )
                        action_feedback = gr.Markdown(
                            value="",
                            elem_id=self.elem_id("action_feedback"),
                            elem_classes=["anima-action-feedback"],
                        )

                    with gr.Column(scale=7, elem_classes=["anima-col-controls"]):
                        with FormRow(elem_classes=["anima-lora-row"]):
                            lora_name = gr.Dropdown(
                                choices=lora_choices,
                                value=default_lora,
                                label="Adapter LoRA",
                                allow_custom_value=False,
                                elem_id=self.elem_id("lora_name"),
                            )
                            refresh_btn = ToolButton(
                                value="\U0001f504",
                                elem_id=self.elem_id("refresh_loras"),
                                tooltip="Refresh LoRA list",
                            )
                            download_tool_btn = ToolButton(
                                value="\u2b07",
                                elem_id=self.elem_id("download_pilot_tool"),
                                tooltip=f"Download {V2_LORA_FILENAME} from Hugging Face",
                            )

                        download_btn = gr.Button(
                            f"Download {V2_LORA_FILENAME}",
                            size="sm",
                            visible=not lora_ok,
                            elem_id=self.elem_id("download_pilot"),
                        )
                        download_status = gr.Markdown(
                            value="" if lora_ok else f"`{default_lora or V2_LORA_FILENAME}` not found in models folder.",
                            visible=not lora_ok,
                            elem_id=self.elem_id("lora_status"),
                            elem_classes=["anima-status-text"],
                        )

                        with gr.Row(elem_classes=["anima-presets-header"]):
                            gr.Markdown("**Presets**", elem_classes=["anima-presets-label"])
                            preset_status = gr.Markdown(
                                "**Balanced**",
                                elem_id=self.elem_id("preset_status"),
                                elem_classes=["anima-preset-badge"],
                            )
                        with gr.Row(elem_classes=["anima-presets-row"]):
                            btn_balanced = gr.Button("Balanced", size="sm", elem_classes=["anima-preset-btn"])
                            btn_fast = gr.Button("Turbo / Fast", size="sm", elem_classes=["anima-preset-btn"])
                            btn_strong = gr.Button("Strong likeness", size="sm", elem_classes=["anima-preset-btn"])
                            btn_rescale = gr.Button("High guidance", size="sm", elem_classes=["anima-preset-btn"])

                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            strength = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Strength",
                                elem_id=self.elem_id("strength"),
                            )
                            ref_guidance = gr.Slider(
                                minimum=0.0,
                                maximum=20.0,
                                value=4.5,
                                step=0.1,
                                label="Ref guidance",
                                elem_id=self.elem_id("ref_guidance"),
                                info="1 = faster, no guidance branch; higher = closer to reference",
                            )

                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            ref_end = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.01,
                                label="Ref guidance end",
                                elem_id=self.elem_id("ref_end"),
                                info="~0.6 is faster with virtually identical likeness",
                            )
                            ref_megapixels = gr.Slider(
                                minimum=0.25,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Ref size (MP)",
                                elem_id=self.elem_id("ref_megapixels"),
                                info="Resolution for v2 adapter (1.0 MP = 1024x1024 equivalent)",
                            )

                        with gr.Row(elem_classes=["anima-correction-row"]):
                            correction = gr.Radio(
                                choices=["apg", "rescale", "none"],
                                value="apg",
                                label="Guidance correction",
                                elem_id=self.elem_id("correction"),
                                elem_classes=["anima-segmented-radio"],
                            )
                            rescale_phi = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Rescale phi",
                                visible=False,
                                elem_id=self.elem_id("rescale_phi"),
                            )

                        with gr.Accordion(
                            "Advanced",
                            open=False,
                            elem_id=self.elem_id("framing_accordion"),
                            elem_classes=["anima-sub-accordion"],
                        ):
                            with gr.Row(elem_classes=["anima-sliders-row"]):
                                fill = gr.Slider(
                                    minimum=0,
                                    maximum=255,
                                    value=255,
                                    step=1,
                                    label="Letterbox fill (0–255)",
                                    elem_id=self.elem_id("fill"),
                                )
                                ref_offset_h = gr.Slider(
                                    minimum=0.0,
                                    maximum=4.0,
                                    value=1.0,
                                    step=0.05,
                                    label="RoPE offset H",
                                    elem_id=self.elem_id("ref_offset_h"),
                                )
                            with gr.Row(elem_classes=["anima-checks-row"]):
                                preview_toggle = gr.Checkbox(
                                    value=False,
                                    label="Preview letterbox / sheet",
                                    elem_id=self.elem_id("preview_toggle"),
                                )
                                include_ref_in_gallery = gr.Checkbox(
                                    value=False,
                                    label="Append reference sheet to gallery",
                                    elem_id=self.elem_id("include_ref_in_gallery"),
                                )
                            letterbox_preview = gr.Image(
                                label="Reference preview",
                                show_label=False,
                                type="pil",
                                interactive=False,
                                visible=False,
                                height=200,
                                show_download_button=False,
                                show_share_button=False,
                                elem_id=self.elem_id("letterbox_preview"),
                                elem_classes=["anima-letterbox-preview"],
                            )

                with gr.Accordion(
                    "Anima Safe PAG",
                    open=False,
                    elem_id=self.elem_id("pag_accordion"),
                    elem_classes=["anima-sub-accordion"],
                ):
                    pag_enable = gr.Checkbox(
                        value=False,
                        label="Enable Anima Safe PAG",
                        elem_id=self.elem_id("pag_enable"),
                    )
                    with gr.Group(visible=False, elem_id=self.elem_id("pag_body")) as pag_body:
                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            pag_scale = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                value=1.3,
                                step=0.05,
                                label="PAG scale",
                                elem_id=self.elem_id("pag_scale"),
                            )
                            pag_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.75,
                                step=0.05,
                                label="Perturbation strength",
                                elem_id=self.elem_id("pag_strength"),
                            )
                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            pag_start = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.05,
                                label="Start percent",
                                elem_id=self.elem_id("pag_start"),
                            )
                            pag_end = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.05,
                                label="End percent",
                                elem_id=self.elem_id("pag_end"),
                            )
                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            pag_rescale = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.2,
                                step=0.05,
                                label="PAG rescale",
                                elem_id=self.elem_id("pag_rescale"),
                            )
                            pag_rescale_mode = gr.Radio(
                                choices=["full", "partial"],
                                value="full",
                                label="PAG rescale mode",
                                elem_id=self.elem_id("pag_rescale_mode"),
                            )
                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            pag_blocks = gr.Textbox(
                                value="18",
                                label="Block indices",
                                placeholder="18 or 18-20",
                                elem_id=self.elem_id("pag_blocks"),
                            )
                            pag_heads = gr.Textbox(
                                value="",
                                label="Head indices (optional)",
                                placeholder="All heads",
                                elem_id=self.elem_id("pag_heads"),
                            )

        def _on_images_change(*imgs):
            loaded = []
            for idx, c in enumerate(imgs, 1):
                p = _coerce_pil_image(c)
                if p is not None:
                    loaded.append((idx, p))
            if not loaded:
                return "*No reference image loaded*"

            primary_idx, primary_pil = loaded[0]
            w, h = primary_pil.size
            rec_w, rec_h = _calc_anima_1mp_resolution(primary_pil)
            gcd = math.gcd(w, h)
            ar_str = (
                f"{w // gcd}:{h // gcd}"
                if gcd > 0 and (w // gcd) <= 32 and (h // gcd) <= 32
                else f"{w/h:.2f}:1"
            )
            mp = (w * h) / 1_000_000.0
            rec_mp = (rec_w * rec_h) / 1_000_000.0

            if len(loaded) == 1:
                return (
                    f"**Ref {primary_idx}**: {w} × {h} ({ar_str}, {mp:.2f} MP) ➔ "
                    f"Recommended canvas: **{rec_w} × {rec_h}** ({rec_mp:.2f} MP)"
                )
            else:
                slots_str = ", ".join(f"#{i}" for i, _ in loaded)
                return (
                    f"✓ **{len(loaded)} reference images active** (slots {slots_str}) ➔ "
                    f"Canvas: **{rec_w} × {rec_h}** ({rec_mp:.2f} MP)"
                )

        for img_comp in ref_images:
            img_comp.change(
                fn=_on_images_change,
                inputs=ref_images,
                outputs=[ref_info],
                queue=False,
                show_progress=False,
            )

        # Presets
        btn_balanced.click(
            fn=lambda: (1.0, 4.5, 1.0, "apg", 0.5, gr.update(visible=False), "**Balanced**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )
        btn_fast.click(
            fn=lambda: (1.0, 4.5, 0.6, "apg", 0.5, gr.update(visible=False), "**Turbo / Fast**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )
        btn_strong.click(
            fn=lambda: (1.1, 6.0, 1.0, "apg", 0.5, gr.update(visible=False), "**Strong likeness**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )
        btn_rescale.click(
            fn=lambda: (1.0, 6.5, 1.0, "rescale", 0.7, gr.update(visible=True), "**High guidance**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )

        for sld in (strength, ref_guidance, ref_end):
            sld.release(fn=lambda: "*Custom*", inputs=[], outputs=[preset_status], queue=False)

        correction.change(
            fn=lambda val: gr.update(visible=(str(val).lower() == "rescale")),
            inputs=[correction],
            outputs=[rescale_phi],
            queue=False,
            show_progress=False,
        )

        pag_enable.change(
            fn=lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[pag_enable],
            outputs=[pag_body],
            queue=False,
            show_progress=False,
        )

        def _on_lora_change(selected_lora):
            path = find_lora_path(selected_lora) if selected_lora else None
            valid = bool(path and _is_valid_safetensors(path) and is_anima_adapter(path))
            if valid:
                return gr.update(visible=False), gr.update(value="", visible=False)
            elif path and not is_anima_adapter(path):
                return (
                    gr.update(visible=True),
                    gr.update(value=f"⚠️ `{os.path.basename(path)}` is not an Anima adapter. Digital noise will occur.", visible=True),
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(value=f"`{selected_lora or V2_LORA_FILENAME}` not found.", visible=True),
                )

        lora_name.change(
            fn=_on_lora_change,
            inputs=[lora_name],
            outputs=[download_btn, download_status],
            queue=False,
            show_progress=False,
        )

        def _refresh_loras(current_val):
            choices = list(list_lora_files().keys())
            path = find_lora_path(current_val) if current_val else None
            val = current_val if (path and _is_valid_safetensors(path) and is_anima_adapter(path)) else _default_lora_choice(choices)
            valid = bool(val and find_lora_path(val) and is_anima_adapter(find_lora_path(val)))
            return (
                gr.update(choices=choices, value=val),
                gr.update(visible=not valid),
                gr.update(value="" if valid else f"`{val or V2_LORA_FILENAME}` not found.", visible=not valid),
            )

        refresh_btn.click(
            fn=_refresh_loras,
            inputs=[lora_name],
            outputs=[lora_name, download_btn, download_status],
            queue=False,
            show_progress=False,
        )

        def _download_adapter():
            try:
                dest, msg = download_adapter_lora(V2_LORA_FILENAME)
                choices = list(list_lora_files().keys())
                val = _default_lora_choice(choices)
                return (
                    gr.update(choices=choices, value=val),
                    gr.update(visible=False),
                    gr.update(value=f"Downloaded into extension folder: `{dest}`", visible=True),
                )
            except Exception as exc:
                return (
                    gr.skip(),
                    gr.update(visible=True),
                    gr.update(value=f"Download failed: {exc}", visible=True),
                )

        for btn in (download_btn, download_tool_btn):
            btn.click(
                fn=_download_adapter,
                inputs=[],
                outputs=[lora_name, download_btn, download_status],
                show_progress=True,
            )

        self._pending_tab_wirings[tab] = {
            "apply_defaults_btn": apply_defaults_btn,
            "match_ar_btn": match_ar_btn,
            "ref_images": ref_images,
            "fill": fill,
            "preview_toggle": preview_toggle,
            "letterbox_preview": letterbox_preview,
            "action_feedback": action_feedback,
        }
        self._try_bind_tab_actions(tab)

        self.infotext_fields = [
            (lora_name, "Anima Ref Adapter LoRA"),
            (strength, "Anima Ref Strength"),
            (ref_guidance, "Anima Ref Guidance"),
            (ref_end, "Anima Ref End"),
            (ref_megapixels, "Anima Ref Megapixels"),
            (correction, "Anima Ref Correction"),
            (combine_mode, "Anima Ref Combine Mode"),
            (rescale_phi, "Anima Ref Rescale Phi"),
            (ref_offset_h, "Anima Ref Offset H"),
            (fill, "Anima Ref Fill"),
            (pag_enable, "Anima Safe PAG"),
            (pag_scale, "Anima Safe PAG Scale"),
            (pag_blocks, "Anima Safe PAG Blocks"),
            (pag_strength, "Anima Safe PAG Strength"),
            (pag_start, "Anima Safe PAG Start"),
            (pag_end, "Anima Safe PAG End"),
            (pag_rescale, "Anima Safe PAG Rescale"),
            (pag_rescale_mode, "Anima Safe PAG Rescale Mode"),
        ]

        return [
            enable,
            ref_images[0],
            ref_weights[0],
            ref_images[1],
            ref_weights[1],
            ref_images[2],
            ref_weights[2],
            ref_images[3],
            ref_weights[3],
            combine_mode,
            lora_name,
            strength,
            ref_guidance,
            correction,
            rescale_phi,
            ref_offset_h,
            fill,
            include_ref_in_gallery,
            pag_enable,
            pag_scale,
            pag_blocks,
            pag_strength,
            pag_start,
            pag_end,
            pag_rescale,
            pag_rescale_mode,
            pag_heads,
            ref_end,
            ref_megapixels,
        ]

    def process_before_every_sampling(
        self,
        p,
        enable: bool,
        ref_image1,
        ref_weight1: float,
        ref_image2,
        ref_weight2: float,
        ref_image3,
        ref_weight3: float,
        ref_image4,
        ref_weight4: float,
        combine_mode: str,
        lora_name: str,
        strength: float,
        ref_guidance: float,
        correction: str,
        rescale_phi: float,
        ref_offset_h: float,
        fill: int,
        include_ref_in_gallery: bool,
        pag_enable: bool,
        pag_scale: float,
        pag_blocks: str,
        pag_strength: float,
        pag_start: float,
        pag_end: float,
        pag_rescale: float,
        pag_rescale_mode: str,
        pag_heads: str,
        ref_end: float = 1.0,
        ref_megapixels: float = 1.0,
        *args,
        **kwargs,
    ):
        self._last_letterboxed_ref = None
        self._last_letterboxed_list = []
        if not enable:
            return

        sampler_name = (getattr(p, "sampler_name", "") or "").lower()
        if any(bad in sampler_name for bad in ("sde", "ancestral", "euler a")):
            logger.warning(
                f"[Anima Ref Adapter] Stochastic sampler '{p.sampler_name}' detected! "
                f"SDE/ancestral samplers inject random noise that degrades Flow Matching reference guidance into digital noise. "
                f"Please switch to 'Euler' with 'Simple' schedule."
            )
            gr.Warning(
                f"[Anima Ref Adapter] Sampler '{p.sampler_name}' produces digital noise! Switch to 'Euler' with 'Simple' schedule."
            )

        dm = p.sd_model.forge_objects.unet.get_model_object("diffusion_model")
        if dm.__class__.__name__ != "Anima" and not getattr(dynamic_args, "anima", False):
            logger.warning(
                f"[Anima Ref Adapter] Current diffusion model is '{dm.__class__.__name__}', not Anima. Skipping."
            )
            return

        unet = p.sd_model.forge_objects.unet

        active_refs = []
        for img, weight in [
            (ref_image1, ref_weight1),
            (ref_image2, ref_weight2),
            (ref_image3, ref_weight3),
            (ref_image4, ref_weight4),
        ]:
            pil = _coerce_pil_image(img)
            if pil is not None and float(weight or 0) > 0:
                active_refs.append((pil, float(weight)))

        if active_refs:
            lora_path = find_lora_path(lora_name)
            if lora_path is None or not is_anima_adapter(lora_path):
                bad_name = os.path.basename(lora_path) if lora_path else (lora_name or "None")
                msg = (
                    f"Selected LoRA '{bad_name}' is not an Anima Reference Adapter. "
                    f"Using it will produce digital noise. Please select '{V2_LORA_FILENAME}'."
                )
                logger.error(f"[Anima Ref Adapter] {msg}")
                raise gr.Error(f"[Anima Ref Adapter] {msg}")
            else:
                dynamic_args.ref_latents.clear()

                x = kwargs.get("x")
                if x is not None and hasattr(x, "shape") and len(x.shape) >= 4:
                    target_h = int(x.shape[-2]) * 8
                    target_w = int(x.shape[-1]) * 8
                else:
                    target_w = int(p.hr_upscale_to_x if getattr(p, "is_hr_pass", False) else p.width)
                    target_h = int(p.hr_upscale_to_y if getattr(p, "is_hr_pass", False) else p.height)

                vae = p.sd_model.forge_objects.vae
                if combine_mode not in ("stitch", "attention", "blend"):
                    combine_mode = "stitch"

                # Check if adapter is v2 asymmetric
                num_blocks = len(getattr(dm, "blocks", [])) or 28
                adapter = _Adapter.get(lora_path, float(strength), num_blocks=num_blocks)
                self.adapter = adapter
                self.cache.clear()

                ref_latents, letterboxed_list = encode_multi_reference_latents(
                    images_with_weights=active_refs,
                    vae=vae,
                    width=target_w,
                    height=target_h,
                    fill=int(fill),
                    combine_mode=combine_mode,
                    asym=adapter.asym,
                    megapixels=float(ref_megapixels),
                )
                if include_ref_in_gallery:
                    self._last_letterboxed_list = letterboxed_list
                    self._last_letterboxed_ref = letterboxed_list[0] if letterboxed_list else None

                try:
                    config = AnimaRefAdapterConfig(
                        lora_path=lora_path,
                        strength=float(strength),
                        ref_offset_h=float(ref_offset_h),
                        ref_guidance=float(ref_guidance),
                        correction=str(correction),
                        rescale_phi=float(rescale_phi),
                        fill=int(fill),
                        combine_mode=combine_mode,
                        ref_end=float(ref_end),
                        ref_megapixels=float(ref_megapixels),
                    )
                    unet = patch_unet_for_anima_ref(unet, ref_latents, config, cache=self.cache)
                except Exception as exc:
                    logger.error(f"[Anima Ref Adapter] {exc}")
                    gr.Warning(f"[Anima Ref Adapter] {exc}")
                    return

                p.extra_generation_params["Anima Ref Adapter LoRA"] = os.path.basename(lora_path)
                p.extra_generation_params["Anima Ref Strength"] = float(strength)
                p.extra_generation_params["Anima Ref Guidance"] = float(ref_guidance)
                p.extra_generation_params["Anima Ref End"] = float(ref_end)
                p.extra_generation_params["Anima Ref Megapixels"] = float(ref_megapixels)
                p.extra_generation_params["Anima Ref Correction"] = str(correction)
                p.extra_generation_params["Anima Ref Combine Mode"] = combine_mode
                p.extra_generation_params["Anima Ref Images"] = len(active_refs)
                p.extra_generation_params["Anima Ref Asym"] = bool(adapter.asym)
                if str(correction).lower() == "rescale":
                    p.extra_generation_params["Anima Ref Rescale Phi"] = float(rescale_phi)
                p.extra_generation_params["Anima Ref Offset H"] = float(ref_offset_h)
                p.extra_generation_params["Anima Ref Fill"] = int(fill)
        else:
            if not pag_enable:
                logger.warning("[Anima Ref Adapter] Enabled, but no reference image was provided.")
                return

        if pag_enable and float(pag_scale) > 0:
            pag_cfg = AnimaSafePAGConfig(
                enabled=True,
                scale=float(pag_scale),
                block_indices=str(pag_blocks),
                perturbation_strength=float(pag_strength),
                head_indices=str(pag_heads or ""),
                start_percent=float(pag_start),
                end_percent=float(pag_end),
                rescale=float(pag_rescale),
                rescale_mode=str(pag_rescale_mode),
            )
            unet = patch_unet_for_safe_pag(unet, pag_cfg)
            p.extra_generation_params["Anima Safe PAG"] = True
            p.extra_generation_params["Anima Safe PAG Scale"] = float(pag_scale)
            p.extra_generation_params["Anima Safe PAG Blocks"] = str(pag_blocks)
            p.extra_generation_params["Anima Safe PAG Strength"] = float(pag_strength)
            p.extra_generation_params["Anima Safe PAG Start"] = float(pag_start)
            p.extra_generation_params["Anima Safe PAG End"] = float(pag_end)
            p.extra_generation_params["Anima Safe PAG Rescale"] = float(pag_rescale)
            p.extra_generation_params["Anima Safe PAG Rescale Mode"] = str(pag_rescale_mode)

        p.sd_model.forge_objects.unet = unet

    def post_sample(self, p, ps, *args):
        if self.adapter is not None:
            self.cache.clear()
            self.adapter.offload()
            self.adapter = None
            memory_management.soft_empty_cache()

    def postprocess(self, p, processed, *args):
        if getattr(self, "_last_letterboxed_list", None):
            for lb in self._last_letterboxed_list:
                processed.extra_images.append(lb)
            self._last_letterboxed_list = []
            self._last_letterboxed_ref = None
        elif self._last_letterboxed_ref is not None:
            processed.extra_images.append(self._last_letterboxed_ref)
            self._last_letterboxed_ref = None
