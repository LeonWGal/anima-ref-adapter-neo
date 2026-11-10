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
from backend.nn.anima import Anima
from lib_anima_ref import (
    PILOT_LORA_FILENAME,
    V2_LORA_FILENAME,
    AnimaRefAdapterConfig,
    AnimaSafePAGConfig,
    _Adapter,
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
from lib_anima_ref.adapter import DOWNLOADABLE_FILES, SNAPREF_FILES, anima_ref_metadata, is_snapref_adapter, logger
from lib_anima_ref.snapref import encode_snapref_image, patch_unet_for_snapref
from modules import scripts, shared
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion


V2_LAYOUTS = [("Side by side", "stitch"), ("Latent blend", "blend")]
PILOT_LAYOUTS = [*V2_LAYOUTS, ("Separate frames", "attention")]


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


def _lora_choices() -> list[str]:
    return list(dict.fromkeys((*DOWNLOADABLE_FILES, *list_lora_files())))


def _adapter_label(name: str) -> str:
    if name in SNAPREF_FILES:
        version = name.removeprefix("snapref_").removesuffix(".safetensors")
        precision = "FP8" if version.endswith("_fp8") else "BF16"
        return f"SnapRef {version.removesuffix('_fp8').replace('_', '-')} · {precision}"
    if name == V2_LORA_FILENAME:
        return "Anima Ref v2"
    if name == PILOT_LORA_FILENAME:
        return "Anima Ref Pilot"
    return name


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
                    gr.update(value="ER SDE"),
                    gr.update(value="Simple"),
                    gr.update(value=30),
                    gr.update(value=4.5),
                    gr.update(value="✓ Applied: ER SDE / Simple / 30 steps / CFG 4.5"),
                )
            else:
                fn_sampler = lambda: (
                    gr.update(value="ER SDE"),
                    gr.update(value="Simple"),
                    gr.update(value=30),
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
            ref_weights = wiring["ref_weights"]
            fill = wiring["fill"]
            preview_toggle = wiring["preview_toggle"]
            letterbox_preview = wiring["letterbox_preview"]
            action_feedback = wiring.get("action_feedback")

            outputs = [comps["width"], comps["height"], letterbox_preview]
            if action_feedback is not None:
                outputs.append(action_feedback)

            def _primary_reference(args):
                for candidate, weight in zip(args[:4], args[4:8]):
                    if float(weight or 0) > 0 and (image := _coerce_pil_image(candidate)) is not None:
                        return image
                return None

            def _match_aspect(*args):
                fill_val = args[-2]
                show_prev = args[-1]
                pil_img = _primary_reference(args)

                if pil_img is None:
                    if action_feedback is not None:
                        return gr.skip(), gr.skip(), gr.skip(), gr.update(value="Load a reference with a weight above 0 to match its aspect ratio.")
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
                inputs=[*ref_images, *ref_weights, fill, preview_toggle],
                outputs=outputs,
                queue=False,
                show_progress=False,
            )

            def _update_letterbox_preview(*args):
                show_prev = args[-4]
                w = args[-3]
                h = args[-2]
                fill_val = args[-1]

                if not show_prev:
                    return gr.update(visible=False)
                pil_img = _primary_reference(args)
                if pil_img is None:
                    return gr.update(value=None, visible=True)
                _, lb_pil = letterbox_image(pil_img, int(w or 1024), int(h or 1024), int(fill_val))
                return gr.update(value=lb_pil, visible=True)

            preview_inputs = [*ref_images, *ref_weights, preview_toggle, comps["width"], comps["height"], fill]
            for trig in [preview_toggle.change, fill.release] + [component.change for component in [*ref_images, *ref_weights]]:
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

        lora_choices = _lora_choices()
        active_lora_path = find_lora_path(V2_LORA_FILENAME)
        lora_ok = is_anima_adapter(active_lora_path)

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
                with gr.Group(elem_classes=["anima-adapter-header"]):
                    lora_name = gr.Dropdown(
                        choices=[(_adapter_label(name), name) for name in lora_choices],
                        value=V2_LORA_FILENAME,
                        label="Adapter LoRA",
                        allow_custom_value=False,
                        elem_id=self.elem_id("lora_name"),
                    )
                    with gr.Row(elem_classes=["anima-adapter-status-row"]):
                        download_status = gr.Markdown(
                            value="Weights ready" if lora_ok else "Weights download on first use",
                            elem_id=self.elem_id("lora_status"),
                            elem_classes=["anima-status-text"],
                        )
                        refresh_btn = gr.Button(
                            "Refresh",
                            size="sm",
                            elem_id=self.elem_id("refresh_loras"),
                            tooltip="Refresh installed adapters",
                        )
                    download_btn = gr.Button(
                        "Download weights",
                        size="sm",
                        visible=not lora_ok,
                        elem_id=self.elem_id("download_pilot"),
                    )
                    snapref_mode = gr.Radio(
                        choices=[("Character & Style", "cs"), ("Character", "char")],
                        value="cs",
                        label="SnapRef mode",
                        visible=False,
                        elem_id=self.elem_id("snapref_mode"),
                    )

                with gr.Row(elem_classes=["anima-workspace-row"], equal_height=False):
                    with gr.Column(scale=5, elem_classes=["anima-col-ref"]):
                        gr.Markdown("### References", elem_classes=["anima-section-title"])
                        ref_images = []
                        ref_weights = []
                        with gr.Tabs(elem_classes=["anima-ref-tabs"]):
                            for idx in range(1, 5):
                                tab_title = f"Ref {idx}"
                                with gr.TabItem(tab_title, elem_id=self.elem_id(f"tab_ref{idx}")):
                                    img_comp = gr.Image(
                                        label=f"Reference {idx}",
                                        show_label=False,
                                        type="pil",
                                        sources=["upload", "clipboard"],
                                        height=176 if lobe_active else 192,
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
                                            label="Reference weight",
                                            visible=False,
                                            info="Set to 0 to skip this slot.",
                                            elem_id=self.elem_id(f"ref_weight_{idx}"),
                                            elem_classes=["anima-ref-weight-slider"],
                                        )
                                        ref_weights.append(w_comp)

                        slot_state = gr.HTML('<span data-anima-filled=""></span>', elem_classes=["anima-slot-state"])
                        ref_info = gr.Markdown(
                            value="",
                            visible=False,
                            elem_id=self.elem_id("ref_info"),
                            elem_classes=["anima-ref-info"],
                        )
                        with gr.Accordion("Canvas & sampler", open=False, elem_classes=["anima-sub-accordion"]):
                            with gr.Row(elem_classes=["anima-util-row"]):
                                apply_defaults_btn = gr.Button(
                                    "Apply sampler preset",
                                    size="sm",
                                    elem_id=self.elem_id("apply_defaults"),
                                    tooltip="Set ER SDE sampler, Simple schedule, 30 steps, CFG 4.5",
                                )
                                match_ar_btn = gr.Button(
                                    "Match canvas to reference",
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
                        gr.Markdown("### Reference settings", elem_classes=["anima-section-title"])
                        with gr.Group(visible=False, elem_classes=["anima-snapref-settings"]) as snapref_controls:
                            snapref_layout = gr.Radio(
                                choices=[("Separate", "separate"), ("Side by side", "stitch")],
                                value="separate",
                                label="SnapRef reference layout",
                                info="Use Separate for views of one character and Side by side for two characters. Weight 0 skips a slot.",
                                elem_id=self.elem_id("snapref_layout"),
                            )

                        with gr.Row(elem_classes=["anima-multi-mode-row"]) as combine_row:
                            combine_mode = gr.Radio(
                                choices=V2_LAYOUTS,
                                value="stitch",
                                label="Reference layout",
                                elem_id=self.elem_id("combine_mode"),
                                tooltip=(
                                    "stitch: glue references side-by-side (Ref 1 on the left, Ref 2 on the right). Best for 2 characters!\n"
                                    "attention: each reference is its own frame (pilot LoRA).\n"
                                    "blend: weighted latent blend."
                                ),
                            )

                        with gr.Row(elem_classes=["anima-presets-header"]) as presets_header:
                            gr.Markdown("**Presets**", elem_classes=["anima-presets-label"])
                            preset_status = gr.Markdown(
                                "**Balanced**",
                                elem_id=self.elem_id("preset_status"),
                                elem_classes=["anima-preset-badge"],
                            )
                        with gr.Row(elem_classes=["anima-presets-row"]) as presets_row:
                            btn_balanced = gr.Button("Balanced", size="sm", elem_classes=["anima-preset-btn"])
                            btn_fast = gr.Button("Turbo / Fast", size="sm", elem_classes=["anima-preset-btn"])
                            btn_strong = gr.Button("Strong likeness", size="sm", elem_classes=["anima-preset-btn"])
                            btn_rescale = gr.Button("High guidance", size="sm", elem_classes=["anima-preset-btn"])

                        with gr.Row(elem_classes=["anima-sliders-row"]) as guidance_row:
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
                                info="0.6 skips the extra guidance pass in later steps",
                            )
                            ref_megapixels = gr.Slider(
                                minimum=0.25,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Ref size (MP)",
                                elem_id=self.elem_id("ref_megapixels"),
                                info="Reference area for v2 and SnapRef (1.0 MP ≈ 1024 × 1024)",
                            )

                        with gr.Row(elem_classes=["anima-correction-row"]) as correction_row:
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

                        include_ref_in_gallery = gr.Checkbox(
                            value=False,
                            label="Show processed references in gallery",
                            elem_id=self.elem_id("include_ref_in_gallery"),
                        )

                        with gr.Accordion(
                            "Advanced",
                            open=False,
                            elem_id=self.elem_id("framing_accordion"),
                            elem_classes=["anima-sub-accordion"],
                        ) as framing:
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

        def _on_images_change(*values):
            images, weights = values[:4], values[4:]
            loaded = [index for index, image in enumerate(images, 1) if image is not None]
            active = [index for index in loaded if float(weights[index - 1] or 0) > 0]
            if active:
                slots = ", ".join(f"Ref {index}" for index in active)
                status = f"**Active references** · {slots}"
            else:
                status = "All loaded references are skipped. Set a weight above 0 to use its image."
            weight_updates = [gr.update(visible=index in loaded) for index in range(1, 5)]
            filled = " ".join(map(str, loaded))
            return gr.update(value=status if loaded else "", visible=bool(loaded)), *weight_updates, f'<span data-anima-filled="{filled}"></span>'

        for component in [*ref_images, *ref_weights]:
            component.change(
                fn=_on_images_change,
                inputs=[*ref_images, *ref_weights],
                outputs=[ref_info, *ref_weights, slot_state],
                queue=False,
                show_progress=False,
            )

        # Presets
        btn_balanced.click(
            fn=lambda: (1.0, 4.5, 1.0, "apg", gr.update(value=0.5, visible=False), "**Balanced**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )
        btn_fast.click(
            fn=lambda: (1.0, 4.5, 0.6, "apg", gr.update(value=0.5, visible=False), "**Turbo / Fast**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )
        btn_strong.click(
            fn=lambda: (1.1, 6.0, 1.0, "apg", gr.update(value=0.5, visible=False), "**Strong likeness**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, preset_status],
            queue=False,
            show_progress=False,
        )
        btn_rescale.click(
            fn=lambda: (1.0, 6.5, 1.0, "rescale", gr.update(value=0.7, visible=True), "**High guidance**"),
            inputs=[],
            outputs=[strength, ref_guidance, ref_end, correction, rescale_phi, preset_status],
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

        def _on_lora_change(selected_lora, layout="stitch"):
            path = find_lora_path(selected_lora) if selected_lora else None
            snapref_file = is_snapref_adapter(path)
            snapref = selected_lora in SNAPREF_FILES or snapref_file
            valid = snapref_file if snapref else is_anima_adapter(path)
            released = selected_lora in DOWNLOADABLE_FILES
            download = gr.update(visible=released and not valid)
            if not selected_lora:
                status = "Select an adapter for your Anima checkpoint."
            elif valid:
                status = "Weights ready"
            elif released:
                status = "Weights download on first use"
            else:
                status = "Adapter unavailable. Restore the file, then refresh, or select another adapter."
            layout_update = gr.skip()
            if not snapref:
                asym = selected_lora == V2_LORA_FILENAME or (valid and anima_ref_metadata(path).get("arch") == "asym")
                choices = V2_LAYOUTS if asym else PILOT_LAYOUTS
                layout_update = gr.update(choices=choices, value=layout if layout in dict(choices).values() else "stitch")
            visibility = [gr.update(visible=not snapref) for _ in legacy_controls]
            return download, gr.update(value=status), gr.update(visible=snapref), *visibility, layout_update, gr.update(visible=snapref)

        legacy_controls = [combine_row, presets_header, presets_row, guidance_row, ref_end, correction_row, framing]
        adapter_controls = [download_btn, download_status, snapref_controls, *legacy_controls, combine_mode, snapref_mode]

        lora_name.change(
            fn=_on_lora_change,
            inputs=[lora_name, combine_mode],
            outputs=adapter_controls,
            queue=False,
            show_progress=False,
        )

        def _refresh_loras(current_val, layout="stitch"):
            choices = _lora_choices()
            if current_val and current_val not in choices:
                choices.append(current_val)
            return gr.update(choices=[(_adapter_label(name), name) for name in choices]), *_on_lora_change(current_val, layout)

        refresh_btn.click(
            fn=_refresh_loras,
            inputs=[lora_name, combine_mode],
            outputs=[lora_name, *adapter_controls],
            queue=False,
            show_progress=False,
        )

        def _download_adapter(selected_lora):
            try:
                dest, _ = download_adapter_lora(selected_lora)
                gr.Info(f"Adapter ready: {os.path.basename(dest)}")
            except Exception as exc:
                gr.Warning(f"Download failed: {exc}")

        download_btn.click(
            fn=_download_adapter,
            inputs=[lora_name],
            outputs=[],
            show_progress=True,
        ).then(
            fn=_on_lora_change,
            inputs=[lora_name, combine_mode],
            outputs=adapter_controls,
            queue=False,
        )

        self._pending_tab_wirings[tab] = {
            "apply_defaults_btn": apply_defaults_btn,
            "match_ar_btn": match_ar_btn,
            "ref_images": ref_images,
            "ref_weights": ref_weights,
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
            (snapref_mode, "Anima SnapRef Mode"),
            (snapref_layout, "Anima SnapRef Layout"),
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
            snapref_mode,
            snapref_layout,
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
        snapref_mode: str = "cs",
        snapref_layout: str = "separate",
        *args,
        **kwargs,
    ):
        self._release_sampling_state()
        self._last_letterboxed_ref = None
        self._last_letterboxed_list = []
        if not enable:
            return

        dm = p.sd_model.forge_objects.unet.get_model_object("diffusion_model")
        if not isinstance(dm, Anima):
            logger.warning(
                f"[Anima Ref Adapter] Current diffusion model is '{dm.__class__.__name__}', not Anima. Skipping."
            )
            gr.Warning("Select an Anima checkpoint to use Anima Ref Adapter.")
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
            lora_name = lora_name or V2_LORA_FILENAME
            lora_path = find_lora_path(lora_name)
            snapref_file = is_snapref_adapter(lora_path)
            snapref = lora_name in SNAPREF_FILES or snapref_file
            lora_ok = snapref_file if snapref else is_anima_adapter(lora_path)
            if lora_name in DOWNLOADABLE_FILES and not lora_ok:
                try:
                    lora_path, _ = download_adapter_lora(lora_name)
                    lora_ok = True
                except RuntimeError as exc:
                    raise gr.Error(str(exc)) from exc
            if not lora_ok:
                bad_name = os.path.basename(lora_path) if lora_path else (lora_name or "None")
                msg = f"Adapter '{bad_name}' is missing or incompatible. Select an Anima Ref or SnapRef adapter."
                logger.error(f"[Anima Ref Adapter] {msg}")
                raise gr.Error(f"[Anima Ref Adapter] {msg}")
            dynamic_args.ref_latents.clear()
            vae = p.sd_model.forge_objects.vae
            if snapref:
                try:
                    megapixels = float(ref_megapixels)
                    layout = "stitch" if snapref_layout == "stitch" else "separate"
                    inputs = active_refs
                    if layout == "stitch" and len(inputs) > 1:
                        _, sheet = build_asym_reference_image(inputs, megapixels)
                        inputs = [(sheet, 1.0)]
                    references, previews = [], []
                    for image, weight in inputs:
                        ref, preview = encode_snapref_image(image, vae, megapixels, 255)
                        references.append((ref, weight if layout == "separate" else 1.0))
                        previews.append(preview)
                    context = p.sd_model.get_learned_conditioning(SdConditioning([""], is_negative_prompt=True))[0]
                    unet = patch_unet_for_snapref(unet, references, context, lora_path, snapref_mode, self.cache)
                except Exception as exc:
                    raise gr.Error(f"[SnapRef] {exc}") from exc
                if include_ref_in_gallery:
                    self._last_letterboxed_list = previews
                p.extra_generation_params["Anima SnapRef Mode"] = snapref_mode
                p.extra_generation_params["Anima SnapRef Layout"] = layout
                p.extra_generation_params["Anima Ref Images"] = len(active_refs)
            else:

                x = kwargs.get("x")
                if x is not None and hasattr(x, "shape") and len(x.shape) >= 4:
                    target_h = int(x.shape[-2]) * 8
                    target_w = int(x.shape[-1]) * 8
                else:
                    target_w = int(p.hr_upscale_to_x if getattr(p, "is_hr_pass", False) else p.width)
                    target_h = int(p.hr_upscale_to_y if getattr(p, "is_hr_pass", False) else p.height)

                if combine_mode not in ("stitch", "attention", "blend"):
                    combine_mode = "stitch"

                # Check if adapter is v2 asymmetric
                num_blocks = len(getattr(dm, "blocks", [])) or 28
                adapter = _Adapter.get(lora_path, float(strength), num_blocks=num_blocks)
                self.adapter = adapter
                if adapter.asym and combine_mode == "attention":
                    combine_mode = "stitch"

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

                p.extra_generation_params["Anima Ref Strength"] = float(strength)
                p.extra_generation_params["Anima Ref Guidance"] = float(ref_guidance)
                p.extra_generation_params["Anima Ref End"] = float(ref_end)
                p.extra_generation_params["Anima Ref Correction"] = str(correction)
                p.extra_generation_params["Anima Ref Combine Mode"] = combine_mode
                p.extra_generation_params["Anima Ref Images"] = len(active_refs)
                p.extra_generation_params["Anima Ref Asym"] = bool(adapter.asym)
                if str(correction).lower() == "rescale":
                    p.extra_generation_params["Anima Ref Rescale Phi"] = float(rescale_phi)
                p.extra_generation_params["Anima Ref Offset H"] = float(ref_offset_h)
                p.extra_generation_params["Anima Ref Fill"] = int(fill)
            p.extra_generation_params["Anima Ref Adapter LoRA"] = os.path.basename(lora_path)
            p.extra_generation_params["Anima Ref Megapixels"] = float(ref_megapixels)
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

    def _release_sampling_state(self):
        had_cache = bool(self.cache)
        self.cache.clear()
        if self.adapter is not None:
            self.adapter.offload()
            self.adapter = None
            had_cache = True
        if had_cache:
            memory_management.soft_empty_cache()

    def post_sample(self, p, ps, *args):
        self._release_sampling_state()

    def postprocess(self, p, processed, *args):
        if getattr(self, "_last_letterboxed_list", None):
            for lb in self._last_letterboxed_list:
                processed.extra_images.append(lb)
            self._last_letterboxed_list = []
            self._last_letterboxed_ref = None
        elif self._last_letterboxed_ref is not None:
            processed.extra_images.append(self._last_letterboxed_ref)
            self._last_letterboxed_ref = None
