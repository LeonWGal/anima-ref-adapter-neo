from __future__ import annotations

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

from backend.args import dynamic_args
from lib_anima_ref import (
    PILOT_LORA_FILENAME,
    AnimaRefAdapterConfig,
    AnimaSafePAGConfig,
    _is_valid_safetensors,
    download_pilot_lora,
    encode_reference_latents,
    find_lora_path,
    letterbox_image,
    list_lora_files,
    patch_unet_for_anima_ref,
    patch_unet_for_safe_pag,
)
from lib_anima_ref.adapter import logger
from modules import scripts, shared
from modules.api import api
from modules.processing import Processed, StableDiffusionProcessing
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
            return Image.open(img)
        return api.decode_base64_to_image(img)
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
        self._last_letterboxed_ref: Optional[Image.Image] = None
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
            wiring["apply_defaults_btn"].click(
                fn=lambda: (
                    gr.update(value="Euler"),
                    gr.update(value="Simple"),
                    gr.update(value=28),
                    gr.update(value=4.5),
                ),
                inputs=[],
                outputs=[
                    comps["sampling"],
                    comps["scheduler"],
                    comps["steps"],
                    comps["cfg_scale"],
                ],
                queue=False,
                show_progress=False,
            )

        if not wiring.get("bound_dims") and all(k in comps for k in ("width", "height")):
            wiring["bound_dims"] = True
            match_ar_btn = wiring["match_ar_btn"]
            ref_image = wiring["ref_image"]
            fill = wiring["fill"]
            preview_toggle = wiring["preview_toggle"]
            letterbox_preview = wiring["letterbox_preview"]

            def _match_aspect(img, fill_val, show_prev):
                pil_img = _coerce_pil_image(img)
                if pil_img is None:
                    return gr.skip(), gr.skip(), gr.skip()
                w, h = _calc_anima_1mp_resolution(pil_img)
                prev_update = gr.skip()
                if show_prev:
                    _, lb_pil = letterbox_image(pil_img, w, h, int(fill_val))
                    prev_update = gr.update(value=lb_pil, visible=True)
                return gr.update(value=w), gr.update(value=h), prev_update

            match_ar_btn.click(
                fn=_match_aspect,
                inputs=[ref_image, fill, preview_toggle],
                outputs=[comps["width"], comps["height"], letterbox_preview],
                queue=False,
                show_progress=False,
            )

            def _update_letterbox_preview(img, show_prev, w, h, fill_val):
                if not show_prev:
                    return gr.update(visible=False)
                pil_img = _coerce_pil_image(img)
                if pil_img is None:
                    return gr.update(value=None, visible=True)
                _, lb_pil = letterbox_image(pil_img, int(w or 1024), int(h or 1024), int(fill_val))
                return gr.update(value=lb_pil, visible=True)

            for trig in (preview_toggle.change, ref_image.change, fill.release):
                trig(
                    fn=_update_letterbox_preview,
                    inputs=[ref_image, preview_toggle, comps["width"], comps["height"], fill],
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
        pilot_path = find_lora_path(default_lora or PILOT_LORA_FILENAME)
        pilot_ok = bool(pilot_path and _is_valid_safetensors(pilot_path))

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
                        ref_image = gr.Image(
                            label="Reference image",
                            show_label=False,
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=240 if lobe_active else 260,
                            show_download_button=False,
                            show_share_button=False,
                            elem_id=self.elem_id("ref_image"),
                            elem_classes=["anima-ref-image"],
                        )
                        with gr.Row(elem_classes=["anima-util-row"]):
                            apply_defaults_btn = gr.Button(
                                "Set Euler / Simple / 28 / CFG 4.5",
                                size="sm",
                                elem_id=self.elem_id("apply_defaults"),
                            )
                            match_ar_btn = gr.Button(
                                "Match aspect (~1MP)",
                                size="sm",
                                elem_id=self.elem_id("match_ar"),
                            )

                    with gr.Column(scale=7, elem_classes=["anima-col-controls"]):
                        with FormRow(elem_classes=["anima-lora-row"]):
                            lora_name = gr.Dropdown(
                                choices=lora_choices,
                                value=default_lora,
                                label="Adapter LoRA",
                                allow_custom_value=True,
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
                                tooltip=f"Download or verify {PILOT_LORA_FILENAME}",
                            )

                        download_btn = gr.Button(
                            f"Download {PILOT_LORA_FILENAME}",
                            size="sm",
                            visible=not pilot_ok,
                            elem_id=self.elem_id("download_pilot"),
                        )
                        download_status = gr.Markdown(
                            value="" if pilot_ok else f"`{PILOT_LORA_FILENAME}` not found in LoRA directory.",
                            visible=not pilot_ok,
                            elem_id=self.elem_id("lora_status"),
                            elem_classes=["anima-status-text"],
                        )

                        with gr.Row(elem_classes=["anima-sliders-row"]):
                            strength = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="LoRA strength",
                                elem_id=self.elem_id("strength"),
                            )
                            ref_guidance = gr.Slider(
                                minimum=0.0,
                                maximum=20.0,
                                value=4.5,
                                step=0.1,
                                label="Reference guidance",
                                elem_id=self.elem_id("ref_guidance"),
                            )

                        with gr.Row(elem_classes=["anima-correction-row"]):
                            correction = gr.Radio(
                                choices=["apg", "rescale", "none"],
                                value="apg",
                                label="Guidance correction",
                                elem_id=self.elem_id("correction"),
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
                                    label="Preview letterbox",
                                    elem_id=self.elem_id("preview_toggle"),
                                )
                                include_ref_in_gallery = gr.Checkbox(
                                    value=False,
                                    label="Append letterbox to gallery",
                                    elem_id=self.elem_id("include_ref_in_gallery"),
                                )
                            letterbox_preview = gr.Image(
                                label="Letterbox preview",
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
            valid = bool(path and _is_valid_safetensors(path))
            return (
                gr.update(visible=not valid),
                gr.update(value="" if valid else f"`{selected_lora or PILOT_LORA_FILENAME}` not found or incomplete.", visible=not valid),
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
            val = current_val if current_val in choices else _default_lora_choice(choices)
            path = find_lora_path(val) if val else None
            valid = bool(path and _is_valid_safetensors(path))
            return (
                gr.update(choices=choices, value=val),
                gr.update(visible=not valid),
                gr.update(value="", visible=False),
            )

        refresh_btn.click(
            fn=_refresh_loras,
            inputs=[lora_name],
            outputs=[lora_name, download_btn, download_status],
            queue=False,
            show_progress=False,
        )

        def _download_pilot():
            try:
                _, msg = download_pilot_lora()
                choices = list(list_lora_files().keys())
                val = _default_lora_choice(choices)
                return (
                    gr.update(choices=choices, value=val),
                    gr.update(visible=False),
                    gr.update(value=msg, visible=True),
                )
            except Exception as exc:
                return (
                    gr.skip(),
                    gr.update(visible=True),
                    gr.update(value=f"Download failed: {exc}", visible=True),
                )

        for btn in (download_btn, download_tool_btn):
            btn.click(
                fn=_download_pilot,
                inputs=[],
                outputs=[lora_name, download_btn, download_status],
                show_progress=True,
            )

        self._pending_tab_wirings[tab] = {
            "apply_defaults_btn": apply_defaults_btn,
            "match_ar_btn": match_ar_btn,
            "ref_image": ref_image,
            "fill": fill,
            "preview_toggle": preview_toggle,
            "letterbox_preview": letterbox_preview,
        }
        self._try_bind_tab_actions(tab)

        self.infotext_fields = [
            (lora_name, "Anima Ref Adapter LoRA"),
            (strength, "Anima Ref Strength"),
            (ref_guidance, "Anima Ref Guidance"),
            (correction, "Anima Ref Correction"),
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
            ref_image,
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
        ]

    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        enable: bool,
        ref_image,
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
        *args,
        **kwargs,
    ):
        self._last_letterboxed_ref = None
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

        pil_ref = _coerce_pil_image(ref_image)
        if pil_ref is not None:
            lora_path = find_lora_path(lora_name)
            if lora_path is None:
                logger.error(
                    f"[Anima Ref Adapter] Adapter LoRA '{lora_name}' not found. "
                    f"Download '{PILOT_LORA_FILENAME}' into your LoRA directory or click the Download button."
                )
                gr.Warning(f"[Anima Ref Adapter] LoRA '{lora_name}' not found.")
                return
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
                ref_latents, letterboxed_pil = encode_reference_latents(
                    image=pil_ref,
                    vae=vae,
                    width=target_w,
                    height=target_h,
                    fill=int(fill),
                )
                if include_ref_in_gallery:
                    self._last_letterboxed_ref = letterboxed_pil

                try:
                    config = AnimaRefAdapterConfig(
                        lora_path=lora_path,
                        strength=float(strength),
                        ref_offset_h=float(ref_offset_h),
                        ref_guidance=float(ref_guidance),
                        correction=str(correction),
                        rescale_phi=float(rescale_phi),
                        fill=int(fill),
                    )
                    unet = patch_unet_for_anima_ref(unet, ref_latents, config)
                except Exception as exc:
                    logger.error(f"[Anima Ref Adapter] {exc}")
                    gr.Warning(f"[Anima Ref Adapter] {exc}")
                    return

                p.extra_generation_params["Anima Ref Adapter LoRA"] = os.path.basename(lora_path)
                p.extra_generation_params["Anima Ref Strength"] = float(strength)
                p.extra_generation_params["Anima Ref Guidance"] = float(ref_guidance)
                p.extra_generation_params["Anima Ref Correction"] = str(correction)
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

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        if self._last_letterboxed_ref is not None:
            processed.extra_images.append(self._last_letterboxed_ref)
            self._last_letterboxed_ref = None
