"""Core inference logic for Anima Reference Adapter (and Anima Safe PAG) in Forge-Neo.

Ported from https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter (Apache-2.0)
for Forge-Neo's Anima DiffusionEngine, UnetPatcher, and sampling pipeline.

How it works:
  * The reference image is letterboxed to the target resolution (aspect ratio preserved,
    padded with grey level `fill`, default 255 white) and encoded by the Wan 2.1 VAE.
  * During diffusion forward passes that carry a reference latent (`positive` and `negative`),
    the reference latent is concatenated as a 2nd temporal frame (`t_index=1`) at clean
    `timestep=0`, with its RoPE height coordinates shifted down by `ref_offset_h * H_p`
    so the model does not copy pose pixel-aligned.
  * The rank-32 adapter LoRA is applied via forward hooks ONLY while the reference frame
    is present (`_active["key"] == adapter.key`).
  * Separate text guidance (`cfg`) and reference guidance (`ref_guidance`) with APG or
    CFG-rescale correction combine `base` (unconditional without reference/LoRA),
    `neg` (unconditional with reference+LoRA), and `pos` (conditional with reference+LoRA).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import struct
import urllib.request
from types import MethodType
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend import memory_management
from backend.logging import setup_logger
from backend.nn.anima import BLOCK_MAPPINGS
from backend.sampling.condition import Condition, repeat_to_batch_size
from backend.sampling.sampling_function import calc_cond_uncond_batch
from backend.utils import load_torch_file, pad_to_patch_size
from modules import shared

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher
    from backend.patcher.vae import VAE

logger = logging.getLogger("anima_ref_adapter")
setup_logger(logger)

REF_KEY = "anima_ref_latent"
REF_HALF_KEY = "anima_ref_latent_half"
HOOKS_ATTR = "_anima_ref_adapter_hooks"

PILOT_LORA_FILENAME = "anima_ref_adapter_pilot_r32.safetensors"
PILOT_LORA_URL = (
    "https://huggingface.co/ElectronGirl/anima-ref-adapter/resolve/main/"
    "anima_ref_adapter_pilot_r32.safetensors"
)

_adapters: dict[tuple, "_Adapter"] = {}
_active = {"key": None}
_warned_sizes: set[tuple[int, int]] = set()
_ref_latent_cache: dict[tuple, dict[str, torch.Tensor]] = {}


@dataclasses.dataclass
class AnimaRefAdapterConfig:
    lora_path: str
    strength: float = 1.0
    ref_offset_h: float = 1.0
    ref_guidance: float = 4.5
    correction: str = "apg"  # "apg", "rescale", "none"
    rescale_phi: float = 0.5
    fill: int = 255


@dataclasses.dataclass
class AnimaSafePAGConfig:
    enabled: bool = False
    scale: float = 1.3
    block_indices: str = "18"
    perturbation_strength: float = 0.75
    head_indices: str = ""
    start_percent: float = 0.3
    end_percent: float = 0.8
    rescale: float = 0.2
    rescale_mode: str = "full"  # "full" or "partial"


def _extension_models_dir() -> str:
    ext_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(ext_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _lora_search_dirs() -> list[str]:
    dirs: list[str] = []
    cmd_opts = getattr(shared, "cmd_opts", None)
    if cmd_opts is not None:
        lora_dir = getattr(cmd_opts, "lora_dir", None)
        if lora_dir and os.path.isdir(lora_dir):
            dirs.append(lora_dir)
        for d in getattr(cmd_opts, "lora_dirs", []) or []:
            if d and os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    ext_models = _extension_models_dir()
    if ext_models not in dirs:
        dirs.append(ext_models)
    return dirs


def list_lora_files() -> dict[str, str]:
    """Return mapping of display name -> absolute path for available LoRA files.

    Prioritizes files containing 'anima' or 'ref' at the top of the list.
    """
    found: dict[str, str] = {}
    allowed_exts = (".safetensors", ".pt", ".ckpt")

    for base_dir in _lora_search_dirs():
        for root, _, files in os.walk(base_dir, followlinks=True):
            for fname in files:
                if not fname.lower().endswith(allowed_exts):
                    continue
                full_path = os.path.abspath(os.path.join(root, fname))
                rel_path = os.path.relpath(full_path, base_dir).replace("\\", "/")
                if rel_path not in found:
                    found[rel_path] = full_path

    def _sort_key(item: tuple[str, str]):
        name_lower = os.path.basename(item[0]).lower()
        is_pilot = name_lower == PILOT_LORA_FILENAME.lower()
        is_anima_ref = "anima" in name_lower and "ref" in name_lower
        is_anima = "anima" in name_lower or "ref" in name_lower
        return (not is_pilot, not is_anima_ref, not is_anima, name_lower)

    return dict(sorted(found.items(), key=_sort_key))


def find_lora_path(lora_name: str) -> Optional[str]:
    """Resolve a LoRA dropdown selection, filename, or full path to an existing file."""
    if not lora_name:
        return None
    if os.path.isfile(lora_name):
        return os.path.abspath(lora_name)

    available = list_lora_files()
    if lora_name in available:
        return available[lora_name]

    target_base = os.path.basename(lora_name).lower()
    for rel, full in available.items():
        if os.path.basename(rel).lower() == target_base or os.path.splitext(os.path.basename(rel))[0].lower() == target_base:
            return full
    return None


def _is_valid_safetensors(path: str) -> bool:
    """Verify that a .safetensors file exists and its size matches the header offsets."""
    if not path or not os.path.isfile(path):
        return False
    if not path.lower().endswith((".safetensors", ".sft")):
        return os.path.getsize(path) > 0
    try:
        file_size = os.path.getsize(path)
        if file_size < 16:
            return False
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len <= 0 or 8 + header_len > file_size:
                return False
            header = json.loads(f.read(header_len))
        max_end = max(
            (v["data_offsets"][1] for k, v in header.items() if k != "__metadata__" and "data_offsets" in v),
            default=0,
        )
        return (8 + header_len + max_end) == file_size
    except Exception:
        return False


def download_pilot_lora() -> tuple[str, str]:
    """Download `anima_ref_adapter_pilot_r32.safetensors` from Hugging Face into the LoRA directory."""
    existing = find_lora_path(PILOT_LORA_FILENAME)
    if existing and _is_valid_safetensors(existing):
        return existing, f"Already installed and verified: {existing}"

    search_dirs = _lora_search_dirs()
    target_dir = os.path.dirname(existing) if existing else (search_dirs[0] if search_dirs else _extension_models_dir())
    os.makedirs(target_dir, exist_ok=True)
    dest_path = existing or os.path.join(target_dir, PILOT_LORA_FILENAME)
    tmp_path = dest_path + ".download"

    logger.info(f"[Anima Ref Adapter] Downloading {PILOT_LORA_URL} -> {dest_path} ...")
    try:
        req = urllib.request.Request(PILOT_LORA_URL, headers={"User-Agent": "Forge-Neo-Anima-Ref-Adapter/1.0"})
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (16 * chunk_size) == 0:
                    logger.info(f"[Anima Ref Adapter] Downloaded {downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB")
        if total > 0 and downloaded != total:
            raise RuntimeError(f"Incomplete download ({downloaded} / {total} bytes)")
        if not _is_valid_safetensors(tmp_path):
            raise RuntimeError("Downloaded safetensors file failed integrity check")
        os.replace(tmp_path, dest_path)
        logger.info(f"[Anima Ref Adapter] Successfully downloaded to {dest_path}")
        return dest_path, f"Downloaded and verified: {dest_path}"
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.error(f"[Anima Ref Adapter] Download failed: {exc}")
        raise RuntimeError(f"Failed to download {PILOT_LORA_FILENAME}: {exc}") from exc


def _adapter_key(path: str, strength: float, num_blocks: int) -> tuple:
    return os.path.abspath(path), os.path.getmtime(path), float(strength), int(num_blocks)


def _remap_anima_blocks(weights: dict[str, list], model_blocks: int) -> dict[str, list]:
    """Remap 28-block (or 40-block) Anima adapter LoRA weights if the loaded Anima model has 40 or 52 blocks."""
    parsed: dict[str, tuple[int, str]] = {}
    prefix = "blocks."
    for k in weights:
        if not k.startswith(prefix):
            continue
        num, sep, tail = k[len(prefix) :].partition(".")
        if sep and num.isdigit():
            parsed[k] = (int(num), tail)

    if not parsed:
        return weights

    max_block = max(n for n, _ in parsed.values()) + 1
    lora_blocks = next((size for size in (28, 40, 52) if max_block <= size), max_block)
    if lora_blocks == model_blocks:
        return weights

    mapping = BLOCK_MAPPINGS.get((lora_blocks, model_blocks))
    if mapping is None:
        logger.warning(f"[Anima Ref Adapter] No block mapping from {lora_blocks} to {model_blocks}; using unmapped keys")
        return weights

    logger.info(f"[Anima Ref Adapter] Remapping adapter LoRA blocks ({lora_blocks} -> {model_blocks})")
    reverse: dict[int, list[int]] = {}
    for target, source in enumerate(mapping):
        reverse.setdefault(source, []).append(target)

    out = {k: v for k, v in weights.items() if k not in parsed}
    for k, (source, tail) in parsed.items():
        down, up, scale = weights[k]
        for target in reverse.get(source, []):
            out[f"blocks.{target}.{tail}"] = [down.clone(), up.clone(), scale]
    return out


class _Adapter:
    """LoRA weights applied through forward hooks, active only while `_active['key'] == self.key`."""

    def __init__(self, path: str, strength: float, num_blocks: int = 28):
        self.key = _adapter_key(path, strength, num_blocks)
        tensors = load_torch_file(path, safe_load=True)
        raw_weights: dict[str, list] = {}

        for key, down in tensors.items():
            if key.endswith(".lora_down.weight"):
                up_key = key[:-len(".lora_down.weight")] + ".lora_up.weight"
                alpha_key = key[:-len(".lora_down.weight")] + ".alpha"
                mod_name = key[:-len(".lora_down.weight")]
            elif key.endswith(".lora_A.weight"):
                up_key = key[:-len(".lora_A.weight")] + ".lora_B.weight"
                alpha_key = key[:-len(".lora_A.weight")] + ".alpha"
                mod_name = key[:-len(".lora_A.weight")]
            else:
                continue

            if mod_name.startswith("diffusion_model."):
                mod_name = mod_name[len("diffusion_model.") :]
            elif mod_name.startswith("transformer."):
                mod_name = mod_name[len("transformer.") :]

            if up_key not in tensors:
                continue
            up = tensors[up_key]
            alpha = float(tensors.get(alpha_key, torch.tensor(down.shape[0])))
            raw_weights[mod_name] = [down, up, float(strength) * alpha / down.shape[0]]

        if not raw_weights:
            raise ValueError(f"{path} has no diffusion_model lora_down/lora_up weights")

        self.weights = _remap_anima_blocks(raw_weights, num_blocks)

    @classmethod
    def get(cls, path: str, strength: float, num_blocks: int = 28) -> "_Adapter":
        key = _adapter_key(path, strength, num_blocks)
        if key not in _adapters:
            _adapters[key] = cls(path, strength, num_blocks)
        return _adapters[key]

    def install(self, diffusion_model: torch.nn.Module) -> None:
        for name, entry in self.weights.items():
            try:
                module = diffusion_model.get_submodule(name)
            except AttributeError:
                logger.warning(f"[Anima Ref Adapter] Submodule '{name}' not found on diffusion_model")
                continue

            installed = getattr(module, HOOKS_ATTR, None)
            if installed is None:
                installed = {}
                setattr(module, HOOKS_ATTR, installed)
            if self.key in installed:
                continue

            def hook(module, inputs, output, entry=entry, key=self.key):
                if _active["key"] != key:
                    return output
                x = inputs[0]
                if entry[0].device != x.device or entry[0].dtype != x.dtype:
                    entry[0] = entry[0].to(device=x.device, dtype=x.dtype)
                    entry[1] = entry[1].to(device=x.device, dtype=x.dtype)
                return output + F.linear(F.linear(x, entry[0]), entry[1]) * entry[2]

            installed[self.key] = module.register_forward_hook(hook)


def _frame_positions(t_index: int, height: int, width: int, h0: int, w0: int, device: torch.device):
    hh, ww = torch.meshgrid(
        torch.arange(height, device=device) + h0,
        torch.arange(width, device=device) + w0,
        indexing="ij",
    )
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def _rope_from_positions(pe, t: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """VideoRopePosition3DEmb embeddings for explicit per-token (t, h, w) coordinates."""
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _pick_ref(refs: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
    """Select an exact-size reference latent if available, else resize the closest one."""
    refs = [r.unsqueeze(2) if r.ndim == 4 else r for r in refs]
    for ref in refs:
        if tuple(ref.shape[-2:]) == size:
            return ref
    ref = min(refs, key=lambda r: abs(r.shape[-2] - size[0]) + abs(r.shape[-1] - size[1]))
    if size not in _warned_sizes:
        _warned_sizes.add(size)
        logger.warning(
            f"[Anima Ref Adapter] No reference encoded at latent size {size}; "
            f"resizing the {tuple(ref.shape[-2:])} one (lower fidelity)"
        )
    shrink = ref.shape[-2] > size[0]
    return F.interpolate(
        ref[:, :, 0].float(),
        size=size,
        mode="area" if shrink else "bilinear",
    ).unsqueeze(2)


def _ref_forward(
    dm,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    refs: list[torch.Tensor],
    offset_ratio: float,
    transformer_options: dict,
) -> torch.Tensor:
    """Anima DiT forward pass with the reference latent as a 2nd frame (timestep 0, shifted RoPE)."""
    orig_shape = list(x.shape)
    was_4d = x.ndim == 4
    if was_4d:
        x = x.unsqueeze(2)

    patch = (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial)
    x = pad_to_patch_size(x, patch)
    ref = _pick_ref(refs, tuple(x.shape[-2:])).to(device=x.device, dtype=x.dtype)
    if ref.shape[0] != x.shape[0]:
        ref = repeat_to_batch_size(ref, x.shape[0])

    frames = torch.cat([x[:, :, :1], ref[:, :, :1]], dim=2)
    x_emb, _, extra = dm.prepare_embedded_sequence(frames, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape

    positions = [
        _frame_positions(0, hp, wp, 0, 0, x.device),
        _frame_positions(1, hp, wp, round(offset_ratio * hp), 0, x.device),
    ]
    rope = _rope_from_positions(dm.pos_embedder, *(torch.cat(axis) for axis in zip(*positions)))

    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    timesteps = torch.cat([timesteps, torch.zeros_like(timesteps)], dim=1)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](timesteps).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)

    transformer_options = (transformer_options or {}).copy()
    block_kwargs = {
        "rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln,
        "extra_per_block_pos_emb": extra,
        "transformer_options": transformer_options,
    }

    if x_emb.dtype is torch.float16:
        x_emb = x_emb.float()

    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        x_emb = block(x_emb, t_emb, context, **block_kwargs)

    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    out = dm.unpatchify(out)[:, :, :1]
    if was_4d:
        return out[:, :, 0, : orig_shape[-2], : orig_shape[-1]]
    return out[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]


def _make_forward_wrapper(dm, orig_forward, adapter: _Adapter, offset_ratio: float):
    orig_forward = getattr(orig_forward, "_anima_ref_inner", orig_forward)

    def wrapped_forward(x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, padding_mask=None, **kwargs):
        refs = [r for r in (kwargs.pop(REF_KEY, None), kwargs.pop(REF_HALF_KEY, None)) if r is not None]
        if not refs:
            return orig_forward(x, timesteps, context, padding_mask=padding_mask, **kwargs)
        adapter.install(dm)
        previous = _active["key"]
        _active["key"] = adapter.key
        try:
            return _ref_forward(
                dm,
                x,
                timesteps,
                context,
                refs,
                offset_ratio,
                kwargs.get("transformer_options", {}),
            )
        finally:
            _active["key"] = previous

    wrapped_forward._anima_ref_inner = orig_forward
    return wrapped_forward


def letterbox_image(src: Image.Image, width: int, height: int, fill: int = 255) -> tuple[torch.Tensor, Image.Image]:
    """Fit `src` into `(width, height)` keeping aspect ratio with `(fill, fill, fill)` padding.

    Returns `(pixels_tensor, pil_image)` where `pixels_tensor` has shape `(1, H, W, 3)` in `[0, 1]`.
    """
    fill = max(0, min(255, int(fill)))
    if src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info):
        rgba = src.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (fill, fill, fill, 255))
        src = Image.alpha_composite(bg, rgba).convert("RGB")
    elif src.mode != "RGB":
        src = src.convert("RGB")

    scale = min(width / max(1, src.width), height / max(1, src.height))
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    canvas = Image.new("RGB", (width, height), (fill, fill, fill))
    canvas.paste(src.resize(size, Image.Resampling.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    pixels = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)
    return pixels, canvas


def _hash_pil_image(img: Image.Image) -> int:
    thumb = img.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
    return hash(thumb.tobytes())


@torch.inference_mode()
def encode_reference_latents(
    image: Image.Image,
    vae: "VAE",
    width: int,
    height: int,
    fill: int = 255,
) -> tuple[dict[str, torch.Tensor], Image.Image]:
    """Letterbox and VAE-encode the reference image (plus half-resolution encode)."""
    img_hash = _hash_pil_image(image)
    vae_id = id(vae.first_stage_model)
    cache_key = (img_hash, vae_id, int(width), int(height), int(fill))

    pixels, letterboxed_pil = letterbox_image(image, width, height, fill)
    if cache_key in _ref_latent_cache:
        return _ref_latent_cache[cache_key], letterboxed_pil

    ref = vae.encode(pixels)
    ref = vae.first_stage_model.process_in(ref).detach().cpu()

    half_w = max(16, (width // 16) * 8)
    half_h = max(16, (height // 16) * 8)
    pixels_half, _ = letterbox_image(image, half_w, half_h, fill)
    half = vae.encode(pixels_half)
    half = vae.first_stage_model.process_in(half).detach().cpu()

    out = {"samples": ref, "samples_half": half}
    if len(_ref_latent_cache) >= 8:
        _ref_latent_cache.pop(next(iter(_ref_latent_cache)))
    _ref_latent_cache[cache_key] = out
    return out, letterboxed_pil


def _flat_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def patch_unet_for_anima_ref(
    unet: "UnetPatcher",
    ref_latents: dict[str, torch.Tensor],
    config: AnimaRefAdapterConfig,
) -> "UnetPatcher":
    """Clone `unet` and attach the Anima Ref Adapter LoRA, forward wrapper, conditioning modifier, and guider."""
    dm = unet.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        raise RuntimeError("Anima Ref Adapter requires an Anima diffusion model with `diffusion_model.blocks`.")

    adapter = _Adapter.get(config.lora_path, config.strength, num_blocks=len(blocks))
    adapter.install(dm)

    m = unet.clone()
    m.add_extra_preserved_memory_during_sampling(128 * 1024 * 1024)

    orig_forward = m.get_model_object("diffusion_model.forward")
    m.add_object_patch(
        "diffusion_model.forward",
        _make_forward_wrapper(dm, orig_forward, adapter, config.ref_offset_h),
    )

    ref_main = ref_latents["samples"]
    ref_half = ref_latents.get("samples_half")

    def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        model_options = model_options.copy()

        uncond_base = []
        if uncond is not None:
            for item in uncond:
                base_item = item.copy()
                base_mc = item["model_conds"].copy()
                base_mc.pop(REF_KEY, None)
                base_mc.pop(REF_HALF_KEY, None)
                base_item["model_conds"] = base_mc
                uncond_base.append(base_item)
        model_options["_anima_ref_uncond_base"] = uncond_base

        def _attach_refs(cond_list):
            if cond_list is None:
                return None
            out_list = []
            for item in cond_list:
                new_item = item.copy()
                mc = item["model_conds"].copy()
                mc[REF_KEY] = Condition(ref_main)
                if ref_half is not None:
                    mc[REF_HALF_KEY] = Condition(ref_half)
                new_item["model_conds"] = mc
                out_list.append(new_item)
            return out_list

        return model, x, timestep, _attach_refs(uncond), _attach_refs(cond), cond_scale, model_options, seed

    m.add_conditioning_modifier(conditioning_modifier)

    ref_guidance = float(config.ref_guidance)
    correction = str(config.correction).lower()
    rescale_phi = float(config.rescale_phi)

    def anima_ref_cfg_function(args: dict) -> torch.Tensor:
        pos: torch.Tensor = args["cond_denoised"]
        neg: torch.Tensor = args["uncond_denoised"]
        x: torch.Tensor = args["input"]
        timestep: torch.Tensor = args["timestep"]
        cfg: float = float(args["cond_scale"])
        model = args["model"]
        model_options: dict = args["model_options"]

        if ref_guidance != 1.0:
            uncond_base = model_options.get("_anima_ref_uncond_base")
            if uncond_base:
                _, base = calc_cond_uncond_batch(model, [], uncond_base, x, timestep, model_options)
            else:
                base = neg
        else:
            base = neg

        guided = base + ref_guidance * (neg - base) + cfg * (pos - neg)
        if correction == "apg":
            update = (guided - pos).double()
            unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
            parallel = (update * unit).sum(dim=_flat_dims(pos), keepdim=True) * unit
            guided = pos + (update - parallel).to(pos.dtype)
        elif correction == "rescale":
            std_pos = pos.std(dim=_flat_dims(pos), keepdim=True)
            std_guided = guided.std(dim=_flat_dims(guided), keepdim=True).clamp_min(1e-6)
            ratio = std_pos / std_guided
            guided = rescale_phi * guided * ratio + (1.0 - rescale_phi) * guided

        # `sampling_function_inner` computes `cfg_result = x - sampler_cfg_function(args)`
        return x - guided

    m.set_model_sampler_cfg_function(anima_ref_cfg_function, disable_cfg1_optimization=True)
    return m


# region Anima Safe PAG


def _sigma_to_float(sigma) -> float:
    if torch.is_tensor(sigma):
        return float(sigma.flatten()[0].item())
    return float(sigma)


def _sigma_active(sigma, sigma_start: float, sigma_end: float) -> bool:
    s_start = _sigma_to_float(sigma_start)
    s_end = _sigma_to_float(sigma_end)
    if s_start < s_end:
        s_start, s_end = s_end, s_start
    val = _sigma_to_float(sigma)
    return s_end <= val <= s_start


def _parse_indices(text: str, max_index: int) -> list[int]:
    values: set[int] = set()
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = [p.strip() for p in part.split("-", 1)]
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise ValueError(f"Invalid block range '{part}'. Use values like 18 or 18-20.")
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    indices = sorted(i for i in values if 0 <= i <= max_index)
    if not indices:
        raise ValueError(f"No valid indices found in '{text}' (valid range: 0..{max_index}).")
    return indices


def _parse_optional_indices(text: Optional[str], max_index: int) -> Optional[list[int]]:
    if text is None or not str(text).strip():
        return None
    return _parse_indices(text, max_index)


def _sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


def _lerp_heads(base: torch.Tensor, target: torch.Tensor, strength: float, heads: Optional[list[int]]) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return base
    if heads is None:
        return base.lerp(target, strength)
    out = base.clone()
    out[:, :, heads, :] = base[:, :, heads, :].lerp(target[:, :, heads, :], strength)
    return out


def _project_attention(attn_module: torch.nn.Module, attn: torch.Tensor) -> torch.Tensor:
    value = attn.reshape(*attn.shape[:-2], attn.shape[-2] * attn.shape[-1])
    if hasattr(attn_module, "output_proj"):
        value = attn_module.output_proj(value)
        if hasattr(attn_module, "output_dropout"):
            value = attn_module.output_dropout(value)
        return value
    if hasattr(attn_module, "o_proj"):
        return attn_module.o_proj(value)
    raise RuntimeError("Unsupported attention module: no output projection found.")


def _soft_pag_attention(attn_module: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, strength: float, heads: Optional[list[int]]) -> torch.Tensor:
    normal = _sdpa_attention(q, k, v)
    weak = _lerp_heads(normal, v, strength, heads)
    return _project_attention(attn_module, weak)


def _patch_anima_attention(blocks, indices: list[int], perturbation_strength: float, head_indices: str):
    patched = []
    for idx in indices:
        block = blocks[idx]
        attn = getattr(block, "self_attn", None)
        if attn is None or not hasattr(attn, "compute_attention"):
            raise RuntimeError(f"Block {idx} does not expose self_attn.compute_attention.")
        original = attn.compute_attention
        heads = _parse_optional_indices(head_indices, attn.n_heads - 1)

        def compute_attention(self, q, k, v, transformer_options=None, _orig=original, _str=perturbation_strength, _heads=heads):
            if _str <= 0:
                return _orig(q, k, v, transformer_options=transformer_options or {})
            return _soft_pag_attention(self, q, k, v, _str, _heads)

        attn.compute_attention = MethodType(compute_attention, attn)
        patched.append((attn, original))
    return patched


def _restore_attention(patched) -> None:
    for attn, original in patched:
        attn.compute_attention = original


def _rescale_pag_guidance(guidance: torch.Tensor, cond_pred: torch.Tensor, cfg_result: torch.Tensor, rescale: float, mode: str) -> torch.Tensor:
    rescale = float(rescale)
    if rescale <= 0:
        return guidance
    guidance_result = cfg_result + guidance if mode == "full" else cond_pred + guidance
    reduce_dims = tuple(range(1, guidance_result.ndim))
    std_cond = torch.std(cond_pred, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    std_guidance = torch.std(guidance_result, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    factor = std_cond / std_guidance
    factor = rescale * factor + (1.0 - rescale)
    return guidance * factor


def patch_unet_for_safe_pag(unet: "UnetPatcher", pag_config: AnimaSafePAGConfig) -> "UnetPatcher":
    """Attach Anima Safe PAG post-CFG guidance to `unet`."""
    if not pag_config.enabled or pag_config.scale <= 0:
        return unet

    dm = unet.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        raise RuntimeError("Anima Safe PAG requires an Anima diffusion model with `diffusion_model.blocks`.")

    indices = _parse_indices(pag_config.block_indices, len(blocks) - 1)
    start_p = max(0.0, min(1.0, float(pag_config.start_percent)))
    end_p = max(0.0, min(1.0, float(pag_config.end_percent)))
    if start_p > end_p:
        start_p, end_p = end_p, start_p

    predictor = unet.model.predictor
    sigma_start = predictor.percent_to_sigma(start_p)
    sigma_end = predictor.percent_to_sigma(end_p)

    m = unet.clone()

    def pag_post_cfg_function(args: dict) -> torch.Tensor:
        cfg_result: torch.Tensor = args["denoised"]
        sigma = args["sigma"]
        if not _sigma_active(sigma, sigma_start, sigma_end):
            return cfg_result

        cond = args["cond"]
        if not cond:
            return cfg_result

        model = args["model"]
        x = args["input"]
        model_options = args["model_options"]
        cond_pred: torch.Tensor = args["cond_denoised"]

        patched = _patch_anima_attention(blocks, indices, pag_config.perturbation_strength, pag_config.head_indices)
        try:
            pag_pred, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)
        finally:
            _restore_attention(patched)

        if pag_pred.shape != cond_pred.shape:
            return cfg_result

        guidance = (cond_pred - pag_pred) * float(pag_config.scale)
        guidance = _rescale_pag_guidance(
            guidance,
            cond_pred,
            cfg_result,
            float(pag_config.rescale),
            str(pag_config.rescale_mode).lower(),
        )
        return cfg_result + guidance

    m.set_model_sampler_post_cfg_function(pag_post_cfg_function)
    return m
