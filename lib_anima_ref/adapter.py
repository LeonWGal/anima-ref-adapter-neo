"""Anima Reference Adapter (v2 Asymmetric + v1 Pilot backward compatible) for Forge Neo.

Supports:
- v2 Adapter (asymmetric cross-attention, cached per-block KV, arbitrary aspect ratio/resolution via megapixels)
- Pilot Adapter (joint temporal attention, canvas letterboxing)
- Multi-character side-by-side reference composition
- Guidance cutoff window (ref_end)
- APG, Rescale, and None guidance corrections
- Safe PAG seamless compatibility
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import struct
import urllib.request
from types import MethodType
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from backend import memory_management
from backend.logging import setup_logger
from backend.sampling.condition import Condition, repeat_to_batch_size
from backend.sampling.sampling_function import calc_cond_uncond_batch
from backend.utils import load_torch_file, pad_to_patch_size

logger = logging.getLogger("AnimaRefAdapter")
setup_logger(logger)

REF_KEY = "anima_ref_latent"
REF_HALF_KEY = "anima_ref_latent_half"
HOOKS_ATTR = "_anima_ref_adapter_hooks"

HF_REPO = "ElectronGirl/anima-ref-adapter"
V2_LORA_FILENAME = "anima_ref_adapter_v2_r64.safetensors"
PILOT_LORA_FILENAME = "anima_ref_adapter_pilot_r32.safetensors"
HF_FILES = (V2_LORA_FILENAME, PILOT_LORA_FILENAME)
NO_REF_KEY = "anima_ref_off"

_adapters: dict[tuple, "_Adapter"] = {}
_active = {"key": None}
_ref_latent_cache: dict[tuple, dict[str, torch.Tensor]] = {}
_warned_sizes: set[tuple[int, int]] = set()


@dataclasses.dataclass
class AnimaRefAdapterConfig:
    lora_path: str
    strength: float = 1.0
    ref_offset_h: float = 1.0
    ref_guidance: float = 4.5
    correction: str = "apg"  # "apg", "rescale", "none"
    rescale_phi: float = 0.5
    fill: int = 255
    combine_mode: str = "stitch"  # "stitch", "attention", "blend"
    ref_end: float = 1.0
    ref_megapixels: float = 1.0


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
    rescale_mode: str = "full"


def _extension_models_dir() -> str:
    """Directory inside this extension where default models are stored."""
    ext_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(ext_root, "models")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def _lora_search_dirs() -> list[str]:
    """Return all directories where Forge searches for LoRA models."""
    dirs: list[str] = [_extension_models_dir()]
    try:
        from modules import shared

        cmd_opts = getattr(shared, "cmd_opts", None)
    except Exception:
        cmd_opts = None

    if cmd_opts:
        lora_dir = getattr(cmd_opts, "lora_dir", None)
        if lora_dir and os.path.isdir(lora_dir) and lora_dir not in dirs:
            dirs.append(lora_dir)
        for d in getattr(cmd_opts, "lora_dirs", []) or []:
            if d and os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    return dirs


def is_anima_adapter(path: str) -> bool:
    """Fast check whether a safetensors file contains Anima DiT blocks.* keys without loading tensors into RAM."""
    if not path or not os.path.isfile(path) or not _is_valid_safetensors(path):
        return False
    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len <= 0 or 8 + header_len > os.path.getsize(path):
                return False
            header = json.loads(f.read(header_len))
        return any("blocks." in k for k in header.keys() if k != "__metadata__")
    except Exception:
        return False


def list_lora_files(all_models: bool = False) -> dict[str, str]:
    """Return mapping of display name -> absolute path for available LoRA files."""
    found: dict[str, str] = {}
    allowed_exts = (".safetensors", ".pt", ".ckpt")
    ext_models = _extension_models_dir()

    # 1. Always include valid Anima files from the extension's own models folder
    if os.path.isdir(ext_models):
        for root, _, files in os.walk(ext_models):
            for fname in files:
                if fname.lower().endswith(allowed_exts):
                    full = os.path.abspath(os.path.join(root, fname))
                    if all_models or is_anima_adapter(full):
                        rel = os.path.relpath(full, ext_models).replace("\\", "/")
                        found[rel] = full

    # 2. Add other search dirs
    for base_dir in _lora_search_dirs():
        if base_dir == ext_models or not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir, followlinks=True):
            for fname in files:
                if not fname.lower().endswith(allowed_exts):
                    continue
                name_lower = fname.lower()
                if not all_models and not ("anima" in name_lower or "ref" in name_lower):
                    continue
                full_path = os.path.abspath(os.path.join(root, fname))
                if not all_models and not is_anima_adapter(full_path):
                    continue
                rel_path = os.path.relpath(full_path, base_dir).replace("\\", "/")
                if rel_path not in found:
                    found[rel_path] = full_path

    # Ensure v2 file is always visible in dropdown if present in models
    ext_v2 = os.path.join(ext_models, V2_LORA_FILENAME)
    if os.path.isfile(ext_v2) and V2_LORA_FILENAME not in found:
        found[V2_LORA_FILENAME] = ext_v2

    def _sort_key(item: tuple[str, str]):
        name_lower = os.path.basename(item[0]).lower()
        is_ext_v2 = (
            name_lower == V2_LORA_FILENAME.lower()
            and os.path.dirname(item[1]).replace("\\", "/").rstrip("/").endswith("sd-forge-anima-ref-adapter/models")
        )
        is_v2 = name_lower == V2_LORA_FILENAME.lower()
        is_ext_pilot = (
            name_lower == PILOT_LORA_FILENAME.lower()
            and os.path.dirname(item[1]).replace("\\", "/").rstrip("/").endswith("sd-forge-anima-ref-adapter/models")
        )
        is_pilot = name_lower == PILOT_LORA_FILENAME.lower()
        return (not is_ext_v2, not is_v2, not is_ext_pilot, not is_pilot, name_lower)

    return dict(sorted(found.items(), key=_sort_key))


def find_lora_path(lora_name: str) -> Optional[str]:
    """Resolve a LoRA dropdown selection, filename, or full path to an existing file."""
    ext_v2 = os.path.join(_extension_models_dir(), V2_LORA_FILENAME)
    ext_pilot = os.path.join(_extension_models_dir(), PILOT_LORA_FILENAME)

    if not lora_name:
        for p in (ext_v2, ext_pilot):
            if os.path.isfile(p) and _is_valid_safetensors(p):
                return p
        return None

    for p in (ext_v2, ext_pilot):
        if (lora_name == os.path.basename(p) or lora_name == p) and os.path.isfile(p) and _is_valid_safetensors(p):
            return p

    if os.path.isabs(lora_name) and os.path.isfile(lora_name) and _is_valid_safetensors(lora_name):
        return lora_name

    target_base = os.path.basename(lora_name).lower()
    target_clean = os.path.splitext(target_base)[0]
    for rel, full in list_lora_files(all_models=True).items():
        if rel == lora_name or full == lora_name:
            return full
        curr_base = os.path.basename(rel).lower()
        if curr_base == target_base or os.path.splitext(curr_base)[0] == target_clean:
            return full

    for p in (ext_v2, ext_pilot):
        if os.path.isfile(p) and _is_valid_safetensors(p):
            return p
    return None


def _is_valid_safetensors(path: str) -> bool:
    """Verify that `path` is an existing, non-empty, valid safetensors file."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return False
            header_len = struct.unpack("<Q", header_bytes)[0]
            if header_len <= 0 or (8 + header_len) > os.path.getsize(path):
                return False
        return True
    except Exception:
        return False


def download_adapter_lora(filename: str = V2_LORA_FILENAME) -> tuple[str, str]:
    """Download adapter safetensors (v2 or pilot) from Hugging Face directly into the extension models directory."""
    if filename not in HF_FILES:
        filename = V2_LORA_FILENAME
    target_dir = _extension_models_dir()
    dest_path = os.path.join(target_dir, filename)
    if os.path.isfile(dest_path) and _is_valid_safetensors(dest_path):
        return dest_path, f"Verified in extension models folder: {dest_path}"

    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
    tmp_path = dest_path + ".download"
    logger.info(f"[Anima Ref Adapter] Downloading {url} -> {dest_path} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Forge-Neo-Anima-Ref-Adapter/2.0"})
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    percent = (downloaded / total) * 100.0
                    logger.info(f"[Anima Ref Adapter] Progress: {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({percent:.1f}%)")

        if os.path.isfile(dest_path):
            os.remove(dest_path)
        os.rename(tmp_path, dest_path)
        logger.info(f"[Anima Ref Adapter] Successfully downloaded {filename} ({os.path.getsize(dest_path)} bytes).")
        return dest_path, f"Successfully downloaded: {dest_path}"
    except Exception as exc:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.error(f"[Anima Ref Adapter] Download failed: {exc}")
        raise RuntimeError(f"Failed to download {filename}: {exc}") from exc


def download_pilot_lora() -> tuple[str, str]:
    return download_adapter_lora(PILOT_LORA_FILENAME)


def _adapter_key(path: str, strength: float, num_blocks: int) -> tuple:
    abs_path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0.0
    return (abs_path, mtime, float(strength), int(num_blocks))


def _remap_anima_blocks(weights: dict[str, list], num_blocks: int) -> dict[str, list]:
    loaded_indices = set()
    for name in weights:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            loaded_indices.add(int(parts[1]))

    if not loaded_indices or len(loaded_indices) == num_blocks:
        return weights

    loaded_n = len(loaded_indices)
    mapping = None
    try:
        from backend.nn.anima import BLOCK_MAPPINGS

        mapping = BLOCK_MAPPINGS.get((loaded_n, num_blocks))
    except Exception:
        pass

    if mapping is None:
        return weights

    remapped: dict[str, list] = {}
    for block_idx, source_idx in enumerate(mapping):
        prefix_src = f"blocks.{source_idx}."
        prefix_dst = f"blocks.{block_idx}."
        for name, entry in weights.items():
            if name.startswith(prefix_src):
                target_name = prefix_dst + name[len(prefix_src) :]
                remapped[target_name] = [entry[0].clone(), entry[1].clone(), entry[2]]
    return remapped


class _Adapter:
    """LoRA weights applied through forward hooks, active only while this adapter's key is selected."""

    def __init__(self, path: str, strength: float, num_blocks: int = 28):
        self.key = _adapter_key(path, strength, num_blocks)
        self.path = path
        self.strength = float(strength)

        # Inspect adapter metadata for v2 / asymmetric architecture
        self.metadata = {}
        try:
            with safe_open(path, "pt") as handle:
                self.metadata = json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}"))
        except Exception:
            self.metadata = {}

        self.asym = self.metadata.get("arch") == "asym"
        self.ref_lora = self.metadata.get("ref_lora", True)
        slots = self.metadata.get("slots") or {}
        char_slot = slots.get("character") or {}
        self.offset_h = float(char_slot.get("offset_h", 1.0))
        self.ref_cache: dict = {}

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

        anima_weights = {k: v for k, v in raw_weights.items() if k.startswith("blocks.")}
        if not anima_weights:
            raise ValueError(
                f"'{os.path.basename(path)}' does not contain Anima DiT adapter weights (0 'blocks.*' keys found). "
                f"Anima Ref Adapter requires '{V2_LORA_FILENAME}' or another Anima reference adapter."
            )

        self.weights = _remap_anima_blocks(anima_weights, num_blocks)

    @property
    def nbytes(self) -> int:
        return sum(down.numel() * 2 + up.numel() * 2 for down, up, _ in self.weights.values())

    def offload(self) -> None:
        """Move weights back to CPU after sampling to release GPU VRAM."""
        for entry in self.weights.values():
            entry[0], entry[1] = entry[0].cpu(), entry[1].cpu()
        self.ref_cache.clear()

    @classmethod
    def get(cls, path: str, strength: float, num_blocks: int = 28) -> "_Adapter":
        key = _adapter_key(path, strength, num_blocks)
        if key not in _adapters:
            _adapters[key] = cls(path, strength, num_blocks)
        return _adapters[key]

    def install(self, diffusion_model: torch.nn.Module) -> None:
        installed_count = 0
        for name, entry in self.weights.items():
            try:
                module = diffusion_model.get_submodule(name)
            except AttributeError:
                continue

            installed = getattr(module, HOOKS_ATTR, None)
            if installed is None:
                installed = {}
                setattr(module, HOOKS_ATTR, installed)
            if self.key in installed:
                installed_count += 1
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
            installed_count += 1

        if installed_count == 0:
            raise RuntimeError(
                f"[Anima Ref Adapter] Failed to install forward hooks into diffusion_model. "
                f"The selected LoRA does not match Anima DiT architecture."
            )


# region v2 Asymmetric Attention Engine


def _frame_positions(t_index: int, height: int, width: int, h0: int, w0: int, device: torch.device):
    hh, ww = torch.meshgrid(
        torch.arange(height, device=device) + h0,
        torch.arange(width, device=device) + w0,
        indexing="ij",
    )
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def _rope_from_positions(pe, t: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """VideoRopePosition3DEmb embeddings for explicit per-token (t, h, w) coordinates: [L, D/2, 2, 2]."""
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _rope(pe, t_index: int, height: int, width: int, h0: int, w0: int, device: torch.device) -> torch.Tensor:
    t, h, w = _frame_positions(t_index, height, width, h0, w0, device)
    return _rope_from_positions(pe, t, h, w)


def _frame_embed(dm, latent: torch.Tensor, t_index: int, h0: int, w0: int, sigma: torch.Tensor):
    """Patch embedding, RoPE and timestep embedding of one frame [B, C, 1, H, W]."""
    x_emb, _, _ = dm.prepare_embedded_sequence(latent)
    _, _, hp, wp, _ = x_emb.shape
    rope = _rope(dm.pos_embedder, t_index, hp, wp, h0, w0, latent.device)
    emb, adaln = dm.t_embedder[1](dm.t_embedder[0](sigma.reshape(-1, 1).float()).to(x_emb.dtype))
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()
    return x_emb, rope.unsqueeze(1).unsqueeze(0), dm.t_embedding_norm(emb), adaln


def _mods(block, emb: torch.Tensor, adaln: torch.Tensor):
    """(shift, scale, gate) [B, T, 1, 1, D] for self-attn, cross-attn, and MLP."""
    return [
        tuple(t[:, :, None, None, :] for t in (m(emb) + adaln).chunk(3, dim=-1))
        for m in (block.adaln_modulation_self_attn, block.adaln_modulation_cross_attn, block.adaln_modulation_mlp)
    ]


def _norm(x: torch.Tensor, layer_norm, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return torch.addcmul(shift, layer_norm(x), 1 + scale)


def _asym_block(
    block,
    xt: Optional[torch.Tensor],
    xr: Optional[torch.Tensor],
    emb_t: Optional[torch.Tensor],
    emb_r: Optional[torch.Tensor],
    adaln_t: Optional[torch.Tensor],
    adaln_r: Optional[torch.Tensor],
    context: Optional[torch.Tensor],
    rope_t: Optional[torch.Tensor],
    rope_r: Optional[torch.Tensor],
    ref_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    options: Optional[dict] = None,
):
    """Block.forward with asymmetric attention. Returns (xt, xr, (k_r, v_r))."""
    attn = block.self_attn
    k_r = v_r = None

    if xr is not None:
        sa, _, mlp = _mods(block, emb_r, adaln_r)
        residual, compute = xr.dtype, emb_r.dtype
        B, T, H, W, _ = xr.shape
        n = _norm(xr, block.layer_norm_self_attn, sa[1], sa[0]).to(compute).flatten(1, 3)
        q_r, k_r, v_r = attn.compute_qkv(n, None, rope_emb=rope_r)
        xr = torch.addcmul(xr, sa[2].to(residual), attn.compute_attention(q_r, k_r, v_r).reshape(B, T, H, W, -1).to(residual))
        n = _norm(xr, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute)
        xr = torch.addcmul(xr, mlp[2].to(residual), block.mlp(n).to(residual))
    elif ref_kv is not None:
        k_r, v_r = ref_kv

    if xt is not None:
        options = options or {}
        sa, ca, mlp = _mods(block, emb_t, adaln_t)
        residual, compute = xt.dtype, emb_t.dtype
        B, T, H, W, _ = xt.shape
        n = _norm(xt, block.layer_norm_self_attn, sa[1], sa[0]).to(compute).flatten(1, 3)
        q_t, k_t, v_t = attn.compute_qkv(n, None, rope_emb=rope_t)
        n_q = q_t.shape[1]
        if k_r is not None:
            k_t = torch.cat([k_t, k_r.to(k_t)], dim=1)
            v_t = torch.cat([v_t, v_r.to(v_t)], dim=1)

        # Safe PAG compatibility: pad queries with zeros so attention modules expecting equal query/key count work
        if k_t.shape[1] != n_q and "compute_attention" in attn.__dict__:
            q_t = torch.cat([q_t, q_t.new_zeros(q_t.shape[0], k_t.shape[1] - n_q, *q_t.shape[2:])], dim=1)

        out = attn.compute_attention(q_t, k_t, v_t, transformer_options=options)[:, :n_q].reshape(B, T, H, W, -1)
        xt = torch.addcmul(xt, sa[2].to(residual), out.to(residual))
        n = _norm(xt, block.layer_norm_cross_attn, ca[1], ca[0]).to(compute).flatten(1, 3)
        out = block.cross_attn(n, context, rope_emb=None, transformer_options=options).reshape(B, T, H, W, -1)
        xt = torch.addcmul(xt, ca[2].to(residual), out.to(residual))
        n = _norm(xt, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute)
        xt = torch.addcmul(xt, mlp[2].to(residual), block.mlp(n).to(residual))

    return xt, xr, (k_r, v_r)


def encode_ref(dm, ref: torch.Tensor, offset: tuple[int, int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-block reference keys/values: the reference stream run once."""
    xr, rope_r, emb_r, adaln_r = _frame_embed(dm, ref, 1, *offset, torch.zeros(ref.shape[0], device=ref.device))
    cache = []
    for block in dm.blocks:
        _, xr, kv = _asym_block(block, None, xr, None, emb_r, None, adaln_r, None, None, rope_r)
        cache.append(kv)
    return cache


def asym_forward(dm, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, ref_kv, options: dict) -> torch.Tensor:
    """Anima.forward for the target frame alone, attending to cached reference keys/values."""
    orig_shape = list(x.shape)
    x = pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    xt, rope_t, emb_t, adaln_t = _frame_embed(dm, x[:, :, :1], 0, 0, 0, timesteps.reshape(x.shape[0], -1)[:, 0])
    batch = xt.shape[0]
    for block, (k, v) in zip(dm.blocks, ref_kv):
        kv = (k.expand(batch, *k.shape[1:]), v.expand(batch, *v.shape[1:])) if k.shape[0] != batch else (k, v)
        xt, _, _ = _asym_block(block, xt, None, emb_t, None, adaln_t, None, context, rope_t, None, kv, options)
    out = dm.unpatchify(dm.final_layer(xt.to(context.dtype), emb_t, adaln_lora_B_T_3D=adaln_t))[:, :, :1]
    return out[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]


def _correct(guided: torch.Tensor, pos: torch.Tensor, correction: str, rescale_phi: float = 0.5) -> torch.Tensor:
    """APG (drop parallel guidance component) or CFG rescale."""
    dims = tuple(range(1, pos.ndim))
    corr = str(correction).lower()
    if corr == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        parallel = (update * unit).sum(dim=dims, keepdim=True) * unit
        return pos + (update - parallel).to(pos.dtype)
    if corr == "rescale":
        ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True).clamp_min(1e-6)
        phi = float(rescale_phi)
        return phi * guided * ratio + (1.0 - phi) * guided
    return guided


# endregion


# region Pilot Joint-Attention Support


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
    b, c, k, h, w = ref.shape
    reshaped = ref.permute(0, 2, 1, 3, 4).reshape(b * k, c, h, w)
    shrink = h > size[0]
    resized = F.interpolate(
        reshaped.float(),
        size=size,
        mode="area" if shrink else "bilinear",
    ).to(ref.dtype)
    return resized.reshape(b, k, c, size[0], size[1]).permute(0, 2, 1, 3, 4)


def _ref_forward(
    dm,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    refs: list[torch.Tensor],
    offset_ratio: float,
    transformer_options: dict,
) -> torch.Tensor:
    """Anima DiT forward pass for pilot joint-attention."""
    orig_shape = list(x.shape)
    was_4d = x.ndim == 4
    if was_4d:
        x = x.unsqueeze(2)

    patch = (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial)
    x = pad_to_patch_size(x, patch)
    ref = _pick_ref(refs, tuple(x.shape[-2:])).to(device=x.device, dtype=x.dtype)
    if ref.shape[0] != x.shape[0]:
        ref = repeat_to_batch_size(ref, x.shape[0])

    num_refs = ref.shape[2]
    frames = torch.cat([x[:, :, :1], ref], dim=2)
    x_emb, _, extra = dm.prepare_embedded_sequence(frames, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape

    positions = [_frame_positions(0, hp, wp, 0, 0, x.device)]
    for _ in range(num_refs):
        positions.append(_frame_positions(1, hp, wp, round(offset_ratio * hp), 0, x.device))
    rope = _rope_from_positions(dm.pos_embedder, *(torch.cat(axis) for axis in zip(*positions)))

    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    timesteps = torch.cat(
        [timesteps, torch.zeros(timesteps.shape[0], num_refs, device=timesteps.device, dtype=timesteps.dtype)],
        dim=1,
    )
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


# endregion


# region Image and Latent Preprocessing


def _flatten_alpha(src: Image.Image, fill: int) -> Image.Image:
    if src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info):
        rgba = src.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (fill, fill, fill, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return src.convert("RGB") if src.mode != "RGB" else src


def _trim_background(img: Image.Image, fill: int, tolerance: int = 10, coverage: float = 0.985) -> Image.Image:
    arr = np.asarray(img, dtype=np.int16)
    h, w = arr.shape[:2]
    if h < 16 or w < 16:
        return img
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    spread = np.abs(corners - corners[0]).max()
    if spread > tolerance * 2:
        return img
    bg_mask = np.abs(arr - corners.mean(axis=0)).max(axis=-1) <= tolerance
    rows = bg_mask.mean(axis=1) >= coverage
    cols = bg_mask.mean(axis=0) >= coverage

    def _span(flags: np.ndarray) -> tuple[int, int]:
        lo, hi = 0, len(flags)
        while lo < hi and flags[lo]:
            lo += 1
        while hi > lo and flags[hi - 1]:
            hi -= 1
        return lo, hi

    y0, y1 = _span(rows)
    x0, x1 = _span(cols)
    if y1 - y0 < h * 0.2 or x1 - x0 < w * 0.2:
        return img
    if (y0, y1, x0, x1) == (0, h, 0, w):
        return img
    pad = round(min(h, w) * 0.02)
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def stitch_reference_images(
    images_with_weights: list[tuple[Image.Image, float]],
    width: int,
    height: int,
    fill: int = 255,
    gap_ratio: float = 0.02,
    trim: bool = True,
) -> Image.Image:
    """Compose multiple references into a single (width x height) sheet for pilot adapter."""
    fill = max(0, min(255, int(fill)))
    imgs = [_flatten_alpha(img, fill) for img, _ in images_with_weights]
    if trim:
        imgs = [_trim_background(img, fill) for img in imgs]
    weights = [max(0.05, float(w)) for _, w in images_with_weights]
    n = len(imgs)
    gap = round(min(width, height) * gap_ratio) if n > 1 else 0

    def cells_for(cols: int, rows: int) -> list[tuple[int, int, int, int]]:
        cells = []
        cell_h = (height - gap * (rows - 1)) / rows
        for r in range(rows):
            row_items = list(range(r * cols, min(n, (r + 1) * cols)))
            if not row_items:
                break
            row_w = [weights[i] for i in row_items] if rows == 1 else [1.0] * len(row_items)
            avail = width - gap * (len(row_items) - 1)
            total = sum(row_w)
            x = 0.0
            for i, rw in zip(row_items, row_w):
                cw = avail * rw / total
                cells.append((round(x), round(r * (cell_h + gap)), max(1, round(cw)), max(1, round(cell_h))))
                x += cw + gap
        return cells

    best, best_score = None, -1.0
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        if (rows - 1) * cols >= n:
            continue
        cells = cells_for(cols, rows)
        score = 0.0
        for img, (_, _, cw, ch) in zip(imgs, cells):
            s = min(cw / img.width, ch / img.height)
            score += img.width * img.height * s * s
        if score > best_score:
            best, best_score = cells, score

    sheet = Image.new("RGB", (width, height), (fill, fill, fill))
    for img, (x, y, cw, ch) in zip(imgs, best):
        s = min(cw / img.width, ch / img.height)
        size = (max(1, round(img.width * s)), max(1, round(img.height * s)))
        resized = img.resize(size, Image.Resampling.LANCZOS)
        sheet.paste(resized, (x + (cw - size[0]) // 2, y + ch - size[1]))
    return sheet


def build_asym_reference_image(
    images_with_weights: list[tuple[Image.Image, float]],
    megapixels: float = 1.0,
    fill: int = 255,
) -> tuple[torch.Tensor, Image.Image]:
    """Compose references side-by-side horizontally and resize to `megapixels` keeping natural aspect ratio.

    Returns (pixels_tensor [1, H, W, 3] in 0..1, composite_pil_image).
    """
    fill = max(0, min(255, int(fill)))
    valid = [_flatten_alpha(img, fill) for img, w in images_with_weights if img is not None and float(w or 0) > 0]
    if not valid:
        raise ValueError("No valid reference images provided")

    if len(valid) == 1:
        canvas = valid[0]
    else:
        # Base height from first image, align other references side-by-side horizontally
        base_h = valid[0].height
        resized_parts = []
        for img in valid:
            w = max(1, round(img.width * base_h / img.height))
            resized_parts.append(img.resize((w, base_h), Image.Resampling.LANCZOS))
        total_w = sum(p.width for p in resized_parts)
        canvas = Image.new("RGB", (total_w, base_h), (fill, fill, fill))
        curr_x = 0
        for p in resized_parts:
            canvas.paste(p, (curr_x, 0))
            curr_x += p.width

    scale = (megapixels * 1024 * 1024 / max(1, canvas.width * canvas.height)) ** 0.5
    target_size = (
        max(16, round(canvas.width * scale / 16) * 16),
        max(16, round(canvas.height * scale / 16) * 16),
    )
    final_img = canvas.resize(target_size, Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(final_img, dtype=np.float32) / 255.0).unsqueeze(0)
    return pixels, final_img


def letterbox_image(src: Image.Image, width: int, height: int, fill: int = 255) -> tuple[torch.Tensor, Image.Image]:
    """Fit `src` into (width, height) keeping aspect ratio with padding."""
    fill = max(0, min(255, int(fill)))
    src = _flatten_alpha(src, fill)
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
    vae,
    width: int,
    height: int,
    fill: int = 255,
) -> tuple[dict[str, torch.Tensor], Image.Image]:
    """Letterbox and VAE-encode the reference image (for pilot adapter)."""
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


@torch.inference_mode()
def encode_multi_reference_latents(
    images_with_weights: list[tuple[Image.Image, float]],
    vae,
    width: int,
    height: int,
    fill: int = 255,
    combine_mode: str = "stitch",
    asym: bool = False,
    megapixels: float = 1.0,
) -> tuple[dict[str, torch.Tensor], list[Image.Image]]:
    """VAE-encode multiple references. Supports both v2 Asymmetric and v1 Pilot paths."""
    if not images_with_weights:
        raise ValueError("No reference images provided.")

    if asym:
        if combine_mode == "blend" and len(images_with_weights) > 1:
            base_w, base_h = images_with_weights[0][0].size
            scale = (megapixels * 1024 * 1024 / max(1, base_w * base_h)) ** 0.5
            target_size = (max(16, round(base_w * scale / 16) * 16), max(16, round(base_h * scale / 16) * 16))
            all_refs = []
            weights = [max(0.01, float(w)) for _, w in images_with_weights]
            norm_w = [w / sum(weights) for w in weights]
            preview_pils = []
            for img, _ in images_with_weights:
                p = _flatten_alpha(img, fill).resize(target_size, Image.Resampling.LANCZOS)
                preview_pils.append(p)
                pix = torch.from_numpy(np.asarray(p, dtype=np.float32) / 255.0).unsqueeze(0)
                lat = vae.first_stage_model.process_in(vae.encode(pix)).detach().cpu()
                if lat.ndim == 4:
                    lat = lat.unsqueeze(2)
                all_refs.append(lat)
            blended = sum(r * w for r, w in zip(all_refs, norm_w))
            return {"samples": blended}, preview_pils

        pixels, ref_pil = build_asym_reference_image(images_with_weights, megapixels, fill)
        ref = vae.encode(pixels)
        ref = vae.first_stage_model.process_in(ref).detach().cpu()
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        return {"samples": ref}, [ref_pil]

    # Pilot LoRA path
    if combine_mode == "stitch" and len(images_with_weights) > 1:
        sheet = stitch_reference_images(images_with_weights, width, height, fill)
        lat_dict, lb_pil = encode_reference_latents(sheet, vae, width, height, fill)
        return lat_dict, [lb_pil]

    all_refs: list[torch.Tensor] = []
    all_halfs: list[torch.Tensor] = []
    letterboxed_list: list[Image.Image] = []
    weights: list[float] = []

    for img, weight in images_with_weights:
        lat_dict, lb_pil = encode_reference_latents(img, vae, width, height, fill)
        all_refs.append(lat_dict["samples"])
        if "samples_half" in lat_dict and lat_dict["samples_half"] is not None:
            all_halfs.append(lat_dict["samples_half"])
        letterboxed_list.append(lb_pil)
        weights.append(max(0.01, float(weight)))

    total_w = sum(weights) or 1.0
    norm_weights = [w / total_w for w in weights]

    if combine_mode == "blend" and len(all_refs) > 1:
        blended_ref = sum(r * w for r, w in zip(all_refs, norm_weights))
        blended_half = (
            sum(h * w for h, w in zip(all_halfs, norm_weights))
            if len(all_halfs) == len(all_refs)
            else None
        )
        out = {
            "samples": blended_ref,
            "samples_half": blended_half,
        }
    else:
        standardized_refs = [r if r.ndim == 5 else r.unsqueeze(2) for r in all_refs]
        stacked_refs = torch.cat(standardized_refs, dim=2)

        stacked_halfs = None
        if len(all_halfs) == len(all_refs):
            standardized_halfs = [h if h.ndim == 5 else h.unsqueeze(2) for h in all_halfs]
            stacked_halfs = torch.cat(standardized_halfs, dim=2)

        out = {
            "samples": stacked_refs,
            "samples_half": stacked_halfs,
        }

    return out, letterboxed_list


# endregion


def _flat_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def patch_unet_for_anima_ref(
    unet,
    ref_latents: dict[str, torch.Tensor],
    config: AnimaRefAdapterConfig,
    cache: Optional[dict] = None,
):
    """Clone `unet` and attach the Anima Ref Adapter LoRA and guidance logic."""
    dm = unet.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        raise RuntimeError("Anima Ref Adapter requires an Anima diffusion model with `diffusion_model.blocks`.")

    adapter = _Adapter.get(config.lora_path, config.strength, num_blocks=len(blocks))
    adapter.install(dm)

    if adapter.asym:
        # --- v2 Asymmetric Attention path ---
        m = unet.clone()
        ref = ref_latents["samples"]
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        tokens = (ref.shape[-2] // 2) * (ref.shape[-1] // 2)
        kv_bytes = 2 * len(dm.blocks) * tokens * dm.blocks[0].x_dim * 2
        m.add_extra_preserved_memory_during_sampling(kv_bytes + adapter.nbytes)

        target_cache = cache if cache is not None else adapter.ref_cache

        def ref_kv(model, x):
            hp = -(-x.shape[-2] // model.diffusion_model.patch_spatial)
            key = (id(model.diffusion_model), round(adapter.offset_h * config.ref_offset_h * hp), x.dtype, x.device)
            if key not in target_cache:
                target_cache.clear()
                r = pad_to_patch_size(ref.to(device=x.device, dtype=x.dtype), (1, 2, 2))
                previous = _active["key"]
                _active["key"] = adapter.key if adapter.ref_lora else None
                try:
                    target_cache[key] = encode_ref(model.diffusion_model, r, (key[1], 0))
                finally:
                    _active["key"] = previous
            return target_cache[key]

        def model_wrapper(apply_model, args):
            c = args["c"]
            options = c.get("transformer_options", {})
            if options.get(NO_REF_KEY):
                return apply_model(args["input"], args["timestep"], **c)
            model = apply_model.__self__
            x, sigma = args["input"], args["timestep"]
            dtype = model.computation_dtype
            xc = model.predictor.calculate_input(sigma, x).to(dtype)
            t = model.predictor.timestep(sigma).float()
            context = c["c_crossattn"].to(dtype)
            adapter.install(model.diffusion_model)
            kv = ref_kv(model, xc)
            previous = _active["key"]
            _active["key"] = adapter.key
            try:
                out = asym_forward(model.diffusion_model, xc, t, context, kv, options).float()
            finally:
                _active["key"] = previous
            return model.predictor.calculate_denoised(sigma, out, x)

        def post_cfg(args):
            guided, pos = args["denoised"], args["cond_denoised"]
            model = args["model"]
            sigma = float(args["sigma"].flatten()[0])
            ref_end = float(getattr(config, "ref_end", 1.0))
            if config.ref_guidance != 1.0 and (ref_end >= 1.0 or sigma >= float(model.predictor.percent_to_sigma(ref_end)) - 1e-5):
                if math.isclose(args["cond_scale"], 1.0):
                    cond, ref_pred = args["cond"], pos
                else:
                    cond, ref_pred = args["uncond"], args["uncond_denoised"]
                options = args["model_options"].copy()
                options["transformer_options"] = {**options.get("transformer_options", {}), NO_REF_KEY: True}
                base, _ = calc_cond_uncond_batch(model, cond, None, args["input"], args["sigma"], options)
                guided = guided + (config.ref_guidance - 1.0) * (ref_pred - base)
            return _correct(guided, pos, config.correction, config.rescale_phi)

        m.set_model_unet_function_wrapper(model_wrapper)
        m.set_model_sampler_post_cfg_function(post_cfg)
        return m

    # --- Pilot Joint Temporal Attention path ---
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
    ref_end = float(getattr(config, "ref_end", 1.0))

    def anima_ref_cfg_function(args: dict) -> torch.Tensor:
        pos: torch.Tensor = args["cond_denoised"]
        neg: torch.Tensor = args["uncond_denoised"]
        x: torch.Tensor = args["input"]
        timestep: torch.Tensor = args["timestep"]
        cfg: float = float(args["cond_scale"])
        model = args["model"]
        model_options: dict = args["model_options"]

        sigma = float(timestep.flatten()[0])
        skip_guidance = ref_end < 1.0 and sigma < float(model.predictor.percent_to_sigma(ref_end)) - 1e-5

        if ref_guidance != 1.0 and not skip_guidance:
            uncond_base = model_options.get("_anima_ref_uncond_base")
            if uncond_base:
                _, base = calc_cond_uncond_batch(model, [], uncond_base, x, timestep, model_options)
            else:
                base = neg
        else:
            base = neg

        guided = base + ref_guidance * (neg - base) + cfg * (pos - neg)
        guided = _correct(guided, pos, correction, rescale_phi)
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


def _soft_pag_attention(
    attn_module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    strength: float,
    heads: Optional[list[int]],
) -> torch.Tensor:
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


def _rescale_pag_guidance(
    guidance: torch.Tensor,
    cond_pred: torch.Tensor,
    cfg_result: torch.Tensor,
    rescale: float,
    mode: str,
) -> torch.Tensor:
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


def patch_unet_for_safe_pag(unet, pag_config: AnimaSafePAGConfig):
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


# endregion
