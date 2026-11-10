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
import tempfile
import urllib.request
from types import MethodType
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import SafetensorError, safe_open

from backend import memory_management
from backend.logging import setup_logger
from backend.sampling.condition import repeat_to_batch_size
from backend.sampling.sampling_function import calc_cond_uncond_batch
from backend.utils import load_torch_file, pad_to_patch_size

logger = logging.getLogger("AnimaRefAdapter")
setup_logger(logger)

OPTIONS_REF_KEY = "anima_ref_adapter"
HOOKS_ATTR = "_anima_ref_adapter_hooks"

HF_REPO = "ElectronGirl/anima-ref-adapter"
V2_LORA_FILENAME = "anima_ref_adapter_v2_r64.safetensors"
PILOT_LORA_FILENAME = "anima_ref_adapter_pilot_r32.safetensors"
SNAPREF_REPOS = {
    "snapref_v1a.safetensors": "0ppxnhximxr/snapref-v1a",
    "snapref_v1a_fp8.safetensors": "0ppxnhximxr/snapref-v1a",
    "snapref_v1b.safetensors": "0ppxnhximxr/snapref-v1b",
    "snapref_v1b_fp8.safetensors": "0ppxnhximxr/snapref-v1b",
    "snapref_v2a_pre1.safetensors": "0ppxnhximxr/snapref-v2a-pre1",
    "snapref_v2a_pre1_fp8.safetensors": "0ppxnhximxr/snapref-v2a-pre1",
    "snapref_v2a_pre2.safetensors": "0ppxnhximxr/snapref-v2a-pre2",
    "snapref_v2a_pre2_fp8.safetensors": "0ppxnhximxr/snapref-v2a-pre2",
}
HF_FILES = (V2_LORA_FILENAME, PILOT_LORA_FILENAME, *SNAPREF_REPOS)
SNAPREF_STABLE_URLS = {
    "snapref_v2a.safetensors": "https://civitai.com/api/download/models/3457619?fileId=3358798",
    "snapref_v2a_fp8.safetensors": "https://civitai.com/api/download/models/3457619?fileId=3361791",
}
SNAPREF_FILES = (*SNAPREF_STABLE_URLS, *SNAPREF_REPOS)
DOWNLOADABLE_FILES = (V2_LORA_FILENAME, PILOT_LORA_FILENAME, *SNAPREF_FILES)
NO_REF_KEY = "anima_ref_off"
ATTENTION_BIAS_KEY = "anima_ref_attention_bias"

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
    """Check for Anima DiT blocks.* keys without loading the tensors."""
    if not path or not os.path.isfile(path):
        return False
    try:
        with safe_open(path, "pt") as handle:
            return any("blocks." in key for key in handle.keys())
    except Exception:
        return False


def snapref_metadata(handle) -> dict:
    metadata = handle.metadata() or {}
    data = json.loads(metadata.get("snapref") or metadata.get("anima_ref_kv") or "{}")
    return data if isinstance(data, dict) and data.get("format") == "anima_ref_kv/1" else {}


def is_snapref_adapter(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        with safe_open(path, "pt") as handle:
            return bool(snapref_metadata(handle))
    except Exception:
        return False


def anima_ref_metadata(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        with safe_open(path, "pt") as handle:
            metadata = json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}"))
    except (OSError, ValueError, RuntimeError, SafetensorError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


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

    if lora_name in SNAPREF_FILES:
        path = os.path.join(_extension_models_dir(), lora_name)
        if _is_valid_safetensors(path):
            return path

    for p in (ext_v2, ext_pilot):
        if (lora_name == os.path.basename(p) or lora_name == p) and os.path.isfile(p) and _is_valid_safetensors(p):
            return p

    if os.path.isabs(lora_name):
        return lora_name if _is_valid_safetensors(lora_name) else None

    target_base = os.path.basename(lora_name).lower()
    target_clean = os.path.splitext(target_base)[0]
    models = list_lora_files(all_models=True)
    if lora_name in models:
        return models[lora_name]
    if os.path.dirname(lora_name):
        return None
    matches = []
    for rel, full in models.items():
        curr_base = os.path.basename(rel).lower()
        if curr_base == target_base or os.path.splitext(curr_base)[0] == target_clean:
            matches.append(full)

    return matches[0] if len(matches) == 1 else None


def _is_valid_safetensors(path: str) -> bool:
    """Verify that `path` is an existing, non-empty, valid safetensors file."""
    try:
        with safe_open(path, "pt") as handle:
            return bool(handle.keys())
    except Exception:
        return False


def download_adapter_lora(filename: str = V2_LORA_FILENAME) -> tuple[str, str]:
    """Download released adapter weights into the extension models directory."""
    if filename not in DOWNLOADABLE_FILES:
        raise ValueError(f"No download is configured for adapter {filename!r}.")
    target_dir = _extension_models_dir()
    dest_path = os.path.join(target_dir, filename)
    is_snapref = filename in SNAPREF_FILES
    if (is_snapref_adapter if is_snapref else is_anima_adapter)(dest_path):
        return dest_path, f"Verified in extension models folder: {dest_path}"

    repo = SNAPREF_REPOS.get(filename, HF_REPO)
    url = SNAPREF_STABLE_URLS.get(filename, f"https://huggingface.co/{repo}/resolve/main/{filename}")
    tmp_path = None
    logger.info(f"[Anima Ref Adapter] Downloading {url} -> {dest_path} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Forge-Neo-Anima-Ref-Adapter/2.0"})
        with urllib.request.urlopen(req, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=target_dir, prefix=f"{filename}.", suffix=".download", delete=False
        ) as out_file:
            tmp_path = out_file.name
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

        with safe_open(tmp_path, "pt") as handle:
            valid = bool(snapref_metadata(handle)) if is_snapref else any(
                key.startswith("diffusion_model.blocks.") for key in handle.keys()
            )
            if not valid:
                raise ValueError(f"Downloaded file does not contain {filename} adapter weights.")
        os.replace(tmp_path, dest_path)
        logger.info(f"[Anima Ref Adapter] Successfully downloaded {filename} ({os.path.getsize(dest_path)} bytes).")
        return dest_path, f"Successfully downloaded: {dest_path}"
    except Exception as exc:
        if tmp_path and os.path.isfile(tmp_path):
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

        self.metadata = anima_ref_metadata(path)
        self.asym = self.metadata.get("arch") == "asym"
        self.ref_lora = self.metadata.get("ref_lora", True)
        slots = self.metadata.get("slots") or {}
        char_slot = slots.get("character") or {}
        self.ref_t = int(char_slot.get("t", 1))
        self.offset_h = float(char_slot.get("offset_h", 1.0))
        self.offset_w = int(char_slot.get("offset_w", 0))

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


def _frame_embed(dm, latent: torch.Tensor, t_index: int, h0: int, w0: int, sigma: torch.Tensor, padding_mask=None):
    """Patch embedding, RoPE and timestep embedding of one frame [B, C, 1, H, W]."""
    x_emb, _, _ = dm.prepare_embedded_sequence(latent, padding_mask=padding_mask)
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


def asym_forward(dm, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, ref_kv, options: dict, padding_mask=None) -> torch.Tensor:
    """Anima.forward for the target frame alone, attending to cached reference keys/values."""
    orig_shape = list(x.shape)
    x = pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    sigma = timesteps.reshape(x.shape[0], -1)[:, 0]
    xt, rope_t, emb_t, adaln_t = _frame_embed(dm, x[:, :, :1], 0, 0, 0, sigma, padding_mask=padding_mask)
    batch = xt.shape[0]
    options = options.copy()
    for index, (block, (k, v)) in enumerate(zip(dm.blocks, ref_kv)):
        options["block_index"] = index
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
    ref_t: int,
    offset_h: float,
    offset_w: int,
    transformer_options: dict,
    padding_mask=None,
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
    x_emb, _, extra = dm.prepare_embedded_sequence(frames, padding_mask=padding_mask)
    _, _, hp, wp, _ = x_emb.shape

    positions = [_frame_positions(0, hp, wp, 0, 0, x.device)]
    for _ in range(num_refs):
        positions.append(_frame_positions(ref_t, hp, wp, round(offset_h * hp), offset_w, x.device))
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


def patch_unet_for_anima_ref(
    unet,
    ref_latents: dict[str, torch.Tensor],
    config: AnimaRefAdapterConfig,
    cache: Optional[dict] = None,
):
    """Clone the UNet and attach reference attention and post-CFG guidance."""
    dm = unet.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if not blocks:
        raise RuntimeError("Anima Ref Adapter requires an Anima diffusion model with diffusion_model.blocks.")
    adapter = _Adapter.get(config.lora_path, config.strength, num_blocks=len(blocks))
    adapter.install(dm)

    m = unet.clone()
    refs = [ref_latents["samples"]]
    if (half := ref_latents.get("samples_half")) is not None:
        refs.append(half)
    ref = refs[0]
    if ref.ndim == 4:
        ref = ref.unsqueeze(2)
    patch = (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial)
    if adapter.asym:
        tokens = math.prod(-(-size // step) for size, step in zip(ref.shape[-3:], patch))
        dtype_size = torch.empty((), dtype=unet.model.computation_dtype).element_size()
        kv_bytes = 2 * len(blocks) * ref.shape[0] * tokens * blocks[0].x_dim * dtype_size
        m.add_extra_preserved_memory_during_sampling(kv_bytes + adapter.nbytes)
    else:
        m.add_extra_preserved_memory_during_sampling(128 * 1024 * 1024)

    target_cache = {} if cache is None else cache
    offset_h = adapter.offset_h * config.ref_offset_h
    patch_token = object()  # Only this patcher's calls should use its references on the shared model.
    orig_forward = m.get_model_object("diffusion_model.forward")
    orig_forward = getattr(orig_forward, "_anima_ref_inner", orig_forward)

    def wrapped_forward(x, timesteps, context, padding_mask=None, **kwargs):
        options = kwargs.get("transformer_options") or {}
        if options.get(OPTIONS_REF_KEY) is not patch_token or options.get(NO_REF_KEY):
            return orig_forward(x, timesteps, context, padding_mask=padding_mask, **kwargs)

        adapter.install(dm)
        previous = _active["key"]
        try:
            if adapter.asym:
                target_rows = -(-x.shape[-2] // dm.patch_spatial)
                offset = (round(offset_h * target_rows), adapter.offset_w)
                key = (patch_token, offset, x.dtype, x.device)
                if key not in target_cache:
                    target_cache.clear()
                    _active["key"] = adapter.key if adapter.ref_lora else None
                    padded_ref = pad_to_patch_size(ref.to(x), patch)
                    target_cache[key] = encode_ref(dm, padded_ref, offset)
                _active["key"] = adapter.key
                return asym_forward(dm, x, timesteps, context, target_cache[key], options, padding_mask=padding_mask)

            _active["key"] = adapter.key
            return _ref_forward(
                dm, x, timesteps, context, refs, adapter.ref_t, offset_h, adapter.offset_w, options,
                padding_mask=padding_mask,
            )
        finally:
            _active["key"] = previous

    def post_cfg(args):
        guided, pos = args["denoised"], args["cond_denoised"]
        model, sigma = args["model"], args["sigma"]
        if config.ref_guidance != 1.0 and (
            config.ref_end >= 1.0 or _sigma_to_float(sigma) >= model.predictor.percent_to_sigma(config.ref_end) - 1e-5
        ):
            options = args["model_options"].copy()
            options["transformer_options"] = {**options.get("transformer_options", {}), NO_REF_KEY: True}
            # Forge passes the original negative conditioning even when it skips CFG 1's negative pass.
            if args["uncond"] and (
                not math.isclose(args["cond_scale"], 1.0) or options.get("disable_cfg1_optimization")
            ):
                ref_pred = args["uncond_denoised"]
                _, base = calc_cond_uncond_batch(model, [], args["uncond"], args["input"], sigma, options)
            else:
                ref_pred = pos
                base, _ = calc_cond_uncond_batch(model, args["cond"], None, args["input"], sigma, options)
            guided = guided + (config.ref_guidance - 1.0) * (ref_pred - base)
        return _correct(guided, pos, config.correction, config.rescale_phi)

    wrapped_forward._anima_ref_inner = orig_forward
    m.add_object_patch("diffusion_model.forward", wrapped_forward)
    m.set_transformer_option(OPTIONS_REF_KEY, patch_token)
    m.set_model_sampler_post_cfg_function(post_cfg)
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


def _sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias=None) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=bias,
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
    bias=None,
) -> torch.Tensor:
    normal = _sdpa_attention(q, k, v, bias)
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
        had_override = "compute_attention" in attn.__dict__
        heads = _parse_optional_indices(head_indices, attn.n_heads - 1)

        def compute_attention(self, q, k, v, transformer_options=None, _orig=original, _str=perturbation_strength, _heads=heads):
            # The v2 reference stream calls attention without target transformer options.
            if transformer_options is None:
                return _orig(q, k, v, transformer_options=transformer_options or {})
            bias = transformer_options.get(ATTENTION_BIAS_KEY)
            if _str <= 0 and bias is None:
                return _orig(q, k, v, transformer_options=transformer_options)
            return _soft_pag_attention(self, q, k, v, _str, _heads, bias)

        attn.compute_attention = MethodType(compute_attention, attn)
        patched.append((attn, original, had_override))
    return patched


def _restore_attention(patched) -> None:
    for attn, original, had_override in patched:
        if had_override:
            attn.compute_attention = original
        else:
            del attn.compute_attention


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
