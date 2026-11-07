"""SnapRef inference adapted from ArcAnima/SnapRef (Apache-2.0)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from backend.utils import pad_to_patch_size

from .adapter import _flatten_alpha, _frame_embed, snapref_metadata


SNAPREF_KEY = "anima_snapref"
_BASE_BLOCKS_IN_29B = (
    0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
    20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39,
)


def _read_lora_matrix(handle, key: str) -> torch.Tensor:
    matrix = handle.get_tensor(key)
    if matrix.dtype == torch.bfloat16:
        return matrix
    if matrix.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Unsupported SnapRef tensor dtype {matrix.dtype} for {key}.")

    scale = handle.get_tensor(f"{key}_scale")
    if scale.dtype != torch.float32 or scale.shape != (matrix.shape[0], 1):
        raise ValueError(f"Expected one FP32 scale per row for SnapRef tensor {key}.")
    return (matrix.float() * scale).to(torch.bfloat16)


@lru_cache(maxsize=2)
def _load_weights(path: str, mtime_ns: int):
    weights = {}
    with safe_open(path, "pt") as handle:
        metadata = snapref_metadata(handle)
        if not metadata:
            raise ValueError("Expected a SnapRef anima_ref_kv/1 adapter.")
        for key in handle.keys():
            if not key.startswith("anima_ref_kv.") or not key.endswith(".lora_down"):
                continue
            stem = key.removesuffix(".lora_down")
            _, stream, module = stem.split(".", 2)
            down = _read_lora_matrix(handle, key)
            up = _read_lora_matrix(handle, stem + ".lora_up")
            scale = float(handle.get_tensor(stem + ".alpha")) / down.shape[0]
            weights.setdefault(module, {})[stream] = down, up, scale
        mode_embed = handle.get_tensor("anima_ref_kv.mode_embed")
    if {stream for streams in weights.values() for stream in streams} != {"ref", "tgt"}:
        raise ValueError("SnapRef needs both reference and target LoRA weights.")
    return metadata, weights, mode_embed


def _block_map(weights, model_blocks: int):
    indices = {int(name.split(".")[1]) for name in weights if name.startswith("blocks.")}
    if indices == set(range(model_blocks)):
        return tuple(range(model_blocks))
    if model_blocks == 40 and indices == set(range(28)):
        return _BASE_BLOCKS_IN_29B
    raise ValueError(f"SnapRef weights for {len(indices)} blocks do not match the {model_blocks}-block model.")


@torch.inference_mode()
def encode_snapref_image(image: Image.Image, vae, megapixels: float, fill: int):
    """Scale down to the reference area limit and encode on the model's latent grid."""
    image = _flatten_alpha(image, fill)
    scale = min(1.0, (megapixels * 1024**2 / (image.width * image.height)) ** 0.5)
    size = tuple(max(16, int(side * scale) // 16 * 16) for side in image.size)
    preview = image.resize(size, Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(preview, dtype=np.float32) / 255.0).unsqueeze(0)
    ref = vae.first_stage_model.process_in(vae.encode(pixels)).detach().cpu()
    return (ref.unsqueeze(2) if ref.ndim == 4 else ref), preview


@dataclass
class _ForwardState:
    stream: str | None = None
    record: list | None = None
    kv: list | None = None
    rows: tuple[list[int], list[int]] | None = None


def _lora_forward(original, streams, state, cache, token):
    def forward(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        weights = streams.get(state.stream)
        if weights is None:
            return out
        down, up, scale = weights
        key = (token, id(down), x.device, x.dtype)
        if key not in cache:
            cache[key] = down.to(x), up.to(x)
        down, up = cache[key]
        return out + F.linear(F.linear(x, down), up) * scale

    return forward


def _with_reference(attn, q, k, v, ref_kv, options):
    ref_k, ref_v = ref_kv
    ref_k = ref_k.to(k).expand(q.shape[0], -1, -1, -1)
    ref_v = ref_v.to(v).expand(q.shape[0], -1, -1, -1)
    if "compute_attention" in attn.__dict__:
        # Safe PAG pairs each query with its own target value.
        target_tokens = q.shape[1]
        q = torch.cat((q, q.new_zeros(q.shape[0], ref_k.shape[1], *q.shape[2:])), dim=1)
        k = torch.cat((k, ref_k), dim=1)
        v = torch.cat((v, ref_v), dim=1)
        return attn.compute_attention(q, k, v, transformer_options=options)[:, :target_tokens]
    k = torch.cat((ref_k, k), dim=1)
    v = torch.cat((ref_v, v), dim=1)
    return attn.compute_attention(q, k, v, transformer_options=options)


def _attention_forward(attn, original, index, state):
    def forward(x, context=None, rope_emb=None, transformer_options=None):
        options = transformer_options or {}
        if state.record is None and state.kv is None:
            return original(x, context, rope_emb=rope_emb, transformer_options=options)

        q, k, v = attn.compute_qkv(x, context, rope_emb=rope_emb)
        if state.record is not None:
            state.record.append((k, v))
            return attn.compute_attention(q, k, v)

        if state.rows is None:
            return _with_reference(attn, q, k, v, state.kv[index], options)

        cond, uncond = state.rows
        pos = _with_reference(attn, q[cond], k[cond], v[cond], state.kv[index], options)
        neg = attn.compute_attention(q[uncond], k[uncond], v[uncond], transformer_options=options)
        out = pos.new_empty((q.shape[0], *pos.shape[1:]))
        out[cond], out[uncond] = pos, neg
        return out

    return forward


def _encode_reference(dm, x, ref, context, mode_embed, ref_t, state):
    stride = dm.patch_spatial
    ref = pad_to_patch_size(ref.to(x), (dm.patch_temporal, stride, stride))
    offset = tuple(
        round((target // stride - source // stride) / 2)
        for target, source in zip(x.shape[-2:], ref.shape[-2:])
    )
    latent, rope, timestep, adaln = _frame_embed(dm, ref, ref_t, *offset, torch.zeros(1, device=x.device))
    timestep = timestep + mode_embed.to(timestep)[None, None, :]
    context = context.to(x)

    state.stream = "ref"
    state.record = []
    for block in dm.blocks:
        latent = block(
            latent, timestep, context,
            rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln, transformer_options={},
        )
    kv = state.record
    state.record = None
    return kv


def patch_unet_for_snapref(unet, ref, context, path: str, mode: str, cache: dict):
    """Attach the SnapRef streams and share the reference cache across sampling steps."""
    dm = unet.get_model_object("diffusion_model")
    metadata, weights, mode_embed = _load_weights(os.path.abspath(path), os.stat(path).st_mtime_ns)
    modes = metadata.get("modes", ["cs"])
    if mode not in modes:
        raise ValueError(f"This SnapRef adapter does not support the {mode!r} mode.")
    mode_embed = mode_embed[modes.index(mode)]
    block_map = _block_map(weights, len(dm.blocks))

    m = unet.clone()
    state = _ForwardState()
    token = object()
    for name, streams in weights.items():
        if name.startswith("blocks."):
            _, index, suffix = name.split(".", 2)
            name = f"blocks.{block_map[int(index)]}.{suffix}"
        attr = f"diffusion_model.{name}.forward"
        original = m.get_model_object(attr)
        m.add_object_patch(attr, _lora_forward(original, streams, state, cache, token))

    for index, block in enumerate(dm.blocks):
        attr = f"diffusion_model.blocks.{index}.self_attn.forward"
        original = m.get_model_object(attr)
        m.add_object_patch(attr, _attention_forward(block.self_attn, original, index, state))

    dtype_size = torch.empty((), dtype=unet.model.computation_dtype).element_size()
    tokens = (ref.shape[-2] // dm.patch_spatial) * (ref.shape[-1] // dm.patch_spatial)
    kv_bytes = 2 * len(dm.blocks) * tokens * dm.blocks[0].x_dim * dtype_size
    lora_bytes = sum((a.numel() + b.numel()) * dtype_size for streams in weights.values() for a, b, _ in streams.values())
    m.add_extra_preserved_memory_during_sampling(kv_bytes + lora_bytes)

    original_forward = m.get_model_object("diffusion_model.forward")
    kv_key = (token, "kv")

    def wrapped_forward(x, timesteps, context, padding_mask=None, **kwargs):
        options = kwargs.get("transformer_options") or {}
        if options.get(SNAPREF_KEY) is not token:
            return original_forward(x, timesteps, context, padding_mask=padding_mask, **kwargs)

        try:
            flags = options.get("cond_or_uncond") or [0]
            if 0 in flags:
                target = (tuple(x.shape[-2:]), x.device, x.dtype)
                cached = cache.get(kv_key)
                if cached is None or cached[0] != target:
                    cache.pop(kv_key, None)
                    cached = target, _encode_reference(dm, x, ref, reference_context, mode_embed, ref_t, state)
                    cache[kv_key] = cached
                state.kv = cached[1]
                if 1 in flags:
                    state.rows = options["cond_indices"], options["uncond_indices"]

            state.stream = "tgt"
            return original_forward(x, timesteps, context, padding_mask=padding_mask, **kwargs)
        finally:
            state.stream = state.record = state.kv = state.rows = None

    reference_context = context.detach().cpu()
    ref_t = int(metadata.get("ref_t", 10))
    m.add_object_patch("diffusion_model.forward", wrapped_forward)
    m.set_transformer_option(SNAPREF_KEY, token)
    return m
