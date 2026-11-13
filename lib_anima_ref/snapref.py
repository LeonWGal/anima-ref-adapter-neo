"""SnapRef inference adapted from ArcAnima/SnapRef (Apache-2.0)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from backend.utils import pad_to_patch_size

from .adapter import (
    ATTENTION_BIAS_KEY,
    NO_REF_KEY,
    _flatten_alpha,
    _frame_positions,
    _project_attention,
    _rope_from_positions,
    _sdpa_attention,
    make_post_cfg,
    snapref_metadata,
)


SNAPREF_KEY = "anima_snapref"
_BASE_BLOCKS_IN_29B = (
    0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
    20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39,
)


def _read_lora_matrix(handle, key: str) -> torch.Tensor:
    matrix = handle.get_tensor(key)
    scale_key = f"{key}_scale"
    if scale_key in handle.keys():
        scale = handle.get_tensor(scale_key)
        if matrix.dtype == torch.float8_e4m3fn and (scale.dtype != torch.float32 or scale.shape != (matrix.shape[0], 1)):
            raise ValueError(f"Expected one FP32 scale per row for SnapRef tensor {key}.")
        return (matrix.float() * scale).to(torch.bfloat16)
    if matrix.dtype == torch.float8_e4m3fn:
        raise ValueError(f"Expected FP32 scale for FP8 SnapRef tensor {key}.")
    if matrix.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return matrix.to(torch.bfloat16)
    raise ValueError(f"Unsupported SnapRef tensor dtype {matrix.dtype} for {key}.")


@lru_cache(maxsize=2)
def _load_weights(path: str, mtime_ns: int):
    weights = {}
    with safe_open(path, "pt") as handle:
        metadata = snapref_metadata(handle)
        if not metadata:
            raise ValueError("Expected a SnapRef anima_ref_kv/1 adapter.")
        keys = set(handle.keys())
        for key in keys:
            if not key.startswith("anima_ref_kv.") or not key.endswith(".lora_down"):
                continue
            stem = key.removesuffix(".lora_down")
            _, stream, module = stem.split(".", 2)
            down = _read_lora_matrix(handle, key)
            up = _read_lora_matrix(handle, stem + ".lora_up")
            scale = float(handle.get_tensor(stem + ".alpha")) / down.shape[0]
            weights.setdefault(module, {})[stream] = down, up, scale
        mode_embed = handle.get_tensor("anima_ref_kv.mode_embed") if "anima_ref_kv.mode_embed" in keys else None
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
    bias: torch.Tensor | None = None
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


def _with_reference(attn, q, k, v, ref_kv, bias, options):
    ref_k, ref_v = ref_kv
    ref_k = ref_k.to(k).expand(q.shape[0], -1, -1, -1)
    ref_v = ref_v.to(v).expand(q.shape[0], -1, -1, -1)
    if "compute_attention" in attn.__dict__:
        # Safe PAG pairs each query with its own target value.
        target_tokens = q.shape[1]
        q = torch.cat((q, q.new_zeros(q.shape[0], ref_k.shape[1], *q.shape[2:])), dim=1)
        k = torch.cat((k, ref_k), dim=1)
        v = torch.cat((v, ref_v), dim=1)
        if bias is not None:
            mask = F.pad(bias, (target_tokens, 0)).unsqueeze(0)
            options = {**options, ATTENTION_BIAS_KEY: mask}
        return attn.compute_attention(q, k, v, transformer_options=options)[:, :target_tokens]
    target_tokens = k.shape[1]
    k = torch.cat((ref_k, k), dim=1)
    v = torch.cat((ref_v, v), dim=1)
    if bias is not None:
        mask = F.pad(bias, (0, target_tokens)).unsqueeze(0)
        return _project_attention(attn, _sdpa_attention(q, k, v, mask))
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
            return _with_reference(attn, q, k, v, state.kv[index], state.bias, options)

        cond, uncond = state.rows
        pos = _with_reference(attn, q[cond], k[cond], v[cond], state.kv[index], state.bias, options)
        neg = attn.compute_attention(q[uncond], k[uncond], v[uncond], transformer_options=options)
        out = pos.new_empty((q.shape[0], *pos.shape[1:]))
        out[cond], out[uncond] = pos, neg
        return out

    return forward


@torch.no_grad()
def _encode_reference(dm, x, ref, context, mode_embed, ref_t, state):
    stride = dm.patch_spatial
    ref = pad_to_patch_size(ref.to(x), (dm.patch_temporal, stride, stride))
    offset = tuple(
        round((target // stride - source // stride) / 2)
        for target, source in zip(x.shape[-2:], ref.shape[-2:])
    )
    latent, _, _ = dm.prepare_embedded_sequence(ref, padding_mask=None)
    _, _, hp, wp, _ = latent.shape
    rope = _rope_from_positions(dm.pos_embedder, *_frame_positions(ref_t, hp, wp, *offset, ref.device)).unsqueeze(1).unsqueeze(0)
    # .forward, not __call__: hooks other extensions put on these modules steer the target, not the reference
    sinusoid = dm.t_embedder[0].forward(torch.zeros(ref.shape[0], 1, device=ref.device))
    timestep, adaln = dm.t_embedder[1].forward(sinusoid.to(latent.dtype))
    timestep = dm.t_embedding_norm.forward(timestep)
    if mode_embed is not None:
        timestep = timestep + mode_embed.to(timestep)[None, None, :]
    context = context.to(x)
    latent = latent.float() if latent.dtype == torch.float16 else latent

    state.stream = "ref"
    state.record = []
    for block in dm.blocks:
        latent = block.forward(
            latent, timestep, context,
            rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None, transformer_options={},
        )
    kv = state.record
    state.record = None
    return kv


def _encode_references(dm, x, references, context, mode_embed, ref_t, state):
    records = [_encode_reference(dm, x, ref, context, mode_embed, ref_t, state) for ref, _ in references]
    bias = None
    if any(weight != 1.0 for _, weight in references):
        bias = torch.cat([
            x.new_full((record[0][0].shape[1],), math.log(weight))
            for record, (_, weight) in zip(records, references)
        ])
    if len(records) == 1:
        return records[0], bias
    combined = []
    for index in range(len(dm.blocks)):
        keys, values = zip(*(record[index] for record in records))
        combined.append((torch.cat(keys, dim=1), torch.cat(values, dim=1)))
        for record in records:
            record[index] = None
    return combined, bias


def patch_unet_for_snapref(
    unet,
    references,
    context,
    path: str,
    mode: str,
    cache: dict,
    in_uncond: bool = False,
    ref_guidance: float = 4.5,
    correction: str = "apg",
    rescale_phi: float = 0.5,
    ref_end: float = 1.0,
):
    """Attach the SnapRef streams and share the reference cache across sampling steps."""
    if not references or all(weight == 0 for _, weight in references):
        return unet
    dm = unet.get_model_object("diffusion_model")
    metadata, weights, mode_embed = _load_weights(os.path.abspath(path), os.stat(path).st_mtime_ns)
    modes = metadata.get("modes", ["cs"])
    if mode not in modes:
        raise ValueError(f"This SnapRef adapter does not support the {mode!r} mode.")
    if mode_embed is not None:
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
    tokens = sum((ref.shape[-2] // dm.patch_spatial) * (ref.shape[-1] // dm.patch_spatial) for ref, _ in references)
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
            use = [not options.get(NO_REF_KEY) and (f == 0 or in_uncond) for f in flags]
            if any(use):
                target = (tuple(x.shape[-2:]), x.device, x.dtype)
                cached = cache.get(kv_key)
                if cached is None or cached[0] != target:
                    cache.pop(kv_key, None)
                    cached = target, _encode_references(dm, x, references, reference_context, mode_embed, ref_t, state)
                    cache[kv_key] = cached
                state.kv, state.bias = cached[1]
                if not all(use):
                    n = x.shape[0] // len(flags)
                    row_use = [use[r // n] for r in range(x.shape[0])]
                    state.rows = (
                        [r for r, u in enumerate(row_use) if u],
                        [r for r, u in enumerate(row_use) if not u],
                    )

            state.stream = "tgt"
            return original_forward(x, timesteps, context, padding_mask=padding_mask, **kwargs)
        finally:
            state.stream = state.record = state.kv = state.bias = state.rows = None

    reference_context = context.detach().cpu()
    ref_t = int(metadata.get("ref_t", 10))
    m.add_object_patch("diffusion_model.forward", wrapped_forward)
    m.set_transformer_option(SNAPREF_KEY, token)
    if in_uncond:
        m.set_model_sampler_post_cfg_function(make_post_cfg(ref_guidance, correction, rescale_phi, ref_end))
    return m
