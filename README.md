# Anima Ref Adapter for Forge-Neo

**Anima Ref Adapter** (`anima-ref-adapter-neo`) is a native extension for [Forge-Neo](https://github.com/Haoming02/sd-webui-forge-classic).

It ports **[ElectronGirl/ComfyUI-Anima-Ref-Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)** and **[Anima Safe PAG](https://github.com/iljung1106/comfyui-anima-safe-pag)** directly into Forge-Neo.

> **Character and style from one reference image; pose, place, and framing from your prompt.**

---

## Features

- **Seamless Reference Conditioning:** Encodes the reference image via Wan 2.1 VAE with automatic letterboxing (`fill = 255`) and injects it as a second clean latent frame (`t = 0`).
- **RoPE Height Shift (`RoPE Offset H = 1.0`):** Shifts the reference frame's position coordinates, transferring identity and art style without pinning the original pose.
- **Selective Adapter LoRA & Reference Guidance:** Applies pilot LoRA weights (`anima_ref_adapter_pilot_r32.safetensors`, rank 32) exclusively to reference branches (`pos`/`neg`) and unhooks them from the base branch, enabling high `Reference Guidance` (4.5–6.0).
- **Anti-Burn Guidance Corrections:** Supports Adaptive Projected Guidance (`apg`) and CFG `rescale` to tame color saturation at high guidance values.
- **Adaptive UI:** Automatically detects [lobe-theme-neo](https://github.com/LeonWGal/lobe-theme-neo) or standard Forge-Neo themes and adjusts layout accordingly.
- **Optional Anima Safe PAG:** Perturbed-Attention Guidance on Anima DiT blocks for crisper line art and fine details.
- **Workflow Shortcuts:** 1-click sampler presets (`Euler`, `Simple`, `28 steps`, `CFG 4.5`), `~1MP` aspect ratio matching, and live letterbox preview.

---

## Installation

### Method 1: Via Forge-Neo Extensions Tab
1. Open Forge-Neo WebUI -> **Extensions** -> **Install from URL**.
2. Paste:
   ```text
   https://github.com/LeonWGal/anima-ref-adapter-neo
   ```
3. Click **Install**, then **Apply and restart UI**.

### Method 2: Manual Clone
```bash
cd <Forge-Neo-Root>/extensions
git clone https://github.com/LeonWGal/anima-ref-adapter-neo sd-forge-anima-ref-adapter
```

---

## Adapter Weights

Download the pilot LoRA **[`anima_ref_adapter_pilot_r32.safetensors`](https://huggingface.co/ElectronGirl/anima-ref-adapter/blob/main/anima_ref_adapter_pilot_r32.safetensors)** and place it into:
```text
models/Lora/anima_ref_adapter_pilot_r32.safetensors
```
*(Or click the download button in the extension interface.)*

---

## Recommended Generation Settings

| Setting | Recommended Value | Notes |
|---|---|---|
| **Model** | Anima Base v1.0 | Standard prompt LoRAs `<lora:...>` work normally |
| **Sampler / Scheduler** | `Euler`, `Simple`, `28–30` steps | Matches training setup |
| **CFG Scale** | `4.5` | Main text CFG |
| **Reference Guidance** | `4.5 – 6.0` | `1.0` = normal CFG; `4.5–6.0` = strong identity & style lock |
| **Guidance Correction** | `apg` | Or `rescale` (`Rescale Phi = 0.5`) to prevent over-saturation |
| **RoPE Offset H** | `1.0` | Training default |
| **Letterbox Fill** | `255` | 255 = white padding (training default) |
| **Anima Safe PAG** *(optional)* | `Scale = 1.3`, `Block = 18`, `Strength = 0.75` | Extra sharpness |

### Prompting Guide
- **Do not** specify character names, artist names, or visual tags already present in the reference image.
- Only describe the **new scene**, pose, framing, lighting, and general quality:
  ```text
  masterpiece, best quality, score_7, newest, 1girl, solo, full body, walking in rain, neon reflections, city street
  ```

---

## License

- Extension code: **Apache-2.0**
- Original nodes: [ElectronGirl/ComfyUI-Anima-Ref-Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter) (Apache-2.0)
- Model adapter weights are distributed under the **CircleStone Non-Commercial License**.
