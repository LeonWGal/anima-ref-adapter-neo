# Anima Ref Adapter for Forge-Neo

**Anima Ref Adapter** (`anima-ref-adapter-neo`) is a native extension for [Forge-Neo](https://github.com/Haoming02/sd-webui-forge-classic).

It ports **[ElectronGirl/ComfyUI-Anima-Ref-Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)** (v2 Asymmetric Architecture) and **[Anima Safe PAG](https://github.com/iljung1106/comfyui-anima-safe-pag)** directly into Forge-Neo.

> **Character and style from reference images; pose, place, and framing from your prompt.**

<p align="center">
  <img src="assets/ui-preview.png" alt="Anima Ref Adapter UI in Forge-Neo" width="620" />
  <br>
  <em>Interface running in Forge-Neo with the <a href="https://github.com/LeonWGal/lobe-theme-neo">Lobe Theme Neo</a> theme.</em>
</p>

---

## What's New in v2 (Asymmetric Attention)

- **⚡ ~2x Faster Generation & Cached Reference KV:**
  In the new v2 architecture (`arch: "asym"`), reference tokens attend only to themselves and skip cross-attention. Reference Key and Value projections across all DiT blocks are computed **once** at the start of generation and cached for all sampling steps and CFG branches.
- **🖼️ Arbitrary Reference Aspect Ratios & Resolutions:**
  Reference images no longer require letterbox padding to canvas size. The reference is scaled cleanly via **Reference size (MP)** (default `1.0 MP`), preserving the image's native aspect ratio.
- **👥 Two Characters Support (Side-by-Side):**
  Add **Ref 1 (Left)** and **Ref 2 (Right)**. The extension automatically aligns them side-by-side into a single reference stream. Simply specify in your prompt who is on the left and who is on the right (`1girl on the left, 1girl on the right, hugging`).
- **⏱️ Denoising Window (`Ref guidance end`):**
  Setting guidance cutoff to `~0.6` skips guidance evaluation in the late steps where overall identity is already locked in, achieving **~40% faster generation** with virtually identical likeness! Quick preset **Turbo / Fast** configures this in 1 click.
- **🧹 Zero VRAM Leaks (Automatic Offloading):**
  Reference KV caches (~1 GB at 1 MP) and adapter weights are automatically released from GPU VRAM in `post_sample`.
- **🔄 Full Backward Compatibility:**
  Pilot adapter (`anima_ref_adapter_pilot_r32.safetensors`) and joint temporal attention continue to work seamlessly.

---

## Comparison & Examples

The table below demonstrates zero-shot character transfer using Anima Ref Adapter:
* **Reference Image:** Generated using model **`betterDaysIllustrious`** (`betterDaysIllustriousXL_illuXLV02BAKEDVAE`).
* **Outputs:** Rendered using model **`rdbtAnima`** (`rdbtAnima_v30BaseSma_fp16`) with Anima Ref Adapter (Strength 1.0, Ref Guidance 4.5, APG).

> [!TIP]
> Notice how the generated images don't require character appearance tags in the prompt — identity, hair, outfit cues, and art style are transferred automatically from the reference image, while the prompt guides only the pose and action!

| Input Reference (`betterDaysIllustrious`) | Output 1 (`rdbtAnima`) | Output 2 (`rdbtAnima`) | Output 3 (`rdbtAnima`) |
| :---: | :---: | :---: | :---: |
| <img src="assets/ref_betterdays_input.png" width="230" alt="Reference betterDaysIllustrious" /> | <img src="assets/output_rdbtanima_wariza.png" width="230" alt="Output rdbtAnima Wariza" /> | <img src="assets/output_rdbtanima_yoga.png" width="230" alt="Output rdbtAnima Yoga" /> | <img src="assets/output_rdbtanima_punch.png" width="230" alt="Output rdbtAnima Punch" /> |
| **Original Reference**<br/>*(betterDaysIllustrious)* | **`wariza, 1girl`**<br/>*(rdbtAnima)* | **`yoga pants, yoga, 1girl`**<br/>*(rdbtAnima)* | **`punching viewer, 1girl`**<br/>*(rdbtAnima)* |

---

## Features

- **Asymmetric Reference Cross-Attention (v2):** Reads reference appearance tokens shifted by one target height below the target grid (`RoPE Offset H = 1.0`), preventing pose copying while preserving identity.
- **Multi-Reference Image Modes:**
  - `Stitch` (default & recommended): Glues references side-by-side (Ref 1 on left, Ref 2 on right). Perfect for dual-character scenes!
  - `Multi-Frame Attention`: Injects each reference as its own frame in the 3D DiT (pilot LoRA).
  - `Latent Blend`: Computes a weighted linear interpolation across reference latents.
- **Anti-Burn Guidance Corrections:**
  - `apg` (Adaptive Projected Guidance, default): Removes guidance components parallel to conditional prediction to prevent over-saturation.
  - `rescale`: CFG rescale mode with customizable `Rescale Phi` (default 0.5).
  - `none`: Raw guidance difference.
- **Presets with Instant Feedback:**
  - **Balanced:** Strength 1.0, Guidance 4.5, End 1.0, APG
  - **Turbo / Fast:** Strength 1.0, Guidance 4.5, End 0.6, APG (~40% faster)
  - **Strong likeness:** Strength 1.1, Guidance 6.0, End 1.0, APG
  - **High guidance:** Strength 1.0, Guidance 6.5, End 1.0, Rescale 0.7
- **Optional Anima Safe PAG:** Perturbed-Attention Guidance on Anima DiT blocks for crisper line art and fine details.
- **Workflow Shortcuts:** 1-click optimal sampler configuration (`Euler`, `Simple`, `28 steps`, `CFG 4.5`), `~1MP` aspect ratio matching, and live reference preview.

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

Download the v2 adapter safetensors **[`anima_ref_adapter_v2_r64.safetensors`](https://huggingface.co/ElectronGirl/anima-ref-adapter/blob/main/anima_ref_adapter_v2_r64.safetensors)** (Rank 64, ~183 MB) and place it into:
```text
extensions/sd-forge-anima-ref-adapter/models/anima_ref_adapter_v2_r64.safetensors
```
*(Or simply click the download button in the extension interface — it will automatically download and verify the weights directly into the extension models directory.)*

The pilot adapter (`anima_ref_adapter_pilot_r32.safetensors`) is also supported.

---

## Recommended Generation Settings

| Setting | Recommended Value | Notes |
|---|---|---|
| **Model** | Anima Base v1.0 / `rdbtAnima` | Standard prompt LoRAs `<lora:...>` work normally |
| **Adapter LoRA** | `anima_ref_adapter_v2_r64.safetensors` | Upstream v2 asymmetric adapter |
| **Sampler / Scheduler** | `Euler`, `Simple` / `Normal`, `28–32` steps | Avoid stochastic samplers (SDE, Ancestral) to prevent digital noise |
| **CFG Scale** | `4.0 – 4.5` | Main text CFG |
| **Reference Guidance** | `4.5 – 6.0` | `1.0` = normal CFG; `4.5–6.0` = strong identity & style lock |
| **Guidance End (`ref_end`)** | `0.6 – 1.0` | `0.6` provides ~40% faster generation with identical output |
| **Reference Size (`ref_megapixels`)** | `1.0 MP` | Native reference resolution |
| **Guidance Correction** | `apg` | Or `rescale` (`Rescale Phi = 0.5`) to prevent over-saturation |
| **RoPE Offset H** | `1.0` | Training default |
| **Anima Safe PAG** *(optional)* | `Scale = 1.3`, `Block = 18`, `Strength = 0.75` | Extra sharpness |

### Prompting Guide
- **Do not** specify character names, artist names, or visual tags already present in the reference image.
- Only describe the **new scene**, pose, framing, lighting, and general quality:
  ```text
  wariza, 1girl, masterpiece, best quality, solo
  ```
- **Two Characters:**
  Place character 1 in Ref 1 (Left) and character 2 in Ref 2 (Right). In prompt, specify positions:
  ```text
  2girls, 1girl on the left, 1girl on the right, hugging, smile, masterpiece, best quality
  ```

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
