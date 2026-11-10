# Anima Ref Adapter for Forge Neo

English | [Русский](README.ru-RU.md)

Generate new scenes with Anima in Forge Neo while carrying over a character's appearance and, optionally, art style from reference images. Choose an Anima Ref or SnapRef adapter in txt2img or img2img, then use the prompt to describe the scene.

## Features

- Up to four reference slots with individual weights. Set a slot's weight to `0` to skip it.
- SnapRef modes for transferring character and style together or focusing on character appearance.
- A panel that shows the selected adapter's settings and stacks the controls on narrow screens.
- Optional Anima Safe PAG, guidance presets for Anima Ref, and canvas and sampler shortcuts.

## Supported adapters

Use an Anima checkpoint with its text encoder and VAE configured in Forge Neo. Choose the reference weights separately in **Adapter LoRA**.

| Adapter | Weights | Reference layouts |
| --- | --- | --- |
| Anima Ref v2 | `anima_ref_adapter_v2_r64.safetensors` | Side by side, Latent blend |
| Anima Ref Pilot | `anima_ref_adapter_pilot_r32.safetensors` | Side by side, Latent blend, Separate frames |
| SnapRef | v2a stable, v2a-pre1, v2a-pre2, v1a, and v1b in BF16 and FP8 | Separate, Side by side |

The extension converts SnapRef FP8 weights to BF16 at load time. SnapRef supports the 28-block Anima layout and maps its weights to the original blocks in the known 40-block Anima 2.9B layout. [Anima model files](https://huggingface.co/circlestone-labs/Anima) · [Anima Ref weights](https://huggingface.co/ElectronGirl/anima-ref-adapter) · [SnapRef releases](https://github.com/ArcAnima/SnapRef#quick-start) · [SnapRef v2a stable files](https://civitai.com/models/3028191/snapref?modelVersionId=3457619)

## Installation

1. In Forge Neo, open **Extensions** and select **Install from URL**.
2. Enter `https://github.com/LeonWGal/anima-ref-adapter-neo` and select **Install**.
3. Restart Forge Neo.

For a manual install, run this command from the Forge Neo `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/LeonWGal/anima-ref-adapter-neo sd-forge-anima-ref-adapter
```

## Quick start

1. Select an Anima checkpoint and open **Anima Ref Adapter** in txt2img or img2img.
2. Select an adapter in **Adapter LoRA**. Use **Download weights** to fetch the file in advance. The extension also fetches missing listed weights when you generate with a reference. For manual installs, put the adapter file in the extension's `models` directory and select **Refresh**.
3. Upload or paste a reference into **Ref 1**. A check mark appears on the slot and its weight control becomes available.
4. With SnapRef, choose **Character & Style** or **Character**. In **Character** mode, describe the desired style in the prompt.
5. Describe the new scene and generate. Enable **Show processed references in gallery** to inspect the images the adapter received.

For multiple views of one character, use SnapRef **Separate** and adjust each slot's weight. **Ref size (MP)** limits each reference individually in this layout. For two characters, use **Side by side**, order the images by slot, and describe who appears on the left and right.

**Canvas & sampler** contains the shortcuts. **Apply sampler preset** sets ER SDE, Simple, 30 steps, and CFG 4.5. **Match canvas to reference** sets the aspect ratio from the first active reference at about 1 MP.

## Examples

These images demonstrate pose changes with **Anima Ref Pilot** on `rdbtAnima_v30BaseSma_fp16`. Each output uses its own seed. The PNGs include their generation settings.

| Input reference | Sitting | Yoga | Punching |
| --- | --- | --- | --- |
| ![Reference generated with betterDaysIllustrious](assets/ref_betterdays_input.png) | ![The reference character sitting](assets/output_rdbtanima_wariza.png) | ![The reference character doing yoga](assets/output_rdbtanima_yoga.png) | ![The reference character punching toward the viewer](assets/output_rdbtanima_punch.png) |

## Model testing

Use the [comparison protocol](docs/model-testing.md#english) to test adapters on one checkpoint or test checkpoints with one adapter. It includes a reference image, prompts, fixed generation settings, and a results template.

## Credits and license

This extension integrates the Anima Ref Adapter, SnapRef, and Anima Safe PAG with Forge Neo. [Anima Ref Adapter](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter) · [SnapRef](https://github.com/ArcAnima/SnapRef) · [Anima Safe PAG](https://github.com/iljung1106/comfyui-anima-safe-pag)

The extension code uses the [Apache-2.0 license](LICENSE). See each model's release page for its weights license.
