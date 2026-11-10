# Model comparison protocol

[English](#english) | [Русский](#русский)

## English

### Prepare a run

Use **txt2img** and the [bundled reference](../assets/ref_betterdays_input.png) in **Ref 1** at weight `1`. Set other reference slots to `0`. Save the generated PNGs with their infotext. Keep the Forge Neo and extension revisions, GPU, checkpoint hash and compute dtype, text encoder, VAE, RNG mode, and shift in the run notes. Disable Anima Safe PAG, Hires. fix, prompt LoRAs, and Forge Neo's built-in Anima reference conditioning for these runs. Disable other extensions that alter generation settings.

| Setting | Value |
| --- | --- |
| Size | 832 × 1216 |
| Sampler / Schedule type | Euler / Simple |
| Steps / CFG scale | 30 / 4.5 |
| Seeds | 104729, 104759, 104761 |
| Batch size / Batch count | 1 / 1 |
| Ref size (MP) | 1.0 where available |
| Negative prompt | `low quality, blurry, text, watermark` |

Run each seed with the same scene prompt. Generate a reference-free baseline by closing **Anima Ref Adapter** with Safe PAG off. Then enable the adapter and reload the reference if needed. **PNG Info** can recover generation settings. Keep the reference image with the run so you can load it again.

| Scene | Prompt |
| --- | --- |
| Scene A, pose and framing | `1girl, solo, full body, sitting on a bench, park, autumn, daytime` |
| Scene B, lighting and setting | `1girl, solo, upper body, holding a cup, cafe, window, rainy evening` |

### Change one variable

| Run | Hold fixed | Change |
| --- | --- | --- |
| Adapter comparison | Anima Base v1.0, encoder, VAE, scene, seed and sampling settings | Reference off, Anima Ref Pilot, Anima Ref v2, SnapRef v1a, v1b, and v2a stable. Use the BF16 files for SnapRef. Add the previews to compare versions in the v2a series. |
| Anima checkpoint comparison | One adapter file and its settings, scene, seed and sampling settings | Anima Base v1.0, rdbtAnima, or another Anima checkpoint. Record any required encoder, VAE, dtype, or shift changes. |
| SnapRef precision comparison | One SnapRef version, Anima checkpoint and all settings | BF16 file versus the matching FP8 file. Record the diffusion model's compute dtype separately. |
| SnapRef mode comparison | One SnapRef file, reference, scene and seed | **Character & Style** versus **Character**. For a style-control check, add the same style tag to both prompts. |

For Anima Ref, start with the **Balanced** preset and **Side by side**. For SnapRef, start with **Character & Style** and **Separate**. These settings establish a starting point per adapter family. For a low-CFG check, rerun the same case at CFG `1` and keep it as a separate comparison.

For timing, warm up each configuration once. Time the next three runs with weights installed, and report the median. Record whether the time includes VAE decoding. Compare likeness, prompt adherence, style, and unwanted artifacts at the same seed. Keep all three seeds so a single result cannot dominate the comparison.

| Scene / seed | Checkpoint / hash | Adapter / mode / file dtype | Image or PNG Info | Time (s) | Peak VRAM (if measured) | Likeness, prompt, style, artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| A / 104729 | | Reference off | | | | |
| A / 104729 | | | | | | |

## Русский

### Подготовьте прогон

Откройте **txt2img**, загрузите [готовый референс](../assets/ref_betterdays_input.png) в **Ref 1** и установите вес `1`. Остальным слотам задайте вес `0`. Сохраняйте PNG с параметрами генерации. Запишите версии Forge Neo и расширения, GPU, хеш и точность вычислений чекпойнта, текстовый энкодер, VAE, режим RNG и shift. Для этих прогонов выключите Anima Safe PAG, Hires. fix, LoRA из промпта и встроенный режим референсов Anima в Forge Neo. Отключите другие расширения, которые меняют параметры генерации.

| Параметр | Значение |
| --- | --- |
| Разрешение | 832 × 1216 |
| Sampler / Schedule type | Euler / Simple |
| Steps / CFG scale | 30 / 4.5 |
| Seed | 104729, 104759, 104761 |
| Batch size / Batch count | 1 / 1 |
| Ref size (MP) | 1.0, если параметр доступен |
| Негативный промпт | `low quality, blurry, text, watermark` |

Повторите сцену с каждым seed. Для контрольной генерации сверните **Anima Ref Adapter**, оставив Safe PAG выключенным. Затем включите расширение и при необходимости загрузите референс заново. Параметры генерации можно вернуть из **PNG Info**. Сохраните референс рядом с результатами, чтобы загрузить его повторно.

| Сцена | Промпт |
| --- | --- |
| A, поза и кадрирование | `1girl, solo, full body, sitting on a bench, park, autumn, daytime` |
| B, свет и обстановка | `1girl, solo, upper body, holding a cup, cafe, window, rainy evening` |

### Меняйте один параметр

| Серия | Зафиксируйте | Меняйте |
| --- | --- | --- |
| Адаптеры | Anima Base v1.0, энкодер, VAE, сцену, seed и настройки сэмплинга | Контроль без референса, Anima Ref Pilot, Anima Ref v2, SnapRef v1a, v1b и v2a stable. Для SnapRef используйте файлы BF16. Для сравнения серии v2a добавьте превью. |
| Чекпойнты Anima | Один файл адаптера и его настройки, сцену, seed и настройки сэмплинга | Anima Base v1.0, rdbtAnima или другой чекпойнт Anima. Запишите обязательные смены энкодера, VAE, точности или shift. |
| Точность SnapRef | Одну версию SnapRef, чекпойнт и все настройки | Файлы BF16 и FP8 одной версии. Точность вычислений модели запишите отдельно. |
| Режимы SnapRef | Один файл SnapRef, референс, сцену и seed | **Character & Style** и **Character**. Для проверки управления стилем добавьте один и тот же тег стиля в оба промпта. |

Для Anima Ref начните с **Balanced** и **Side by side**, для SnapRef с **Character & Style** и **Separate**. Это исходные настройки для каждой группы адаптеров. Для проверки при низком CFG повторите тот же случай при CFG `1` и оформите его отдельным сравнением.

Перед измерением времени сделайте один прогревочный прогон каждой конфигурации. С установленными весами измерьте ещё три прогона и запишите медиану. Отметьте, входит ли в замер декодирование VAE. Сравнивайте сходство, следование промпту, стиль и артефакты при одинаковом seed. Сохраните результаты всех трёх seed.

| Сцена / seed | Чекпойнт / хеш | Адаптер / режим / точность файла | Картинка или PNG Info | Время (с) | Пиковая VRAM (если измерялась) | Сходство, промпт, стиль, артефакты |
| --- | --- | --- | --- | --- | --- | --- |
| A / 104729 | | Без референса | | | | |
| A / 104729 | | | | | | |
