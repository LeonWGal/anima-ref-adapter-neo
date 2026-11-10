# Anima Ref Adapter для Forge Neo

[English](README.md) | Русский

Генерируйте новые сцены в Forge Neo с Anima, перенося внешность персонажа и при желании стиль с референсов. В txt2img или img2img выберите адаптер Anima Ref или SnapRef, а в промпте опишите сцену.

## Возможности

- До четырёх референсов с отдельным весом для каждого. Вес `0` исключает слот.
- Режимы SnapRef для переноса персонажа вместе со стилем или с акцентом на внешности.
- Панель показывает настройки выбранного адаптера и складывает колонки на узком экране.
- Дополнительный Anima Safe PAG, пресеты guidance для Anima Ref и команды для настройки холста и сэмплера.

## Поддерживаемые адаптеры

Выберите чекпойнт Anima и подключите подходящие текстовый энкодер и VAE в Forge Neo. Веса референса выбираются отдельно в **Adapter LoRA**.

| Адаптер | Веса | Компоновка референсов |
| --- | --- | --- |
| Anima Ref v2 | `anima_ref_adapter_v2_r64.safetensors` | Side by side, Latent blend |
| Anima Ref Pilot | `anima_ref_adapter_pilot_r32.safetensors` | Side by side, Latent blend, Separate frames |
| SnapRef | v2a stable, v2a-pre1, v2a-pre2, v1a и v1b в BF16 и FP8 | Separate, Side by side |

При загрузке расширение преобразует веса SnapRef FP8 в BF16. SnapRef поддерживает 28-блочную Anima и направляет веса в исходные блоки известной 40-блочной архитектуры Anima 2.9B. [Файлы Anima](https://huggingface.co/circlestone-labs/Anima) · [Веса Anima Ref](https://huggingface.co/ElectronGirl/anima-ref-adapter) · [Релизы SnapRef](https://github.com/ArcAnima/SnapRef#quick-start) · [Файлы SnapRef v2a stable](https://civitai.com/models/3028191/snapref?modelVersionId=3457619)

## Установка

1. В Forge Neo откройте **Extensions** и выберите **Install from URL**.
2. Введите `https://github.com/LeonWGal/anima-ref-adapter-neo` и нажмите **Install**.
3. Перезапустите Forge Neo.

Для ручной установки выполните команду из каталога `extensions` Forge Neo, затем перезапустите Forge Neo.

```bash
git clone https://github.com/LeonWGal/anima-ref-adapter-neo sd-forge-anima-ref-adapter
```

## Быстрый старт

1. Выберите чекпойнт Anima и откройте **Anima Ref Adapter** в txt2img или img2img.
2. В **Adapter LoRA** выберите адаптер. Кнопка **Download weights** скачивает файл заранее. Расширение также скачивает отсутствующие веса из списка при первой генерации с референсом. Для ручной установки положите файл в каталог `models` расширения и нажмите **Refresh**.
3. Загрузите или вставьте изображение в **Ref 1**. На вкладке появится отметка, а под картинкой откроется регулятор веса.
4. Для SnapRef выберите **Character & Style** или **Character**. В режиме **Character** задайте желаемый стиль в промпте.
5. Опишите новую сцену и запустите генерацию. Включите **Show processed references in gallery**, чтобы увидеть обработанные изображения, которые получил адаптер.

Для нескольких ракурсов одного персонажа используйте **Separate** и настройте веса слотов. В этом режиме **Ref size (MP)** ограничивает размер каждого референса отдельно. Для двух персонажей выберите **Side by side**, разложите картинки по слотам и укажите в промпте, кто находится слева и справа.

Команды для холста и сэмплера находятся в **Canvas & sampler**. **Apply sampler preset** выставляет ER SDE, Simple, 30 шагов и CFG 4.5. **Match canvas to reference** подбирает пропорции холста по первому активному референсу при площади около 1 Мп.

## Примеры

Примеры показывают смену позы с **Anima Ref Pilot** на `rdbtAnima_v30BaseSma_fp16`. У каждой генерации свой seed. Настройки сохранены в PNG.

| Исходный референс | Сидя | Йога | Удар |
| --- | --- | --- | --- |
| ![Референс, созданный в betterDaysIllustrious](assets/ref_betterdays_input.png) | ![Персонаж с референса сидит](assets/output_rdbtanima_wariza.png) | ![Персонаж с референса занимается йогой](assets/output_rdbtanima_yoga.png) | ![Персонаж с референса наносит удар в сторону зрителя](assets/output_rdbtanima_punch.png) |

## Тестирование моделей

В [протоколе сравнения](docs/model-testing.md#русский) есть референс, промпты, фиксированные настройки генерации и шаблон результатов. Используйте его для сравнения адаптеров на одном чекпойнте или чекпойнтов с одним адаптером.

## Авторы и лицензия

Расширение интегрирует Anima Ref Adapter, SnapRef и Anima Safe PAG в Forge Neo. [Anima Ref Adapter](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter) · [SnapRef](https://github.com/ArcAnima/SnapRef) · [Anima Safe PAG](https://github.com/iljung1106/comfyui-anima-safe-pag)

Код расширения распространяется под [Apache-2.0](LICENSE). Лицензии весов указаны на страницах соответствующих моделей.
