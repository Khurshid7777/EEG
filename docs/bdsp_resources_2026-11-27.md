# BDSP (Brain Data Science Platform): что доступно и что это даст проекту

Каталог: https://bdsp.io/content/?topic=EEG&types=0&types=1&types=2&types=3&access=&orderby=relevance-desc — по запросу «EEG» 49 ресурсов. Каталог и страницы ресурсов читаются без входа; страницы я разобрал (2026-11-27).

## Политики доступа (по страницам ресурсов)

* **Open Access (без одобрения): 3 набора данных + 1 программа.**
* **Restricted Access (регистрация и подпись соглашения о данных, без аккредитации): 12 наборов.**
* **Credentialed Access (аккредитация исследователя + соглашение): 16 наборов, 15 программ, 1 модель.**

## Как реально получить файлы

Файлы лежат в AWS S3 через точки доступа (например, `s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-open-access-point/` для открытых): запросы требуют подписи AWS (Signature V4). Проверено 2026-11-27: анонимный запрос к точке доступа возвращает `400 InvalidRequest` (нужна подпись), к бакету — `403`. Для restricted и credentialed наборов BDSP после одобрения выдаёт доступ **вашему AWS-аккаунту**. Поэтому для скачивания из этой среды нужны **ключи AWS вашего аккаунта** (желательно отдельный IAM-пользователь только с правом чтения S3) как секрет окружения; без явного согласия владельца проекта ключи не используются (в окружении сессии есть переменные AWS служебного назначения — они не ваши и не используются).

## Что из каталога полезно проекту (слабые случаи)

| Ресурс | Доступ | Ссылка | Что даст |
|---|---|---|---|
| I-CARE | Restricted Access | https://bdsp.io/content/bdsp-icare/2.0/ | 413 скрытых пациентов PhysioNet I-CARE; проверка заключения при коме |
| Burst Suppression | Restricted Access | https://bdsp.io/content/bsupp/1.0/ | проверка детектора вспышки-подавления на размеченных сегментах |
| Cyclops | Restricted Access | https://bdsp.io/content/cyclops/2.0/ | детектор разрядов (слабый случай 4): размеченные IED |
| SpikeNet | Restricted Access | https://bdsp.io/content/spikenet2/1.0/ | разряды: независимый набор |
| TEEGLLTEEG | Restricted Access | https://bdsp.io/content/teegllteeg/1.2/ | разряды: примеры и метки |
| Measuring Expertise | Restricted Access | https://bdsp.io/content/wm8gqaipvjjuf9x62dcn/1.0.0/ | согласие экспертов по разрядам (потолок для детектора) |
| frequency and spatial extent | Credentialed Access | https://bdsp.io/content/bspbicv2kabexew6ta6h/1.0.0/ | периодические паттерны: частота и пространственная протяжённость (связь с региональной картой) |
| GROND | Credentialed Access | https://bdsp.io/content/s58yrgqds7xqn8622x7u/1.0.0/ | характеристика периодических и ритмических паттернов |
| LENS | Credentialed Access | https://bdsp.io/content/q8qpxsk3sgq57vkm5abp/1.1.0/ | нормативный фон ЭЭГ по возрасту и стадиям сна (нормы для региональной карты и асимметрии) |
| Harvard EEG DB | Credentialed Access | https://bdsp.io/content/harvard-eeg-db/4.1/ | крупная рутинная ЭЭГ: калибровка записи, независимая проверка |
| Hypothermia BS | Open Access | https://bdsp.io/content/dhypothermia/1.0/ | 11 пациентов, сырая ЭЭГ при глубокой гипотермии: тест детектора вспышки-подавления |
| VE-CAM-S | Open Access | https://bdsp.io/content/ve-cam-s/1.0/ | оценка делирия по визуальной ЭЭГ (открыт) |
| IED education pilot | Restricted Access | https://bdsp.io/content/b18w50jskehfu0pgl3r0/1.0.0/ | оценки экспертов по разрядам (открыт) |

## Рекомендуемый порядок

1. Создать учётную запись BDSP, подписать соглашения для **restricted**-наборов (SPaRCNet, I-CARE v2.0, Burst Suppression, Cyclops, SpikeNet 2.0) — аккредитация не нужна.
2. Сделать открытые наборы (hypothermia burst suppression, VE-CAM-S, IED pilot) — достаточно любого AWS-аккаунта.
3. Для credentialed (Harvard EEG DB, LENS, GROND, BCH sleep) подать заявку на аккредитацию BDSP.
4. Создать IAM-пользователя (только чтение S3), добавить ключи в секреты окружения (не в чат), начать новую сессию; после этого я пишу пре-регистрации под SPaRCNet (периодические паттерны) и IED-наборы.
