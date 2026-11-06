# Данные для следующих внешних проверок — что нужно запросить

Все открытые наборы, подходящие для независимой проверки, использованы (CHB-MIT, Siena,
Helsinki, NMT, NMT-Events, IFCN, Cork, Sleep-EDF, CAP; позже — OpenNeuro ds004504, ds003775,
PhysioNet I-CARE, vepiset). Следующие проверки требуют данных,
доступ к которым выдаётся **на имя исследователя** — это может сделать только владелец
проекта; код и пре-регистрация готовятся заранее, данные не открываются до фиксации гипотез.

## 1. TUH Abnormal EEG Corpus (TUAB) — главный кандидат: заключение рутинной ЭЭГ

* **Что:** Temple University Hospital (США), рутинные ЭЭГ с меткой «норма/отклонения» по
  заключению невролога; официальное разделение train/eval (оценочная часть — 276 записей).
  Другой центр, страна, прибор и популяция — то, чего не даёт NMT.
* **Как получить:** бесплатно; форма регистрации на сайте Neural Engineering Data Consortium
  (isip.piconepress.com → TUH EEG Corpus → «request access»), подписанное соглашение об
  использовании данных, после одобрения — логин для загрузки (rsync/https).
* **Готово к запуску:** клиент `datasets/tuab.py`, скрипт `scripts/run_tuab_eval.py`, анализ
  `evaluation/prereg32.py`, пре-регистрация `docs/preregistration_increment32.md` (зафиксирована).
  Корпус положить в `data/tuab/`.
* **Что проверим (пре-регистрация до открытия):** модель уровня записи v2 с порогами по возрасту
  и пограничной зоной, как в `configs/impression.yaml`, на оценочной части TUAB:
  специфичность, чувствительность, AUC против чисел NMT (0.789 / 0.800 / 0.897); перенос модели
  разрядов; доля пограничных заключений. Порогов по TUAB не подбирать.

Текст запроса (поле «purpose of use»):

> Independent external validation of an open-source EEG decision-support pipeline
> (NeuroLens). The record-level normal/abnormal model was developed on the NMT Scalp EEG
> dataset (single centre, Pakistan); we will evaluate it, frozen and pre-registered, on the
> TUAB evaluation split to measure transfer to another hospital, device and population.
> Non-commercial research; no attempt at re-identification; data will not be redistributed.

## 2. NMT-4K-EEG — теперь на Zenodo, доступ по запросу

* **Что:** 4 500 рутинных ЭЭГ двух больниц Пакистана (19 каналов, 200 Гц, EDF), заключение
  «норма/патология» по проверке невролога, обезличенные клинические отчёты. У 1 164 патологических
  записей — экспертные метки событий: спайк-волна, острая волна, полиспайк, медленные волны,
  трифазные, вспышка-подавление, низкая амплитуда, артефакты.
* **Разделение:** заданное, по субъектам. Обучающая часть — 3 500 записей. Оценочная — 1 000
  (540 норм, 460 патологий), **ранее не публиковавшиеся** записи.
* **Как получить:** Zenodo, запись 21405022 (https://zenodo.org/records/21405022), доступ
  «restricted». Владелец проекта нажимает «Request access» под своей учётной записью Zenodo и
  указывает цель (текст ниже). После одобрения файлы скачиваются по ссылке или токену.
* **Что проверим:** только оценочную часть, независимую от NMT (D62).
  - Модель уровня записи v2: специфичность, чувствительность, AUC, по больницам.
  - Модель разрядов на экспертных метках событий: третий набор после NMT-Events и vepiset.
  - Детектор вспышки-подавления.

  Пре-регистрация — до скачивания, по образцу инкремента 32.
## 3. Неонатальные приступы другого центра

Открытого набора, кроме Helsinki, нет (авторы Helsinki называют свой единственным
общедоступным). Варианты — по запросу у групп, публиковавших детекторы на собственных данных
(например, INFANT, Cork — набор ANSeR не открыт). Без него чувствительность монитора к приступам
у новорождённых проверена только на одном центре.

## 4. Запись Natus (NicoletOne `.e`) вместе с её EDF-экспортом

Формат `.e` закрытый и восстановлен сторонними разработчиками. Открытые образцы (GIN, python-neo)
синтетические: сигнал перемешан во времени. Эталонные читатели недоступны:
* pynicolet 0.1.2 эти файлы не читает;
* читатель neo (PR 1750) не выпущен.

Без независимой сверки читатель не делается (D79–D81 сверены с MNE или neo). Лучшая сверка —
одна обезличенная запись с прибора Natus в двух видах: исходный `.e` и EDF-экспорт той же записи
из программы прибора. Отсчёты и метки сравниваются напрямую.

## 5. Экспертные метки периодических паттернов ACNS (IIIC / SPaRCNet, HMS)

Детекторы LPD, GPD, LRDA, GRDA на размеченных данных не проверены. Метки экспертов есть:
* **IIIC / SPaRCNet** (MGH): 50 697 фрагментов, 2 711 пациентов, голоса 124 экспертов из 18
  центров. Хранится в BDSP (AWS); нужно подписать Data Use Agreement BDSP на bdsp.io от имени
  исследователя.
* **HMS Harmful Brain Activity** (Kaggle): производная от тех же данных. Нужны учётная запись
  Kaggle, принятие правил конкурса и API-токен как секрет среды, а также разрешение kaggle.com в
  сети.

## Состояние на 2026-11-06 (проверено запросами из сессии) и что нужно от вас

| Источник | Сеть сессии | Как получить доступ (по публичным страницам) | Что готово у меня |
|---|---|---|---|
| **NMT-4K** (Zenodo 21405022) | zenodo.org открыт; список файлов закрыт (restricted) | На странице записи нажать «Request access» под вашей учётной записью Zenodo, указать цель (текст ниже) | Пре-регистрация `docs/preregistration_increment59.md` (зафиксирована до доступа); после одобрения нужен токен Zenodo — его я запрошу защищённой формой, не в чате |
| **TUH** (TUAB/TUSZ/TUEV) | `isip.piconepress.com` открыт | Заполнить и подписать PDF-форму со страницы TUH EEG Corpus и отправить на help@nedcdata.org (тема «Download The TUH EEG Corpus»); после регистрации ключей SSH загрузка по rsync — **на вашем компьютере**, затем подключить папку через Remote Control | Клиент `datasets/tuab.py`, `scripts/run_tuab_eval.py`, пре-регистрация инкремента 32 |
| **NCH Sleep DataBank** (3 984 детских ПСГ) | physionet.org открыт, файлы отдают 403 | Аккредитация PhysioNet (CITI-курс + соглашение), затем «credentialed access» к проекту `nch-sleep` | Пока ничего: нужна новая модель сна у детей (старая даёт каппу 0.17), пре-регистрация и разработка — после доступа |
| **BDSP** (GROND, IIIC, Pediatric SleepNet, Sleep Commons) | `bdsp.io` открыт; AWS-домены доступа не проверялись | Аккредитация BDSP (соглашение + курс); ключи AWS я не использую и не запрашиваю без вашего прямого «да» | Метки IIIC/SPaRCNet и план проверки периодических паттернов (раздел 5) |
| **Kaggle HMS** | `www.kaggle.com` закрыт (код 000) | Учётная запись Kaggle, правила соревнования, токен | Нужно открыть `www.kaggle.com`, `storage.googleapis.com` |
| **NSRR** (SHHS и др.) | `sleepdata.org` закрыт (код 000) | Соглашение NSRR на ваше имя | Нужно открыть `sleepdata.org` |

Текст запроса для NMT-4K (поле «purpose»):

> Independent external validation of an open-source EEG decision-support pipeline (NeuroLens).
> The record-level normal/abnormal model was developed on the NMT Scalp EEG dataset; we will evaluate it, frozen and
> pre-registered, on the evaluation split of NMT-4K-EEG only (1,000 previously unreleased recordings) to measure
> sensitivity and specificity on new recordings. Non-commercial research; no attempt at re-identification; data will
> not be redistributed.

Порядок по пользе: **1) NMT-4K** (одно нажатие, закрывает находку F1 по чувствительности); **2) TUH** (другая
больница; форма + SSH-ключи); **3) аккредитации PhysioNet и BDSP** (долго, закрывают сон у детей и периодические
паттерны); **4) Kaggle и NSRR** — только если вы готовы открыть домены.

