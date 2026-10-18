# Данные для следующих внешних проверок — что нужно запросить

Все открытые наборы, подходящие для независимой проверки, использованы (CHB-MIT, Siena,
Helsinki, NMT, NMT-Events, IFCN, Cork, Sleep-EDF, CAP). Следующие проверки требуют данных,
доступ к которым выдаётся **на имя исследователя** — это может сделать только владелец
проекта; код и пре-регистрация готовятся заранее, данные не открываются до фиксации гипотез.

## 1. TUH Abnormal EEG Corpus (TUAB) — главный кандидат: заключение рутинной ЭЭГ

* **Что:** Temple University Hospital (США), рутинные ЭЭГ с меткой «норма/отклонения» по
  заключению невролога; официальное разделение train/eval (оценочная часть — 276 записей).
  Другой центр, страна, прибор и популяция — то, чего не даёт NMT.
* **Как получить:** бесплатно; форма регистрации на сайте Neural Engineering Data Consortium
  (isip.piconepress.com → TUH EEG Corpus → «request access»), подписанное соглашение об
  использовании данных, после одобрения — логин для загрузки (rsync/https).
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

## 2. NMT-4K (NUST-MH-TUKL-4k) — вторая больница той же группы

* **Что:** ~4 000 рутинных ЭЭГ, 21 канал 10-20, 200 Гц, метки «норма/патология» по консенсусу
  неврологов; в именах файлов — больница (вероятно, больше одной), год, идентификатор.
  IEEE DataPort, DOI 10.21227/zs7f-gc90; на странице набора файлы не выложены.
* **Как получить:** запрос авторам (группа NUST / TUKL; на странице DataPort указаны авторы,
  в т. ч. из Fauji Foundation Hospital, Rawalpindi).
* **Что проверим:** то же, что для TUAB, отдельно по больницам — перенос внутри одной группы
  разметчиков, но на другом приборе.

Текст запроса:

> We are developing an open-source EEG decision-support pipeline whose record-level
> normal/abnormal model was trained on the NMT Scalp EEG dataset. We would like to request
> access to NUST-MH-TUKL-4k for a pre-registered external validation (frozen model, per
> hospital), non-commercial, without redistribution; results and code will be shared with you.

## 3. Неонатальные приступы другого центра

Открытого набора, кроме Helsinki, нет (авторы Helsinki называют свой единственным
общедоступным). Варианты — по запросу у групп, публиковавших детекторы на собственных данных
(например, INFANT, Cork — набор ANSeR не открыт). Без него чувствительность монитора к приступам
у новорождённых проверена только на одном центре.
