# Порядок реализации: что делаете вы, что запускаю я (2026-11-24)

Всё, что можно сделать без внешнего доступа, уже сделано (инкременты 56–72). Ниже — очередь по принципу «минимум усилий с вашей стороны → максимум разблокированных проверок». После каждого шага указано, что я запускаю сам.

| № | Ваше действие | Время у вас | Что я запускаю после | Что это даёт |
|---|---|---|---|---|
| 1 ✅ | **Решение по v5** (принято 2026-11-24: оставить v4, D112): оставить v4 или принять v5 по точечной оценке (COVISLEEP +0.009 каппы, ДИ [−0.006, +0.024]) | 1 минута | при «принять»: запись отступления в `DECISIONS.md`, переключение `sleep.aasm_model`, хэш в `tests/test_model_inventory.py`, карточка модели, тесты | закрывает N1-линию |
| 2 | **NMT-4K**: на `zenodo.org/records/21405022` нажать «Request access», цель — текст из `data_access_requests.md` §2 | 5 минут + ожидание автора | инкремент 59 (пре-регистрация готова): модель записи, разряды, вспышка-подавление на 1 000 новых записей | самая большая независимая проверка заключения и детекторов |
| 3 | **Kaggle HMS**: создать API-токен Kaggle, принять правила конкурса, добавить токен в окружение сессии (Settings → Environment → network secrets, хост `www.kaggle.com`/`storage.googleapis.com`), начать новую сессию | 15 минут | инкремент 60 (черновик пре-регистрации готов): LPD/GPD/LRDA/GRDA на метках экспертов | первая проверка периодических паттернов ACNS |
| 4 | **PhysioNet-аккредитация** (CITI «Data or Specimens Only Research» + заявка): откроет NCH Sleep DataBank и CPS; затем заявки на `jsmf-access` (интраоперационная ЭЭГ 646 пациентов под наркозом; текст ниже) | 1–3 часа + ожидание | независимый большой тест v5/N1 (NCH) → закрывает решение по v5; наркозные состояния (jsmf) | правило N1 и седация на независимых данных |
| 5 | **TUH**: заполнить форму NEDC (help@nedcdata.org), подписанное соглашение | 15 минут + ожидание | `scripts/run_tuab_eval.py`, пре-регистрация 32 готова | внешняя проверка заключения на другой стране/приборе |
| 6 | **Открыть домены** `eutils.ncbi.nlm.nih.gov` и издательские сайты для сеанса (настройки сети окружения) | 5 минут | сверка всех цитат в `neuro_rules.yaml` (отметки «(сверить)») | снимает главный пункт аудита правил |
| 7 | **Нейролог-рецензент**: передать `docs/rule_base_audit_2026-11-19.md` + `configs/rules/neuro_rules.yaml` | зависит от рецензента | правки по замечаниям, повторный тест регрессий | клиническая валидность правил (без этого — не медицинское изделие) |
| 8 | **BDSP/IIIC**: аккредитация BDSP, DUA, AWS-ключи только с вашего явного «да» | недели | IIIC/SPaRCNet: метки экспертов для периодических паттернов (включая 413 скрытых пациентов I-CARE) | расширение пункта 3 |

## Текст заявки на jsmf-access (поле «research project / purpose»)

> Independent, pre-registered evaluation of an open-source EEG decision-support pipeline (NeuroLens) on anaesthetic brain states: burst suppression, sedation-related slowing and emergence. We will use the processed frontal-EEG measures to test whether the pipeline's context-conditioned rule base (sedatives present → slowing is interpreted as sedation, not encephalopathy) agrees with documented anaesthetic depth. Non-commercial research; no attempt at re-identification; the data will not be redistributed.

## Что именно ждёт на моей стороне (код и пре-регистрации готовы)

* NMT-4K — пре-регистрация 59, скрипты оценки.
* TUH — `datasets/tuab.py`, `scripts/run_tuab_eval.py`, пре-регистрация 32.
* Kaggle HMS — черновик пре-регистрации 60.
* NCH/CPS — после доступа пишу пре-регистрацию 73 (v5 против v4, те же критерии, что в инкременте 72, без пересмотра порогов).
* Цитаты — скрипт сверки по DOI/PMID добавлю, когда откроют домены.

## Чего я не сделаю без вас

Заявки, подписи соглашений, ввод токенов и AWS-ключей, переключение продукта на v5 в обход пре-регистрированного правила — только по вашему явному решению.

## Ссылки для подачи заявок (проверены запросом 2026-11-24; Kaggle из сессии не открывается — ссылки без проверки)

| Источник | Ссылка |
|---|---|
| NMT-4K (Request access) | https://zenodo.org/records/21405022 |
| PhysioNet — аккредитация | https://physionet.org/settings/credentialing/ |
| PhysioNet — курс CITI | https://physionet.org/about/citi-course/ |
| NCH Sleep DataBank | https://physionet.org/content/nch-sleep/ |
| CPS (сон, ПСГ) | https://physionet.org/content/cps-dataset-sleep/ |
| jsmf-access (наркоз, 646 пациентов) | https://physionet.org/content/jsmf-access/ |
| TUH EEG (форма NEDC) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| BDSP (IIIC/SPaRCNet) | https://bdsp.io/ |
| Kaggle HMS (правила конкурса) | https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification |
| Kaggle API-токен | https://www.kaggle.com/settings (раздел API → Create New Token) |
