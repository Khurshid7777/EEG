# Какие домены открыть сети сессии (анализ от 2026-10-29)

Проверено прямыми запросами через прокси сессии. Код `000` — соединение отклонено политикой сети.

## Уже открыто (ничего добавлять не нужно)

physionet.org, archive.physionet.org, physionet-open.s3.amazonaws.com, zenodo.org, figshare.com,
api.figshare.com, data.mendeley.com, gin.g-node.org, huggingface.co, raw.githubusercontent.com,
github.com (git), s3.amazonaws.com (в том числе весь OpenNeuro: `s3.amazonaws.com/openneuro.org/dsNNNNNN/`),
openneuro-derivatives.s3.amazonaws.com, pypi.org, repo.anaconda.com, conda.anaconda.org.

## Что добавить, по приоритету

| № | Домены | Что даёт | Что ещё нужно от вас |
|---|---|---|---|
| 1 | `isip.piconepress.com`, `www.isip.piconepress.com` | TUH: TUAB (оценочная часть 276 записей — заключение рутинной ЭЭГ другой больницы), TUSZ (приступы, есть длительные), TUEV (спайки/периодика), TUAR (артефакты) | Регистрация и подписанное соглашение (help@nedcdata.org). Загрузка идёт по rsync через SSH: HTTPS-прокси её не пропустит, поэтому см. «Обходной путь» ниже |
| 2 | `www.kaggle.com`, `storage.googleapis.com` (и `*.kaggleusercontent.com`) | HMS Harmful Brain Activity: ~100 тыс. отрезков ЭЭГ с экспертными голосами по ритмам IIIC (приступ, LPD, GPD, LRDA, GRDA) — проверка периодических паттернов ACNS | Токен Kaggle в секретах окружения (`KAGGLE_USERNAME`, `KAGGLE_KEY`) и принятие правил соревнования на вашей учётной записи |
| 3 | `bdsp.io`, `*.s3-accesspoint.us-east-1.amazonaws.com` | Harvard EEG Database (HEEDB), >280 тыс. записей четырёх больниц Бостона, ICU/рутина, BIDS | Аккредитация BDSP (соглашение, курс CITI) и ключи AWS в секретах. Самый ценный источник по ICU и статусу, но самый тяжёлый по оформлению |
| 4 | `cdn-lfs.hf.co`, `cdn-lfs-us-1.hf.co`, `cas-bridge.xethub.hf.co`, `transfer.xethub.hf.co` | Крупные файлы с Hugging Face (huggingface.co уже открыт, но веса и большие наборы отдаёт CDN): предобученные ЭЭГ-модели, наборы на HF | — |
| 5 | `openneuro.org` | Поиск наборов через API OpenNeuro (сами данные уже доступны через s3.amazonaws.com) | — |
| 6 | `pubmed.ncbi.nlm.nih.gov`, `pmc.ncbi.nlm.nih.gov`, `europepmc.org`, `arxiv.org`, `doi.org`, `www.nature.com`, `link.springer.com`, `onlinelibrary.wiley.com` | Чтение описаний наборов и критериев (ACNS, IFCN) при пре-регистрациях: сейчас часть статей недоступна | — |
| 7 | `osf.io`, `files.osf.io`, `api.osf.io`, `datadryad.org`, `dataverse.harvard.edu`, `www.synapse.org`, `dandiarchive.org`, `api.dandiarchive.org` | Общие хранилища: многие открытые наборы ЭЭГ лежат на OSF и Dryad | — |
| 8 | `sleepdata.org` (NSRR) | Крупные ночные записи с гипнограммами (SHHS и др.) — внешняя проверка стадирования сна взрослых | Соглашение NSRR на ваше имя |
| 9 | `nemar.org`, `data.nemar.org` | Зеркало OpenNeuro-наборов ЭЭГ с метаданными | — |
| 10 | `rdr.kuleuven.be`, `www.epileptologie-bonn.de` | Малые наборы приступов (KU Leuven, Бонн): низкая ценность | — |

## Чего не хватает по сути (зачем это всё)

* Статус эпилептический и длительные приступы: открытых наборов с такой разметкой нет (проверено,
  `docs/increment47_report.md`). Реальные источники — TUSZ (№1) и HEEDB (№3).
* Независимая проверка заключения рутинной ЭЭГ и разрядов: TUAB/TUEV (№1), NMT-4K (запрос на Zenodo).
* Периодические паттерны ACNS и ICU: HMS (№2), HEEDB (№3).
* Уже доступно без изменений: SeizeIT2 (OpenNeuro ds005873, 125 пациентов, 886 фокальных приступов),
  но это носимая ЭЭГ за ухом (4 канала, не 10–20), поэтому для текущих моделей не подходит.

## Обходной путь для данных, которые нельзя скачать по HTTPS

Для TUH (rsync по SSH) и других наборов по соглашению: данные скачиваете вы на своём компьютере,
а сессия подключается к папке на нём через Remote Control (папка — корень git-репозитория; в
приложении Claude: «Настройки → Claude Code → Remote Control»). Тогда домены не нужны, а
пре-регистрация и анализ идут как раньше. Данные по соглашению не нужно загружать в репозиторий
или на Zenodo: лицензии TUH и BDSP запрещают распространение.
