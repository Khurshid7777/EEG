# Data access request text (English, universal)

Fill the `[brackets]` before sending. Do not claim an affiliation, ethics approval or training you do not have; if a form asks for one, say so honestly (an independent researcher is accepted by most of these resources, though some require an institutional supervisor or reference).

## 1. Full version (project description / "purpose of use" / "research plan", ~250 words)

**Project.** NeuroLens is an open-source, non-commercial research pipeline for decision support in clinical EEG: routine-EEG screening (normal/abnormal), event detection (spikes, ictal patterns, burst suppression, periodic patterns), automatic sleep staging, and a context-aware, rule-based interpretation layer. It is research software and is not used for patient care or clinical decisions.

**Why we need the data.** We evaluate the pipeline strictly *externally*. Every model is trained and frozen on other, open data first; we then run it once on the requested dataset. We do not tune thresholds, features or models on your data, and we never train on the held-out evaluation split.

**Method.** Each evaluation is pre-registered in a public, time-stamped document *before* the data are opened: hypotheses, metrics, decision rules and acceptance thresholds. Deviations are disclosed in the report. Negative and inconclusive results are published with the same prominence as positive ones; to date several of our candidate improvements were rejected under these rules and reported as such.

**Data handling.** Data are processed locally for the stated analysis only. We will not attempt re-identification or linkage, will not redistribute the data or derivatives that allow reconstruction of recordings, will not share access with third parties, and will delete the data on request. Only aggregate statistics and per-record scores without identifiers are published.

**Outputs.** Open-source code, pre-registrations, evaluation reports and aggregate results; we will cite the dataset and its authors and will share the findings with the data providers.

**Contact:** [name], [email], [affiliation or "independent researcher"]. Repository: [link to khurshid7777/eeg or its public mirror].

## 2. Short version (≤ 500 characters, for single-line fields)

Independent external validation of an open-source, non-commercial EEG decision-support pipeline (NeuroLens). Models are frozen beforehand; the evaluation is pre-registered and run once on your data, with no tuning on it. No re-identification, no redistribution, local processing, results published (including negative ones). Research use only, no clinical use.

## 3. One-line add-on per source (append to the text above)

* **NMT-4K (Zenodo):** "We will use only the held-out evaluation split (1,000 recordings) to test record-level normal/abnormal classification, expert-labelled event detectors (spike/sharp wave, slowing, triphasic, burst suppression) and per-hospital transfer."
* **TUH EEG / TUAB (NEDC):** "We will evaluate the frozen record-level normal/abnormal model on the TUAB evaluation split to measure transfer to another hospital, device and population."
* **NCH Sleep DataBank / CPS (PhysioNet):** "We will test whether a sleep stager trained on five centres generalises to your paediatric/clinical polysomnography, using a pre-registered, paired comparison against the current model; no training on your data."
* **jsmf-access (PhysioNet):** "We will test, on derived frontal-EEG measures under general anaesthesia, whether a context-conditioned rule (sedatives present → slowing read as sedation, not encephalopathy) agrees with documented anaesthetic state; analysis is pre-registered and descriptive."
* **BDSP / IIIC (SPaRCNet), HMS:** "We will evaluate frozen detectors of periodic and rhythmic patterns (LPD, GPD, LRDA, GRDA) against expert-vote labels; no training on the evaluation partition."
* **Kaggle HMS:** (competition rules) use the data only as permitted by the rules; no redistribution.

## 4. If a form asks for…

* **"Funding":** none (self-funded, non-commercial).
* **"IRB / ethics":** "Secondary analysis of de-identified, publicly distributed or access-controlled data; no interaction with subjects. [Add your institution's exemption/waiver if you have one; otherwise state that none exists.]"
* **"Reference / supervisor":** [name and contact of a clinician or researcher who can vouch for the project, if you have one — this materially raises approval chances for PhysioNet credentialing.]
* **"Intended publications":** open reports in the project repository; a manuscript or preprint summarising the external validations is planned.
