# Datasets: what the decision model learns from, and where data can legally come from

pankhllm's decision model learns the **shape** of a question, not the numbers behind it.
Before training, every question is abstracted: codes become `<code>`, months `<date>`,
numbers `<num>`, quoted text `<text>`. So "how is T-112 doing vs peers in 2025-07" and
"how is D-9 doing vs peers in 2024-01" are one training example. Two consequences:

- **Training data is questions, labelled with the operation that answers them.** KPI values,
  prescriber records and sales data are not needed to train routing.
- **Public KPI data is still useful** for vocabulary (drug names, specialties, geographies)
  and for a realistic demo executor. It is not what makes the model accurate.

## What we built (pharma KPI reporting)

The catalog is `evals/pharma-kpi/router.yaml`: seven operations plus `UNSUPPORTED` for
questions none of them answers. Product names are fictional.

| Operation | Answers |
|---|---|
| `kpi_value` | the value of one KPI for a geography, product, period |
| `kpi_rank` | top or bottom N territories, districts, regions or prescribers by a KPI |
| `kpi_vs_benchmark` | "how am I doing versus my peers, region, nation" |
| `kpi_change` | change versus prior period or prior year |
| `kpi_trend` | trend over the last N weeks, months or quarters |
| `prescriber_list` | prescribers in a geography by segment and specialty |
| `coverage_owner` | who covers a territory, district or region |

KPIs covered: TRx, NRx, NBRx, calls, reach, frequency, market share, samples, writers,
new writers, each with the phrasings people actually type ("total scripts", "HCP calls",
"new-to-brand"). Geographies are territory, district and region codes (`T-123`, `D-12`, `R-3`).

### Three ways we produced labelled questions

1. **Template generator** (`evals/pharma-kpi/gen.py`). 106 phrasing templates across the
   seven operations and the reject class, 26 metric phrasings, random entities.
   One third of the templates per operation are held out, so test phrasings are ones the
   model never saw.

   | Size of the question space | |
   |---|---|
   | Distinct question shapes | about 22,000 |
   | Distinct questions with entity values | over 89 million |
   | Training set used | 20,000 questions from training templates |
   | Test set used | 2,000 questions from held-out templates |

2. **Teacher-written questions** (`pankhllm augment`). A generative model writes many
   phrasings per operation, plus questions none of them should answer. The teacher then
   labels them (`pankhllm label`), and plans that fail catalog validation are dropped.
   First run with a local 12B model: 242 generated, 82 kept with valid labels. Second run: 392 generated, 367 kept. The generator
   prompt originally showed parameter regexes and the model copied them literally
   ("TDR-102"); it now shows parameter descriptions instead.

3. **Production traffic.** With a trace store configured, every generative-planner decision
   and every agent tool choice is recorded as a label, shape only. `pankhllm mine --apply`
   retrains nightly.

Skill files add a fourth source for skill routing: `examples:` in a skill's front matter.

### Results

See [BENCHMARK-DECISIONS.md](BENCHMARK-DECISIONS.md#pharma-kpi-catalog-at-scale) for the full
table and the steps that got there. Summary on held-out questions:

| Test set (held out) | Questions | Decided without a model | Precision of those | Decision p50 |
|---|---|---|---|---|
| Templates | 2,000 | 63.3% | 100% | 0.21 ms |
| Teacher-written A | 82 | 67.1% | 100% | 0.36 ms |
| Teacher-written B, held-out half | 184 | 61.4% | 96.5% | 0.23 ms |

Trained on 20,000 template questions plus 183 teacher-labelled ones.
Undecided questions go to the generative planner, exactly as before training.

## Public data sources (checked 26 Sep 2026)

Every URL below was opened and its license or terms read on that date. Terms change;
check again before use. This is not legal advice.

### KPI-like value data

| Dataset | Contents | Granularity | License | Commercial training use |
|---|---|---|---|---|
| [CMS Part D Prescribers](https://data.cms.gov/) (by Provider & Drug, by Provider, by Geography & Drug) | prescriber NPI, specialty, drug, claims, 30-day fills, day supply, cost, beneficiaries | annual, 2013 to 2024; claims include refills (closer to TRx), no NRx | [US government works](https://www.usa.gov/government-copyright); do not imply endorsement | Yes |
| [CMS Open Payments](https://openpaymentsdata.cms.gov/) | industry payments to prescribers: manufacturer, amount, nature, drug | per payment, program years 2019 to 2025 | US government works | Yes |
| [NPPES NPI file](https://download.cms.gov/nppes/NPI_Files.html) | every NPI: address, taxonomy (specialty) | monthly and weekly | FOIA-disclosable, no explicit license | With care: it describes individuals |
| [Medicaid State Drug Utilization Data](https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data) | state by NDC by quarter: units, prescriptions, reimbursement | quarterly, latest 2026 Q1 | [Public domain](https://www.usa.gov/publicdomain/label/1.0/) | Yes |
| [FDA NDC Directory](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) | brand and generic names, labeler, form, route, pharmacologic class | daily | public domain ([FDA policy](https://www.fda.gov/about-fda/about-website/website-policies)) | Yes |
| [Drugs@FDA](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files) | applications, sponsors, products, marketing status | weekdays | public domain; openFDA is [CC0](https://open.fda.gov/license/) | Yes |
| Census ZCTA shapes (2025 vintage) | ZIP-code tabulation area polygons | annual | free to reproduce, citation requested; "TIGER/Line" is a trademark, keep it out of product names | Yes |
| [HUD-USPS ZIP crosswalk](https://www.huduser.gov/portal/datasets/usps_crosswalk.html) | ZIP to tract, county, CBSA | quarterly | registration required; API terms not reviewed | With conditions |
| [NHS England Prescribing Data](https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code) | practice by item by month: items, quantity, cost | monthly, latest Jul 2026 | Open Government Licence v3, attribution | Yes |

None of the public US data has NRx, weekly grain, or sales territories. Territory rollups
would have to be built from NPI and ZIP.

### Labelled question datasets (for phrasing variety and rejection)

| Dataset | Size | License | Commercial use | Use here |
|---|---|---|---|---|
| [CLINC150](https://github.com/clinc/oos-eval) | 22,500 in-scope, 1,200 out-of-scope | CC BY 3.0 | Yes, attribution | its out-of-scope questions are good `UNSUPPORTED` examples |
| [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) | 13,083, 77 intents | CC BY 4.0 | Yes, attribution | closest public analogue to fine-grained KPI intents |
| [MASSIVE](https://github.com/alexa/massive) | over 1M utterances, 52 languages | CC BY 4.0 | Yes, attribution | intent plus slot format, multilingual |
| [SNIPS](https://github.com/sonos/nlu-benchmark) | about 14,000, 7 intents | CC0 | Yes | slot filling |
| [Spider](https://yale-lily.github.io/spider) | 10,181 questions, 138 domains | CC BY-SA 4.0 | Yes, attribution and share-alike | cross-domain analytic phrasing |
| [BIRD](https://bird-bench.github.io/) | 12,751 questions | CC BY-SA 4.0 | Yes, attribution and share-alike | business-style metric questions |
| WikiSQL | 80,654 | code BSD-3; no separate data license stated | Unverified | not recommended |
| ATIS | 7,300+ | LDC non-commercial agreement | No, without LDC membership | do not use |

Whether a trained model is an "adaptation" under CC BY-SA share-alike terms is legally
unsettled. Ask counsel before shipping a model trained on Spider or BIRD.

### Not public

- **IQVIA Xponent** is licensed prescriber-level data with no public download. IQVIA's
  [third-party access program](https://www.iqvia.com/about-us/third-party-access-program)
  grants vendors limited use for the client's sole benefit, confidentiality, and return or
  destruction at project end. Do not train on it without a license that allows it.
- **Symphony Health** (HealthVerity) data is offered through licensed partnerships only.
- **Individual manufacturers** rarely publish datasets. Their products do appear in the public
  sources above: Open Payments lists each company's payments, and Medicaid utilization lists
  each product by NDC.

## How to use public data with pankhllm

- **Vocabulary.** Load brand and generic names from the NDC Directory into the `product`
  enum and its aliases, specialties from NPPES taxonomy codes into `specialty`, states and
  ZIPs into geography parameters. The slot filler then recognises real names in any casing.
- **Rejection.** Add CLINC150's out-of-scope questions (with attribution) as
  `planner.unsupported_examples`, so the model learns what not to answer.
- **Phrasing variety.** Rewrite Banking77 or MASSIVE-style phrasings into your KPI domain
  with the template generator or `pankhllm augment`, then label with the teacher.
- **A realistic demo executor.** Part D by Provider & Drug answers "how am I doing versus my
  peers" at prescriber, state and specialty level on real public numbers.
- **Your own data** is the best source: questions from your application logs, labelled by
  the teacher once, never leave your machine when the teacher is local.
