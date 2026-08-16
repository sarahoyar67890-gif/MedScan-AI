# MedScan AI — AI-Assisted Skin Lesion Screening

A research/portfolio project that fine-tunes a ResNet18 on HAM10000 to produce
a **preliminary, explainable screening signal** from an uploaded skin-lesion
photo, with Grad-CAM visual explanations and a custom-designed Streamlit
interface.

> **This is not a medical device.** It does not diagnose disease. See
> [Responsible AI & Limitations](#responsible-ai--limitations).

---

## Table of contents

- [What this is and why](#what-this-is-and-why)
- [How the AI works](#how-the-ai-works)
- [Dataset: HAM10000](#dataset-ham10000)
- [Class mapping (read this before training)](#class-mapping-read-this-before-training)
- [Model architecture & transfer learning](#model-architecture--transfer-learning)
- [Image preprocessing](#image-preprocessing)
- [Handling class imbalance](#handling-class-imbalance)
- [Training](#training)
- [Evaluation](#evaluation)
- [Grad-CAM explainability](#grad-cam-explainability)
- [Application architecture (v2)](#application-architecture-v2)
- [The Streamlit application](#the-streamlit-application)
- [The FastAPI backend](#the-fastapi-backend)
- [Analysis history & dashboard](#analysis-history--dashboard)
- [Input validation & security](#input-validation--security)
- [Responsible AI & limitations](#responsible-ai--limitations)
- [Project structure](#project-structure)
- [Windows setup](#windows-setup)
- [Dataset setup](#dataset-setup)
- [How to train](#how-to-train)
- [How to evaluate](#how-to-evaluate)
- [How to launch the app](#how-to-launch-the-app)
- [Running the API backend](#running-the-api-backend)
- [Running tests](#running-tests)
- [Docker & deployment](#docker--deployment)
- [Environment variables](#environment-variables)
- [Limitations & future improvements](#limitations--future-improvements)

---

## What this is and why

Dermatologist access is limited in many places, and people often don't have
an easy way to get even a rough sense of whether a skin lesion looks
worth having checked. MedScan AI explores whether a transfer-learned CNN can
provide a **preliminary, explainable screening signal** — deliberately not a
diagnosis — to help someone decide whether to seek professional evaluation
sooner rather than later.

It's built as a portfolio project demonstrating a real, end-to-end computer
vision workflow: data preparation with leakage-free splitting, transfer
learning, class-imbalance handling, proper evaluation, model explainability
(Grad-CAM), and a deployed inference interface — the kind of pipeline used
in real applied ML/CV roles, applied honestly to a genuinely hard domain.

## How the AI works

```
Upload Image → Preprocess → ResNet18 (transfer learning) → Prediction
             → Confidence Score → Grad-CAM Explanation → Disclaimer
```

1. **Upload** — a JPG/PNG lesion photo.
2. **Preprocess** — resized and normalized identically to training (see
   `utils/common.py`, shared between training and inference so there's no
   train/inference skew).
3. **Analyze** — a ResNet18 pretrained on ImageNet, fine-tuned on HAM10000,
   extracts visual features.
4. **Predict** — a softmax over two classes produces a probability for
   `benign-pattern` and `suspicious-pattern`.
5. **Explain** — Grad-CAM shows which regions of the image drove the
   prediction.
6. **Next step** — the app consistently recommends professional evaluation
   for any lesion the user is concerned about, independent of the model's
   output.

The app never says "you have skin cancer" or "you do not have skin cancer."
It only ever reports a **screening-pattern prediction** and a **confidence
score**.

## Dataset: HAM10000

["Human Against Machine with 10000 training images"](https://doi.org/10.1038/sdata.2018.161)
(Tschandl, Rosendahl, Kittler — Medical University of Vienna) is a public
dataset of ~10,015 dermatoscopic images of pigmented skin lesions, labeled
into 7 diagnostic categories, collected from two sites over 20 years.

**Important limitation:** HAM10000 images are dermatoscopic — captured with
a specialized magnifying/polarized-light instrument that removes surface
glare and reveals sub-surface structure. Ordinary smartphone photos look
different (different lighting, focus, resolution, skin surface reflection).
A model trained purely on dermatoscopic images should **not** be assumed to
generalize equally well to smartphone photos, and this project makes no
such claim.

## Class mapping (read this before training)

HAM10000's native `dx` column has 7 classes:

| Code    | Meaning                                            | Screening group     |
|---------|-----------------------------------------------------|----------------------|
| `mel`   | Melanoma                                             | suspicious-pattern   |
| `bcc`   | Basal cell carcinoma                                 | suspicious-pattern   |
| `akiec` | Actinic keratoses / intraepithelial carcinoma        | suspicious-pattern   |
| `bkl`   | Benign keratosis-like lesions                        | benign-pattern       |
| `nv`    | Melanocytic nevi (moles)                             | benign-pattern       |
| `df`    | Dermatofibroma                                       | benign-pattern       |
| `vasc`  | Vascular lesions                                     | benign-pattern       |

MedScan AI trains a **binary** screening classifier, not the full 7-way
diagnosis. `akiec` is pre-malignant/early-malignant and is grouped on the
"suspicious" side deliberately — the mapping is a modeling simplification
made explicitly here (`config.py`), not a clinical standard, and errs
toward flagging the borderline category rather than hiding it. This is the
kind of judgment call worth being able to explain clearly in an interview.

## Model architecture & transfer learning

- Backbone: `torchvision.models.resnet18` pretrained on ImageNet.
- The final fully-connected layer is replaced with `Dropout(0.3) → Linear(512, 2)`.
- **Stage 1** — train only the new head, backbone fully frozen (fast, stabilizes
  the new layer before touching pretrained weights).
- **Stage 2** (optional) — unfreeze `layer3` + `layer4` + the head, fine-tune
  at a lower learning rate. Earlier layers (`conv1`, `layer1`, `layer2`) stay
  frozen, preserving generic low-level ImageNet features (edges, textures)
  while adapting higher-level features to skin-lesion imagery.
- Best checkpoint selection is by **validation F1**, not raw accuracy —
  accuracy is a misleading metric on an imbalanced medical dataset (see
  below).

## Image preprocessing

Training uses stochastic augmentation (`utils/common.py::get_train_transforms`):
resize → random crop → random horizontal/vertical flip → random rotation
(±20°) → mild color jitter → normalize (ImageNet mean/std).

Validation, test, **and live Streamlit inference** all use the same
deterministic transform (`get_eval_transforms`): resize → normalize. Using
one shared function for both training and inference is deliberate — it's a
common, easy-to-miss source of silent accuracy loss in deployed CV models.

## Handling class imbalance

HAM10000 is dominated by `nv` (benign moles) — roughly 67% of the dataset.
After binary mapping, benign-pattern still substantially outnumbers
suspicious-pattern. For a screening tool this matters a lot: a model that
just predicts "benign" for everything scores high accuracy while being
useless (and actively harmful, since it would miss the suspicious cases
it's meant to catch).

MedScan AI uses **two complementary strategies** (both on by default,
toggle in `config.py`):

1. **Weighted random sampling** (`WeightedRandomSampler`) — training batches
   are resampled so both classes appear roughly equally often, regardless of
   the raw dataset's skew.
2. **Class-weighted loss** (`nn.CrossEntropyLoss(weight=...)`) — inverse-
   frequency weights, computed directly from the training split, so
   misclassifying the minority (suspicious) class is penalized more.

Together these push the model to actually learn the minority class's
patterns rather than defaulting to the majority prediction.

## Training

`training/train.py` implements a full PyTorch training pipeline:

- Automatic CUDA/CPU device detection (`config.get_device()`).
- Reproducibility via a fixed seed across `random`, `numpy`, and `torch`
  (`utils/common.py::set_seed`), plus deterministic cuDNN settings.
- Two-stage training (head-only, then optional fine-tuning).
- `ReduceLROnPlateau` learning-rate scheduling on validation F1.
- Early stopping on validation F1 plateau (`training/engine.py::EarlyStopper`).
- Checkpointing: `checkpoints/medscan_resnet18_best.pt` (best val F1 so far)
  and `medscan_resnet18_last.pt` (most recent epoch), each storing model
  weights, optimizer state, epoch, stage, and validation metrics.
- Full training history (loss/accuracy/F1/recall/LR per epoch) saved to
  `outputs/training_history.json`.
- All paths built with `pathlib`, so this runs unmodified on Windows.

**No results are fabricated anywhere in this repo.** Until you train the
model on real HAM10000 data, `outputs/` is empty and the Streamlit app's
"Model information" panel will read "Not trained yet" / "Not yet evaluated."

## Evaluation

`evaluation/evaluate.py` runs the best checkpoint against the **held-out
test split** (never seen during training or model selection) and reports:

- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix (saved as a PNG)
- ROC curve (saved as a PNG)
- Full `sklearn` classification report (saved as text)
- All of the above written to `outputs/metrics.json`

**Why recall matters here:** in a screening context, a false negative (the
model says "benign-pattern" on something actually suspicious) is a worse
outcome than a false positive (flagging something benign as worth a second
look, which just costs an unnecessary doctor visit). That's part of why this
project selects its best checkpoint by F1 rather than raw accuracy, and why
recall is reported prominently rather than buried. That said: **no single
metric on a test set of a few thousand dermatoscopic images demonstrates
real clinical usefulness.** Metrics here describe model behavior on this
dataset's test split — nothing more.

If you haven't trained the model yet, running this script will tell you so
and exit cleanly rather than inventing numbers.

## Grad-CAM explainability

`explainability/gradcam.py` implements Grad-CAM (Selvaraju et al., 2017)
against ResNet18's `layer4` (the last convolutional block):

1. Forward-hook captures `layer4`'s activations.
2. Backward-hook captures the gradient of the predicted class's score with
   respect to those activations.
3. Gradients are global-average-pooled per channel to get importance
   weights, which weight-sum the activations and pass through ReLU — the
   standard Grad-CAM formula.
4. The resulting low-resolution heatmap is upsampled to the original image
   size and blended on top as a colored overlay.

The Streamlit app always shows the **original image** and **Grad-CAM
overlay** side by side, with an explicit caption: *"Grad-CAM highlights
image regions that influenced the model's prediction. It is an
interpretability aid and should not be interpreted as a medical
explanation."*

## Application architecture (v2)

The original version was a single Streamlit script that loaded the model and
ran inference inline. That's fine for a demo, but it means the model-loading
and prediction logic can't be reused by anything else, there's no history of
what's been analyzed, and there's no way to serve predictions to a
non-Streamlit client.

The current architecture separates those concerns:

```
                     ┌───────────────────────┐
                     │  inference/predictor.py │  ← one shared inference
                     │  inference/validation.py│    service + input validation
                     └───────────┬────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                 │
        ┌────────▼─────────┐            ┌──────────▼─────────┐
        │  app/app.py       │            │  backend/main.py    │
        │  (Streamlit UI,    │           │  (FastAPI REST API,  │
        │   in-process calls)│           │   for external/API   │
        └────────┬───────────┘           │   consumers)         │
                 │                       └──────────┬───────────┘
                 └───────────────┬────────────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │  storage/db.py          │  ← SQLite analysis
                     │  (outputs/*.db)         │    history + stats
                     └─────────────────────────┘
```

- **`inference/predictor.py`** — the single place that loads the checkpoint,
  runs preprocessing, calls the model, and generates the Grad-CAM overlay.
  Both the Streamlit app and the FastAPI backend import this module rather
  than duplicating model-loading logic, so there's exactly one code path
  that can go stale or drift.
- **`inference/validation.py`** — validates every uploaded file (size,
  extension, declared content-type, that it actually decodes as an image,
  sane dimensions) *before* it reaches the model. See
  [Input validation & security](#input-validation--security).
- **`app/app.py`** — the Streamlit UI. Calls the predictor in-process (no
  network hop), which keeps local/demo usage to a single process with no
  extra moving parts.
- **`backend/main.py`** — a separate, deployable FastAPI service wrapping
  the *same* predictor, for anything that isn't Streamlit (a future
  frontend, a mobile client, a third-party integration, automated testing).
- **`storage/db.py`** — a small SQLite store (no extra service to run) that
  both front-ends write to, powering the dashboard and history views.

This keeps the "keep it simple, don't add unnecessary complexity" principle
intact — SQLite over Postgres, in-process calls over a network hop for the
UI, one inference module instead of two copies of the same logic.

## The Streamlit application

`app/app.py` (+ `app/style.py`, `app/style_extras.py`) is a custom-designed
interface, not default Streamlit styling — a hero section with a signature
animated "scan" panel, a numbered process explainer, a styled upload flow, a
result card with confidence bars, a side-by-side Grad-CAM comparison, and a
clearly visible (not fine-print) medical disclaimer. Fonts: Space Grotesk
(display), IBM Plex Sans (body), IBM Plex Mono (technical/data labels).

It's organized into five tabs:

| Tab | What it does |
|---|---|
| **Screening** | The original single-image upload → analyze → result + Grad-CAM flow. |
| **Batch analysis** | Upload several images at once; each is validated and analyzed independently, with per-image results and a progress indicator. |
| **Dashboard** | Aggregate stats (total analyses, suspicious-pattern rate, average confidence, average inference time) computed from the local analysis history, plus the held-out test-set metrics from `outputs/metrics.json` when available. |
| **History** | A log of recent analyses (filename, prediction, confidence, timestamp), stored locally, with a clear-history action. |
| **About & model info** | Model architecture/checkpoint status (with a manual refresh button) and the full medical disclaimer. |

The app **gracefully degrades** when there's no trained checkpoint yet — it
still renders fully so you can review the design, with image analysis
disabled and a clear explanatory note (or, if a checkpoint exists but fails
to load, a distinct error state explaining that) instead of crashing.

## The FastAPI backend

`backend/main.py` exposes the same screening capability as a REST API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check for load balancers / Docker healthchecks. |
| `/model/info` | GET | Whether a checkpoint is loaded, its validation F1, device, any load error. |
| `/model/reload` | POST | Re-checks disk for a newly trained checkpoint without restarting the process. |
| `/predict` | POST | Upload an image (multipart), get back the prediction, probabilities, and a base64-encoded Grad-CAM overlay PNG. |
| `/stats` | GET | Aggregate analysis stats (same data the dashboard tab shows). |
| `/history` | GET | Recent analyses (`?limit=`, capped at 100). |

Interactive docs are auto-generated at `/docs` (Swagger UI) once the service
is running. See [Running the API backend](#running-the-api-backend).

## Analysis history & dashboard

Every analysis (from the Screening tab, Batch analysis tab, or the API's
`/predict` endpoint) is logged to a local SQLite database at
`outputs/medscan_history.db` — filename, prediction, confidence, probability
breakdown, inference time, and which checkpoint produced it. No images are
stored, only metadata. This powers the Dashboard and History tabs and the
`/stats` / `/history` API endpoints. History writes are best-effort: a
storage failure is logged but never breaks an analysis in progress.

## Input validation & security

Every upload — whether through Streamlit or the API — passes through
`inference/validation.py` before it touches the model:

- File size capped (10 MB by default).
- Extension allowlist (`.jpg`, `.jpeg`, `.png` only) and, on the API,
  declared content-type checked against an allowlist too.
- The file is verified as a genuinely decodable image (`PIL.Image.verify()`
  followed by a real decode) — corrupted or non-image files are rejected
  with a clear message rather than crashing the app.
- Decoded image dimensions are bounds-checked (32px–8000px) to reject
  degenerate or decompression-bomb-style inputs.

Other security-relevant choices:

- No API keys or secrets exist anywhere in this project — there's nothing to
  leak, and `.env.example` documents the only configuration that does exist
  (CORS origins, log level).
- The FastAPI backend's CORS policy is restrictive by default
  (`MEDSCAN_CORS_ORIGINS`, defaults to the local Streamlit origin only) and
  configurable via environment variable for real deployments.
- The backend only accepts `GET`/`POST` and only the documented routes —
  no arbitrary file-path or shell inputs are ever accepted from a request.
- SQLite history stores only prediction metadata, never the uploaded image
  bytes, minimizing what's retained about any given upload.

## Responsible AI & limitations

- This is an educational/research project, **not a medical device**, and has
  not been through any regulatory or clinical validation process.
- It never outputs a diagnosis — only a screening-pattern prediction and a
  confidence score.
- False positives and false negatives are both possible and expected.
- HAM10000 is dermatoscopic-image data; performance may differ meaningfully
  on ordinary smartphone photos, different skin tones, lighting, or lesion
  types underrepresented in the dataset.
- The binary class mapping (see above) is a project-specific simplification,
  not a clinical grouping standard.
- **If you or someone you know has a lesion that's a concern, see a
  dermatologist or physician — regardless of anything this tool outputs.**

---

## Project structure

```
MedScan_AI/
├── app/
│   ├── app.py                  # Streamlit application (5-tab UI)
│   ├── style.py                 # core CSS design system
│   └── style_extras.py          # v2 additions: tabs, dashboard, states, history
├── backend/
│   ├── main.py                  # FastAPI REST API
│   └── schemas.py                # pydantic request/response models
├── inference/
│   ├── predictor.py              # shared model-loading + prediction service
│   └── validation.py             # upload validation (size/type/dimensions)
├── storage/
│   └── db.py                     # SQLite analysis history + stats
├── config.py                     # paths, class mapping, hyperparameters
├── data/
│   ├── prepare_data.py           # builds leakage-free lesion-grouped splits
│   ├── dataset.py                 # PyTorch Dataset + class-weight helpers
│   ├── raw/                       # put HAM10000 files here (gitignored)
│   └── processed/                 # generated train/val/test CSVs (gitignored)
├── models/
│   └── resnet_model.py            # ResNet18 builder, freeze/unfreeze helpers
├── training/
│   ├── engine.py                   # train/val epoch loop, early stopping
│   └── train.py                    # two-stage training script
├── evaluation/
│   └── evaluate.py                 # test-set metrics, confusion matrix, ROC
├── explainability/
│   └── gradcam.py                   # Grad-CAM implementation
├── utils/
│   └── common.py                    # seeding, shared train/eval transforms
├── tests/                            # pytest suite (see Running tests)
├── checkpoints/                       # trained weights land here (gitignored)
├── outputs/                            # metrics, plots, history db (gitignored)
├── requirements.txt                     # runtime dependencies
├── requirements-dev.txt                  # + pytest/httpx for testing
├── .env.example                          # documented config knobs (no secrets)
├── Dockerfile.backend / Dockerfile.frontend
└── docker-compose.yml                     # runs both services together
```

## Windows setup

```powershell
# From the project root, in PowerShell or cmd:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Everything in this project uses `pathlib` and cross-platform commands — no
hard-coded Linux paths.

## Dataset setup

1. Download HAM10000 (e.g. from the
   [ISIC archive](https://api.isic-archive.com/collections/212/) or
   [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)).
2. Place the files like this:

```
data/raw/HAM10000_metadata.csv
data/raw/HAM10000_images_part_1/*.jpg
data/raw/HAM10000_images_part_2/*.jpg
```

(If your download merges both image parts into one folder, that's fine —
`prepare_data.py` searches every subfolder under `data/raw/`.)

3. Build the splits:

```powershell
python -m data.prepare_data
```

This prints the class distribution, computes the imbalance ratio, and
writes `data/processed/train.csv`, `val.csv`, `test.csv` — split by
**lesion_id**, not by image, so no lesion's images leak across splits.

## How to train

```powershell
# Full two-stage training (head, then fine-tune)
python -m training.train

# Head-only (faster on CPU, skip fine-tuning)
python -m training.train --skip-finetune

# Override epoch counts
python -m training.train --head-epochs 5 --finetune-epochs 10
```

Progress, metrics, and early-stopping decisions print per epoch. Best
checkpoint is written to `checkpoints/medscan_resnet18_best.pt`.

## How to evaluate

```powershell
python -m evaluation.evaluate
```

Writes `outputs/confusion_matrix.png`, `outputs/roc_curve.png`,
`outputs/classification_report.txt`, and `outputs/metrics.json` from the
held-out test split.

## How to launch the app

```powershell
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Works with no trained checkpoint (design
review mode, analysis disabled) or with a trained one (full inference +
Grad-CAM).

## Running the API backend

The FastAPI service is independent of Streamlit — run it if you want to hit
the model from `curl`, a script, Postman, or any other client:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl http://localhost:8000/health
curl -F "file=@lesion.jpg" http://localhost:8000/predict
```

If no checkpoint is trained yet, `/predict` returns `503` with a clear
message rather than a stack trace; invalid uploads return `400` with the
specific validation failure.

## Running tests

```powershell
pip install -r requirements-dev.txt
pytest
```

The suite covers: image-validation edge cases (oversized, corrupted, wrong
type, too small), model construction and freeze/unfreeze staging, Grad-CAM
output shape and value range, the predictor's "not ready" path when no
checkpoint exists, the SQLite history store, dataset class-weight
computation, and the FastAPI endpoints (health, model info, predict
success/failure paths, stats, history) via `TestClient`. Tests don't require
a trained checkpoint — they exercise the "no checkpoint" and validation
paths directly, since that's the state a fresh clone of this repo starts in.

## Docker & deployment

Two services, one for each front-end, sharing the `checkpoints/` and
`outputs/` directories as volumes so they see the same model and history:

```bash
cp .env.example .env    # adjust if needed — defaults work for local use
docker compose up --build
```

- Streamlit UI: `http://localhost:8501`
- API + docs: `http://localhost:8000` / `http://localhost:8000/docs`

Both images are `python:3.11-slim` based, install only `requirements.txt`,
and include a `HEALTHCHECK`. Note that **training is not meant to run
inside these containers** — train locally (or in a separate GPU
environment) and mount the resulting `checkpoints/` directory in, which is
exactly what the compose file's volumes do.

To run just the backend or just the frontend:

```bash
docker compose up backend
docker compose up frontend
```

## Environment variables

See `.env.example` for the full list. Nothing in this project requires an
API key — the only knobs are the backend's CORS allowlist and log level.

| Variable | Default | Purpose |
|---|---|---|
| `MEDSCAN_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated origins allowed to call the API |
| `LOG_LEVEL` | `INFO` | Backend logging verbosity |
| `MEDSCAN_API_BASE_URL` | `http://localhost:8000` | Reserved for a future split-service frontend config; unused by the current in-process Streamlit app |

## Limitations & future improvements

Being direct about what this project is and isn't, on top of the modeling
limitations already covered above:

- **No authentication.** The API and app are open by default — fine for a
  local/portfolio demo, not for a public deployment with real patient data.
  Adding auth (e.g. API keys or OAuth) would be the first change before any
  real-world exposure.
- **SQLite, not a production database.** Appropriate at this scale; would
  need to move to Postgres/managed storage under real concurrent load.
- **No rate limiting.** Would matter before exposing `/predict` publicly.
- **No image data is retained**, which is good for privacy but also means
  there's no way to re-run or audit a specific past prediction against its
  original image.
- **No calibration study.** The confidence score is the model's softmax
  output, not a formally calibrated probability — a project extension would
  add temperature scaling and a reliability diagram against the test set.
- **Single-model, single-architecture.** An ensemble or a second backbone
  for comparison would strengthen the "portfolio" story further but wasn't
  added here to avoid complexity that doesn't serve the core project.
