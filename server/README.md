# Voice Dataset Uploader Backend

## 1) Create and activate virtualenv

On Windows CMD:

```bat
cd server
python -m venv .venv
.venv\Scripts\activate
```

## 2) Install dependencies

```bat
pip install -r requirements.txt
```

## 3) Configure environment

Create a `.env` file in `server/` from `.env.example` and set:

- `HF_TOKEN`: Hugging Face token with write access.
- `HF_DATASET_REPO_ID`: Single public dataset repo id.
- `ALLOWED_ORIGINS`: `*` for quick testing, or comma-separated frontend origins.

## 4) Run API

```bat
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

API endpoints:

- `GET /api/health`
- `POST /api/tools/voice-dataset/submit`
