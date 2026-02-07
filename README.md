
# ibraahim.ml

Portfolio frontend (React + Vite) with a custom `Tools` page.

## What was added

- `/tools` route now points to a dedicated `Tools` page.
- Tool #1: `Voice Dataset Uploader`
- Flow: record/upload audio -> manual transcript -> submit to backend.
- Backend validates max duration (`<= 30s`) and uploads:
  - WAV file to Hugging Face dataset repo
  - One-line JSONL metadata file for the same sample

## Frontend setup

```bat
npm install
copy .env.example .env
npm run dev
```

`VITE_API_BASE_URL` can be:

- local: `http://localhost:8000`
- deployed: your public backend URL
- empty: use same-origin `/api`
- dev proxy target (when using same-origin): `VITE_API_PROXY_TARGET=http://127.0.0.1:8000`
- optional dev override: `VITE_USE_ABSOLUTE_API_IN_DEV=true`

## Backend setup

```bat
cd server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

Required backend env vars:

- `HF_TOKEN`: Hugging Face token with dataset write access.
- `HF_DATASET_REPO_ID`: Single public dataset repo id, for example `IbraahimLab/voice-dataset`.
- `ALLOWED_ORIGINS`: `*` for quick testing, or comma-separated frontend origins in production.

## API

- `GET /api/health`
- `POST /api/tools/voice-dataset/submit`

`POST` payload (`multipart/form-data`):

- `audio` (WAV, required)
- `transcript` (manual transcript, required)
- `language` (optional, default `en`)

## Notes

- Browser side normalizes recorded/uploaded audio to mono 16kHz WAV before upload.
- Backend enforces duration limit again for safety.
- Backend writes both:
  - per-sample JSONL in `data/jsonl/...`
  - global `data/train.jsonl` (for Hub viewer preview)
- First upload after this patch auto-backfills `data/train.jsonl` from existing per-sample JSONL files.

## Troubleshooting

- Frontend shows `Failed to fetch`:
  - Ensure backend is running.
  - Check `VITE_API_BASE_URL` is reachable from the browser.
  - Avoid HTTPS frontend -> HTTP backend mismatch.
  - Set `ALLOWED_ORIGINS` to include your frontend origin (or `*` while testing), then restart backend.
- Hugging Face dataset card only shows files:
  - Submit one new sample after this update to generate/update `README.md` + `data/train.jsonl`.
  - Dataset viewer can take a short time to refresh.
