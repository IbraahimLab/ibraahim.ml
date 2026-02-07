import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


@dataclass
class UploadResult:
  dataset_repo: str
  sample_id: str
  audio_path: str
  jsonl_path: str
  commit_url: Optional[str]


class HuggingFaceDatasetWriter:
  def __init__(self, token: str, repo_id: str, private: bool = False) -> None:
    self._token = token
    self.repo_id = repo_id
    self._api = HfApi(token=token)
    self._api.create_repo(
      repo_id=repo_id,
      repo_type="dataset",
      private=private,
      exist_ok=True,
    )

  def _strip_front_matter(self, markdown: str) -> str:
    if not markdown.startswith("---\n"):
      return markdown
    marker = "\n---\n"
    end = markdown.find(marker, 4)
    if end == -1:
      return markdown
    return markdown[end + len(marker):].lstrip("\n")

  def _read_repo_text_file(self, path_in_repo: str) -> str:
    try:
      local_path = hf_hub_download(
        repo_id=self.repo_id,
        repo_type="dataset",
        filename=path_in_repo,
        token=self._token,
      )
    except EntryNotFoundError:
      return ""

    with open(local_path, "r", encoding="utf-8") as handle:
      return handle.read()

  def _build_train_jsonl_from_sample_files(self, repo_files: set[str]) -> str:
    lines: list[str] = []
    sample_jsonl_files = sorted(
      path for path in repo_files
      if path.startswith("data/jsonl/") and path.endswith(".jsonl")
    )
    for path in sample_jsonl_files:
      content = self._read_repo_text_file(path)
      if content:
        normalized = self._normalize_jsonl_content(content)
        lines.append(normalized if normalized.endswith("\n") else f"{normalized}\n")
    return "".join(lines)

  def _audio_resolve_url(self, path_in_repo: str) -> str:
    # Use a Hub resolve URL so the dataset viewer can stream audio files.
    encoded_path = quote(path_in_repo, safe="/")
    return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{encoded_path}"

  def _normalize_row_record(self, record: dict) -> dict:
    row = dict(record)
    row.pop("created_at", None)
    row.pop("speaker_id", None)
    row.pop("source", None)
    audio_value = row.get("audio")
    if isinstance(audio_value, str) and not audio_value.startswith(("http://", "https://")):
      row["audio"] = self._audio_resolve_url(audio_value)
    return row

  def _normalize_jsonl_content(self, content: str) -> str:
    out_lines: list[str] = []
    for raw_line in content.splitlines():
      line = raw_line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        # Keep malformed lines untouched to avoid data loss.
        out_lines.append(raw_line)
        continue
      normalized = self._normalize_row_record(record)
      out_lines.append(json.dumps(normalized, ensure_ascii=False))
    return "\n".join(out_lines) + ("\n" if out_lines else "")

  def _default_dataset_card(self) -> str:
    return (
      "---\n"
      "configs:\n"
      "- config_name: default\n"
      "  data_files:\n"
      "  - split: train\n"
      "    path: data/train.jsonl\n"
      "dataset_info:\n"
      "  features:\n"
      "  - name: id\n"
      "    dtype: string\n"
      "  - name: audio\n"
      "    dtype: audio\n"
      "  - name: transcript\n"
      "    dtype: string\n"
      "  - name: language\n"
      "    dtype: string\n"
      "  - name: duration_sec\n"
      "    dtype: float32\n"
      "---\n\n"
      "# Voice Dataset\n\n"
      "Collected from the web uploader tool.\n"
    )

  def _ensure_readme_has_viewer_config(self, existing_readme: str) -> str:
    if (
      "configs:" in existing_readme
      and "data/train.jsonl" in existing_readme
      and "- name: speaker_id" not in existing_readme
      and "- name: created_at" not in existing_readme
      and "- name: source" not in existing_readme
    ):
      return existing_readme

    if not existing_readme.strip():
      return self._default_dataset_card()

    body = self._strip_front_matter(existing_readme)
    if not body.strip():
      return self._default_dataset_card()
    return f"{self._default_dataset_card()}\n\n{body}"

  def upload_sample(
    self,
    sample_id: str,
    audio_bytes: bytes,
    transcript: str,
    language: str,
    duration_sec: float,
  ) -> UploadResult:
    now = datetime.now(timezone.utc)
    ym = now.strftime("%Y/%m")

    audio_path = f"data/audio/{ym}/{sample_id}.wav"
    jsonl_path = f"data/jsonl/{ym}/{sample_id}.jsonl"
    train_jsonl_path = "data/train.jsonl"

    record = {
      "id": sample_id,
      "audio": self._audio_resolve_url(audio_path),
      "transcript": transcript,
      "language": language,
      "duration_sec": round(duration_sec, 2),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    jsonl_bytes = line.encode("utf-8")
    repo_files = set(self._api.list_repo_files(repo_id=self.repo_id, repo_type="dataset"))
    if train_jsonl_path in repo_files:
      existing_train = self._read_repo_text_file(train_jsonl_path)
      train_jsonl = self._normalize_jsonl_content(existing_train) + line
    else:
      # Backfill from previously uploaded per-sample JSONL files so the viewer can render historical rows.
      train_jsonl = self._build_train_jsonl_from_sample_files(repo_files) + line
    existing_readme = self._read_repo_text_file("README.md") if "README.md" in repo_files else ""
    next_readme = self._ensure_readme_has_viewer_config(existing_readme)

    operations = [
      CommitOperationAdd(path_in_repo=audio_path, path_or_fileobj=io.BytesIO(audio_bytes)),
      CommitOperationAdd(path_in_repo=jsonl_path, path_or_fileobj=io.BytesIO(jsonl_bytes)),
      CommitOperationAdd(path_in_repo=train_jsonl_path, path_or_fileobj=io.BytesIO(train_jsonl.encode("utf-8"))),
      CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(next_readme.encode("utf-8"))),
    ]

    commit_info = self._api.create_commit(
      repo_id=self.repo_id,
      repo_type="dataset",
      operations=operations,
      commit_message=f"Add voice sample {sample_id}",
    )
    commit_url = getattr(commit_info, "commit_url", None)

    return UploadResult(
      dataset_repo=self.repo_id,
      sample_id=sample_id,
      audio_path=audio_path,
      jsonl_path=jsonl_path,
      commit_url=commit_url,
    )
