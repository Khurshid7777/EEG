"""FastAPI application (TZ §12, §14).

Thin service over the pipeline and real-time monitor. Endpoints:
  GET  /health                 liveness + version
  GET  /version                software + rule-base versions, available schemas
  GET  /config/montages        available montages
  GET  /schemas/{name}         a generated JSON Schema
  POST /analyze                analyze an EDF by server-side path (JSON body)
  POST /analyze/upload         analyze an uploaded recording (multipart; EDF/BDF, or a .zip with
                               BrainVision / EEGLAB / WFDB files)
  POST /check/upload           intake check of an uploaded recording (recording regulation)
  POST /monitor                real-time cEEG monitor on an EDF by path (JSON body)
  POST /feedback               capture a neurophysiologist correction (TZ §13)

Results are deidentified (§12) and every request is audit-logged.

Security (D86): every endpoint except /health requires an API key (``X-API-Key``, keys from
``api_keys`` or NEUROLENS_API_KEYS, comma-separated); without keys the API refuses requests unless
it was explicitly created with ``allow_no_auth`` (local use). Path-based endpoints read only inside
``data_root`` (NEUROLENS_DATA_ROOT) and are disabled without it. Uploads are read in chunks up to
``max_upload_bytes`` (NEUROLENS_MAX_UPLOAD_MB, default 1024).
"""

from __future__ import annotations

import hmac
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import __version__
from ..contracts.signal import PatientInfo
from ..feedback.log import Correction, FeedbackLogger
from ..pipeline.config_loader import load_configs
from .audit import AuditLogger
from .models import AnalyzeParams, AnalyzeRequest, FeedbackRequest, MonitorRequest
from .service import run_analyze, run_monitor
from .uploads import UploadError, materialize

_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

DISCLAIMER = (
    "Decision-support tool. Not a medical diagnosis. Final interpretation is the "
    "responsibility of a qualified neurophysiologist/intensivist."
)


OPEN_PATHS = {"/health"}
_SCHEMA_NAME = re.compile(r"^[a-z0-9_]+$")
_CHUNK = 1 << 20


def _keys(api_keys) -> list[str]:
    if api_keys is None:
        api_keys = [k.strip() for k in os.environ.get("NEUROLENS_API_KEYS", "").split(",")]
    return [k for k in api_keys if k]


async def read_capped(file: UploadFile, limit: int) -> bytes:
    """The whole upload, refusing it (413) as soon as it exceeds ``limit`` bytes."""
    parts, n = [], 0
    while chunk := await file.read(_CHUNK):
        n += len(chunk)
        if n > limit:
            raise HTTPException(status_code=413, detail=f"upload larger than {limit >> 20} MB")
        parts.append(chunk)
    return b"".join(parts)


def create_app(config=None, audit_path=None, feedback_path=None, *, api_keys: list[str] | None = None,
               allow_no_auth: bool | None = None, data_root: str | Path | None = None,
               max_upload_bytes: int | None = None) -> FastAPI:
    cfg = config or load_configs()
    keys = _keys(api_keys)
    if allow_no_auth is None:
        allow_no_auth = os.environ.get("NEUROLENS_ALLOW_NO_AUTH") == "1"
    root = data_root if data_root is not None else os.environ.get("NEUROLENS_DATA_ROOT")
    root = Path(root).resolve() if root else None
    limit = max_upload_bytes or int(os.environ.get("NEUROLENS_MAX_UPLOAD_MB", "1024")) << 20
    audit = AuditLogger(audit_path)
    feedback = FeedbackLogger(feedback_path or "feedback.jsonl")

    app = FastAPI(
        title="NeuroLens API",
        version=__version__,
        description="EEG detection & interpretation service (decision-support). " + DISCLAIMER,
    )

    @app.middleware("http")
    async def require_key(request: Request, call_next):
        if request.url.path in OPEN_PATHS or (allow_no_auth and not keys):
            return await call_next(request)
        if not keys:
            return JSONResponse({"detail": "API keys are not configured (NEUROLENS_API_KEYS)"}, status_code=401)
        given = request.headers.get("x-api-key", "")
        if not any(hmac.compare_digest(given.encode(), k.encode()) for k in keys):
            return JSONResponse({"detail": "missing or invalid X-API-Key"}, status_code=401)
        return await call_next(request)

    def server_path(p: str) -> Path:
        """A path the client names, confined to the data root (path endpoints are off without it)."""
        if root is None:
            raise HTTPException(status_code=403, detail="path-based analysis is disabled (NEUROLENS_DATA_ROOT "
                                                        "not set); use the /upload endpoints")
        path = (root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=403, detail="path outside the data root")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="file not found in the data root")
        return path

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/version")
    def version() -> dict:
        return {
            "neurolens_version": __version__,
            "rule_base_version": cfg.rule_base.version,
            "schemas": sorted(p.stem.replace(".schema", "") for p in _SCHEMAS_DIR.glob("*.schema.json")),
            "disclaimer": DISCLAIMER,
        }

    @app.get("/config/montages")
    def montages() -> list[dict]:
        return [{"name": m.name, "type": m.type.value} for m in cfg.montages.values()]

    @app.get("/schemas/{name}")
    def get_schema(name: str) -> dict:
        if not _SCHEMA_NAME.match(name):
            raise HTTPException(status_code=404, detail="unknown schema")
        path = _SCHEMAS_DIR / f"{name}.schema.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"unknown schema {name!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        path = server_path(req.path)
        try:
            return run_analyze(path, req, cfg, audit)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/analyze/upload")
    async def analyze_upload(
        file: UploadFile = File(...),
        mode: str = Form("A"),
        montage: str = Form("double_banana"),
        provider: str = Form("deterministic"),
        age_years: Optional[float] = Form(None),
        postmenstrual_age_weeks: Optional[float] = Form(None),
        clinical_question: Optional[str] = Form(None),
        recording_setting: Optional[str] = Form(None),
        prior_record_summary: Optional[str] = Form(None),
    ) -> dict:
        """``prior_record_summary``: the ``record_summary`` of a previous response for the same
        patient (JSON) - adds a serial comparison (D76)."""
        try:
            prior = json.loads(prior_record_summary) if prior_record_summary else None
            if prior is not None and not isinstance(prior, dict):
                raise ValueError("prior_record_summary must be a JSON object")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"prior_record_summary: {exc}")
        try:
            params = AnalyzeParams(
                mode=mode, montage=montage, provider=provider,
                age_years=age_years, postmenstrual_age_weeks=postmenstrual_age_weeks,
                clinical_question=clinical_question, recording_setting=recording_setting,
            )
        except ValueError as exc:  # pydantic ValidationError: invalid form value -> 422, not 500
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            tmp, tmpdir = materialize(await read_capped(file, limit), file.filename)
        except UploadError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            return run_analyze(tmp, params, cfg, audit, prior_summary=prior)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @app.post("/check/upload")
    async def check_upload(
        file: UploadFile = File(...),
        age_years: Optional[float] = Form(None),
        postmenstrual_age_weeks: Optional[float] = Form(None),
    ) -> dict:
        """Intake check of a recording against docs/recording_regulation.md (no analysis)."""
        from ..layer1_ingest.file_check import check_recording

        try:
            patient = PatientInfo(age_years=age_years, postmenstrual_age_weeks=postmenstrual_age_weeks)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            tmp, tmpdir = materialize(await read_capped(file, limit), file.filename)
        except UploadError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            rep = check_recording(tmp, patient).to_dict()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        rep["path"] = file.filename
        audit.record("/check/upload", None, cfg.rule_base.version,
                     {"accepted": rep["accepted"], "rejected": [c["name"] for c in rep["checks"] if c["status"] == "reject"]})
        return rep

    @app.post("/monitor")
    def monitor(req: MonitorRequest) -> dict:
        path = server_path(req.path)
        return run_monitor(path, req.montage, req.window_s, req.step_s, cfg, audit,
                           patient=PatientInfo(age_years=req.age_years,
                                               postmenstrual_age_weeks=req.postmenstrual_age_weeks))

    @app.post("/feedback")
    def post_feedback(req: FeedbackRequest) -> dict:
        feedback.record(Correction(**req.model_dump()))
        audit.record("/feedback", None, cfg.rule_base.version,
                     {"target_kind": req.target_kind, "action": req.action})
        return {"status": "recorded"}

    return app
