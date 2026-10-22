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
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

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


def create_app(config=None, audit_path=None, feedback_path=None) -> FastAPI:
    cfg = config or load_configs()
    audit = AuditLogger(audit_path)
    feedback = FeedbackLogger(feedback_path or "feedback.jsonl")

    app = FastAPI(
        title="NeuroLens API",
        version=__version__,
        description="EEG detection & interpretation service (decision-support). " + DISCLAIMER,
    )

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
        path = _SCHEMAS_DIR / f"{name}.schema.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"unknown schema {name!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        if not Path(req.path).exists():
            raise HTTPException(status_code=404, detail=f"file not found: {req.path}")
        try:
            return run_analyze(req.path, req, cfg, audit)
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
    ) -> dict:
        try:
            params = AnalyzeParams(
                mode=mode, montage=montage, provider=provider,
                age_years=age_years, postmenstrual_age_weeks=postmenstrual_age_weeks,
                clinical_question=clinical_question, recording_setting=recording_setting,
            )
        except ValueError as exc:  # pydantic ValidationError: invalid form value -> 422, not 500
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            tmp, tmpdir = materialize(await file.read(), file.filename)
        except UploadError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            return run_analyze(tmp, params, cfg, audit)
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
            tmp, tmpdir = materialize(await file.read(), file.filename)
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
        if not Path(req.path).exists():
            raise HTTPException(status_code=404, detail=f"file not found: {req.path}")
        return run_monitor(req.path, req.montage, req.window_s, req.step_s, cfg, audit,
                           patient=PatientInfo(age_years=req.age_years,
                                               postmenstrual_age_weeks=req.postmenstrual_age_weeks))

    @app.post("/feedback")
    def post_feedback(req: FeedbackRequest) -> dict:
        feedback.record(Correction(**req.model_dump()))
        audit.record("/feedback", None, cfg.rule_base.version,
                     {"target_kind": req.target_kind, "action": req.action})
        return {"status": "recorded"}

    return app
