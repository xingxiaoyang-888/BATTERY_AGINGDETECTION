"""Versioned application factory for a local battery research workspace."""

import csv
from io import StringIO
import logging
from pathlib import Path
import time
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import ROOT, Settings
from .data import parse_csv
from .forecast import robust_forecast
from .registry import model_registry
from .schemas import ForecastRequest, ImportRequest, SimulationRequest
from .simulation import simulate
from .store import Store

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    app = FastAPI(
        title="Sodium Twin",
        version=__version__,
        description="A local workspace for battery health, forecasting, and simulation.",
    )
    app.state.store = store

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id, start = uuid4().hex, time.perf_counter()
        # Restrict browser mutations to this origin without blocking CLI clients.
        origin = request.headers.get("origin")
        allowed = {
            str(request.base_url).rstrip("/"),
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and origin not in allowed:
            return JSONResponse({"detail": "Cross-origin writes are disabled."}, status_code=403)
        try:
            length = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)
        if length > 2_100_000:
            return JSONResponse(
                {"detail": "Request exceeds the 2 MB import limit."}, status_code=413
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - start) * 1000:.2f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_contract(_request, exc):
        # Exclude raw inputs so NaN/Infinity cannot break JSON error serialization.
        return JSONResponse(
            {
                "detail": [
                    {"loc": list(error["loc"]), "msg": error["msg"]} for error in exc.errors()
                ]
            },
            status_code=422,
        )

    @app.exception_handler(ValueError)
    async def invalid_input(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(_request, exc):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @app.get("/api/v2/health", tags=["Workspace"])
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "mode": "local-workspace",
            "training_enabled": False,
            "fmu_files": len(list((ROOT / "fmu_models").glob("*.fmu"))),
        }

    @app.get("/api/v2/datasets", tags=["Data"])
    def datasets():
        return store.datasets()

    @app.post("/api/v2/datasets", status_code=201, tags=["Data"])
    def import_dataset(body: ImportRequest):
        return store.import_frame(body.name, body.chemistry, parse_csv(body.csv))

    @app.get("/api/v2/datasets/{dataset_id}/cells/{cell_id}", tags=["Data"])
    def cell_history(dataset_id: str, cell_id: str):
        frame, _ = store.frame(dataset_id)
        selected = frame[frame.cell_id == cell_id]
        if selected.empty:
            raise KeyError("Cell not found.")
        return selected.to_dict("records")

    @app.get("/api/v2/template.csv", tags=["Data"])
    def template():
        lines = ["cell_id,cycle_index,soh_pct,temperature_c,c_rate,capacity_ah"]
        lines += [f"MY-CELL-01,{i},{100 - i * 0.02:.2f},25,1,50" for i in range(32)]
        return Response(
            "\n".join(lines),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="cycle-template.csv"'},
        )

    @app.get("/api/v2/models", tags=["Models"])
    def models():
        return model_registry(settings.model_dir)

    @app.post("/api/v2/forecasts", tags=["Forecasting"])
    def forecast(body: ForecastRequest):
        frame, dataset = store.frame(body.dataset_id)
        if body.model == "robust-trend":
            result = robust_forecast(frame, body, dataset["provenance"])
        else:
            try:
                from .ml.inference import learned_forecast
            except ImportError as exc:
                raise HTTPException(503, "Install sodium-twin[ml] to use learned models.") from exc
            result = learned_forecast(frame, body, dataset, settings.model_dir)
        return store.save_run("forecast", body.model_dump(), result)

    @app.post("/api/v2/simulations", tags=["Digital twin"])
    def simulation(body: SimulationRequest):
        return store.save_run("simulation", body.model_dump(), simulate(body))

    @app.get("/api/v2/fmu/configurations", tags=["Digital twin"])
    def fmu_configurations():
        from zipfile import ZipFile
        import re

        entries = []
        for file in sorted((ROOT / "fmu_models").glob("SystemForFMI_*x*.fmu")):
            match = re.fullmatch(r"SystemForFMI_(\d+)x(\d+).fmu", file.name)
            if not match:
                continue
            with ZipFile(file) as archive:
                platforms = sorted(
                    {
                        name.split("/")[1]
                        for name in archive.namelist()
                        if name.startswith("binaries/") and len(name.split("/")) > 2
                    }
                )
            entries.append(
                {
                    "series": int(match[1]),
                    "parallel": int(match[2]),
                    "file": file.name,
                    "platforms": platforms,
                }
            )
        return entries

    @app.post("/api/v2/simulations/fmu", tags=["Digital twin"])
    def fmu_simulation(body: SimulationRequest):
        import asyncio

        try:
            import server

            request = server.SimulationRequest(
                duration_s=body.duration_s,
                pack_current=body.current_a,
                env_temp=body.ambient_c,
                cell_capacity_ah=body.capacity_ah,
                init_soc=body.initial_soc_pct,
                init_soh=body.initial_soh_pct,
                cooling_ua_w_per_k=body.cooling_w_k,
                series_num=body.series,
                parallel_num=body.parallel,
                fault_s_index=1,
            )
            result = asyncio.run(server.run_twin_prediction(request))
            return store.save_run("fmu", body.model_dump(), {"engine": "fmu", **result})
        except ImportError as exc:
            raise HTTPException(
                503, "FMU dependencies are not installed; install requirements.txt."
            ) from exc

    @app.get("/api/v2/runs", tags=["Reports"])
    def runs():
        return store.runs()

    @app.get("/api/v2/runs/{run_id}", tags=["Reports"])
    def run(run_id: str):
        return store.run(run_id)

    @app.get("/api/v2/runs/{run_id}/export.csv", tags=["Reports"])
    def export(run_id: str):
        run = store.run(run_id)
        rows = run["result"].get("trajectory")
        if not rows:
            raise HTTPException(
                422, "This run uses the FMU contract; export its JSON record instead."
            )
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="sodium-{run_id}.csv"'},
        )

    if (settings.web_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=settings.web_dir / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        path = settings.web_dir / "index.html"
        if not path.is_file():
            return JSONResponse(
                {
                    "message": "API is ready. Build the web app: cd web && npm ci && npm run build.",
                    "docs": "/docs",
                }
            )
        return FileResponse(path)

    return app
