import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from apps.api.routes import (
    arbiter,
    broker,
    fault_harness,
    lakehouse,
    rate_limiter,
    router,
    simulator,
    windows,
)
from apps.api.security import SecurityHeadersMiddleware

logger = logging.getLogger("iip.server")
STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"

_bg_task = None


async def background_loop():
    logger.info("Starting background simulator and arbiter loop...")
    while True:
        try:
            fault = fault_harness.get_status()
            active_fault = fault if fault.get("active") else None
            tick = simulator.generate_tick(active_fault=active_fault)

            for evt in tick:
                broker.publish("telemetry.events", evt)
                windows.add_event(evt)

            # Ingest batch into lakehouse
            lakehouse.write_events(tick)

            # Evaluate windows across anomaly detectors
            for svc in simulator.services:
                win = windows.get_window(svc)
                if win:
                    arbiter.evaluate_window(win, active_incident=bool(active_fault))

        except Exception as e:
            logger.error(f"Error in background telemetry loop: {e}")

        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_task
    _bg_task = asyncio.create_task(background_loop())
    yield
    if _bg_task:
        _bg_task.cancel()


app = FastAPI(
    title="Infrastructure Intelligence Platform",
    version="1.0.0",
    description="Real-Time Data Science System for Telemetry Ingestion, Anomaly Detection, and Reliability Operations",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defensive security headers
app.add_middleware(SecurityHeadersMiddleware)


# Sliding window rate limiter
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static") or path in ["/", "/dashboard"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests - Rate limit exceeded."},
            headers={"Retry-After": str(retry_after)},
        )
    response = await call_next(request)
    headers = rate_limiter.get_rate_limit_headers(client_ip)
    for k, v in headers.items():
        response.headers[k] = v
    return response


app.include_router(router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
@app.get("/dashboard")
async def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Infrastructure Intelligence Platform API is running. Dashboard HTML not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.server:app", host="0.0.0.0", port=8000, reload=True)
