import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
import httpx
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("telemetry_demo")

# OpenTelemetry Resource & Tracer / Meter Setup
otel_resource = Resource.create({
    "service.name": "order-service",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

tracer_provider = TracerProvider(resource=otel_resource)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("order-service", "1.0.0")

meter_provider = MeterProvider(resource=otel_resource)
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("order-service", "1.0.0")

http_request_counter = meter.create_counter(
    "http.server.requests",
    unit="1",
    description="Total count of incoming HTTP requests",
)
http_request_duration = meter.create_histogram(
    "http.server.duration",
    unit="ms",
    description="Duration of incoming HTTP requests in milliseconds",
)


class OrderItem(BaseModel):
    item_id: str
    name: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0.0)


class CreateOrderRequest(BaseModel):
    customer_id: str
    items: List[OrderItem]
    simulate_delay_ms: Optional[float] = 0.0
    simulate_error: Optional[bool] = False


class Order(BaseModel):
    order_id: str
    customer_id: str
    items: List[OrderItem]
    total_amount: float
    status: str
    created_at: str


class TelemetryBatchExporter:
    """Buffers and exports telemetry events to the IIP platform."""

    def __init__(self, collector_url: str = "http://127.0.0.1:8000/api/telemetry/ingest", flush_interval_secs: float = 1.0):
        self.collector_url = collector_url
        self.flush_interval = flush_interval_secs
        self.buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.client: Optional[httpx.AsyncClient] = None

    async def start(self):
        self._running = True
        self.client = httpx.AsyncClient(timeout=3.0)
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        await self.flush()
        if self.client:
            await self.client.aclose()

    async def record_event(self, event_dict: Dict[str, Any]):
        async with self._lock:
            self.buffer.append(event_dict)
            if len(self.buffer) >= 50:
                await self._flush_unlocked()

    async def flush(self):
        async with self._lock:
            await self._flush_unlocked()

    async def _flush_unlocked(self):
        if not self.buffer or not self.client:
            return
        batch = list(self.buffer)
        self.buffer.clear()
        try:
            await self.client.post(self.collector_url, json={"events": batch})
        except Exception as e:
            logger.debug(f"Telemetry export failed: {e}")

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self.flush()


exporter = TelemetryBatchExporter()


class TelemetryInstrumentationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_name: str = "order-service"):
        super().__init__(app)
        self.service_name = service_name

    async def dispatch(self, request: Request, call_next):
        req_timestamp = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        status_code = 500
        error_instance: Optional[Exception] = None

        with tracer.start_as_current_span(
            f"HTTP {request.method} {request.url.path}",
            attributes={
                "http.method": request.method,
                "http.route": request.url.path,
                "service.name": self.service_name,
            },
        ) as span:
            span_ctx = span.get_span_context()
            trace_id = format(span_ctx.trace_id, "032x") if span_ctx.trace_id else uuid.uuid4().hex
            span_id = format(span_ctx.span_id, "016x") if span_ctx.span_id else uuid.uuid4().hex[:16]

            try:
                response: Response = await call_next(request)
                status_code = response.status_code
                span.set_attribute("http.status_code", status_code)
                if status_code >= 500:
                    span.set_status(trace.StatusCode.ERROR, description=f"HTTP {status_code} Error")
                else:
                    span.set_status(trace.StatusCode.OK)
                return response
            except Exception as ex:
                status_code = 500
                error_instance = ex
                span.set_attribute("http.status_code", status_code)
                span.set_status(trace.StatusCode.ERROR, description=str(ex))
                span.record_exception(ex)
                raise ex
            finally:
                dur_ms = max(0.01, (time.perf_counter() - start_time) * 1000.0)
                is_err = 1.0 if status_code >= 500 else 0.0

                # Record OTel SDK metrics
                http_request_counter.add(1, {"http.status_code": status_code, "http.method": request.method})
                http_request_duration.record(dur_ms, {"http.status_code": status_code, "http.method": request.method})

                # Normalized events for IIP ingestion
                evt_lat = {
                    "event_id": f"evt-{uuid.uuid4().hex[:12]}",
                    "event_time": req_timestamp.isoformat(),
                    "service": self.service_name,
                    "environment": "production",
                    "event_type": "service_metric",
                    "metric_name": "request_latency_ms",
                    "metric_value": round(dur_ms, 2),
                    "unit": "ms",
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "attributes": {
                        "http.method": request.method,
                        "http.route": request.url.path,
                        "http.status_code": status_code,
                    },
                }
                evt_err = {
                    "event_id": f"evt-{uuid.uuid4().hex[:12]}",
                    "event_time": req_timestamp.isoformat(),
                    "service": self.service_name,
                    "environment": "production",
                    "event_type": "service_metric",
                    "metric_name": "error_rate",
                    "metric_value": is_err,
                    "unit": "ratio",
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "attributes": {
                        "http.method": request.method,
                        "http.route": request.url.path,
                        "http.status_code": status_code,
                    },
                }
                asyncio.create_task(exporter.record_event(evt_lat))
                asyncio.create_task(exporter.record_event(evt_err))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await exporter.start()
    yield
    await exporter.stop()


order_app = FastAPI(title="Order Service", lifespan=lifespan)
order_app.add_middleware(TelemetryInstrumentationMiddleware, service_name="order-service")
orders_db: Dict[str, Order] = {}


@order_app.get("/")
async def root():
    return {
        "service": "order-service",
        "description": "Demo microservice instrumented with OpenTelemetry SDK",
        "status": "UP",
        "endpoints": {
            "health": "GET /health",
            "orders": "POST /orders",
            "docs": "GET /docs",
        },
        "collector_target": exporter.collector_url,
        "platform_dashboard": "http://127.0.0.1:8000/dashboard",
    }


@order_app.get("/health")
async def health_check():
    return {"status": "UP", "service": "order-service", "timestamp": datetime.now(timezone.utc).isoformat()}


@order_app.get("/orders")
async def list_orders():
    return {
        "service": "order-service",
        "total_orders": len(orders_db),
        "recent_orders": list(orders_db.values())[-20:],
    }


@order_app.post("/orders", response_model=Order)
async def create_order(payload: CreateOrderRequest):
    if payload.simulate_delay_ms and payload.simulate_delay_ms > 0:
        time.sleep(payload.simulate_delay_ms / 1000.0)

    if payload.simulate_error:
        raise HTTPException(status_code=500, detail="Simulated payment failure")

    total = sum(item.price * item.quantity for item in payload.items)
    order_id = f"ord-{uuid.uuid4().hex[:8]}"
    order = Order(
        order_id=order_id,
        customer_id=payload.customer_id,
        items=payload.items,
        total_amount=round(total, 2),
        status="CONFIRMED",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    orders_db[order_id] = order
    return order
