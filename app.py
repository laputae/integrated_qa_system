import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from base import logger
from gateway.middleware import GatewayMiddleware
from gateway.security_headers import SecurityHeadersMiddleware
from main import IntegratedQASystem
from routers.v1.auth import router as auth_router
from routers.v1.chunk_config_routes import router as chunk_config_router
from routers.v1.eval_routes import router as eval_router
from routers.v1.routes import router as routes_router
from routers.v1.ws import router as ws_router

qa_system = IntegratedQASystem()

logger.info("Config loaded from config.ini — all required settings present.")

_llm_semaphore = asyncio.Semaphore(qa_system.config.MAX_CONCURRENT_LLM_CALLS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await qa_system.health.start_background_recovery()
    if qa_system.eval_service:
        await qa_system.eval_service.start_periodic_eval()
    yield
    if qa_system.eval_service:
        await qa_system.eval_service.stop_periodic_eval()
    await qa_system.health.close()


app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统", lifespan=lifespan)

app.add_middleware(GatewayMiddleware)

if qa_system.config.SECURE_HEADERS_ENABLED:
    app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=qa_system.config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Include sub-routers
app.include_router(auth_router)
app.include_router(routes_router)
app.include_router(eval_router)
app.include_router(chunk_config_router)
app.include_router(ws_router)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
