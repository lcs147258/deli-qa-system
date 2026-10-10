# backend/main.py
# DeliRAG Web 服务入口：装配日志/CORS/静态资源/问答路由。
# 启动：python -m uvicorn backend.main:app --host 0.0.0.0 --port 8082
# （HOST/PORT 环境变量兼容旧启动脚本）

import os
import pathlib

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.v1.qa import router as qa_router
from backend.config import PROJECT_ROOT, get_settings
from backend.core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

STATIC_DIR = pathlib.Path(PROJECT_ROOT) / "static"

app = FastAPI(title="DeliRAG 熟食智能问答系统",
              description="FAQ 热路径 + pgvector/pg_search 双路召回四策略父子 RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.on_event("startup")
async def _on_startup():
    import asyncio

    settings = get_settings()
    logger.info("app.startup",
                port=settings.app_port,
                kb_table=settings.pg_table,
                tenant=settings.default_tenant_id)

    # 后台预热 FAQ 通道（MySQL/Redis 连接 + BM25 语料），
    # 组件内部已做失败降级，预热结果不影响 Web 服务就绪。
    from backend.dependencies import get_faq

    async def _warmup():
        try:
            await asyncio.to_thread(get_faq)
        except Exception as e:
            logger.warning("startup.faq_warmup_failed", error=str(e))

    asyncio.create_task(_warmup())


if __name__ == "__main__":
    settings = get_settings()
    host = os.getenv("HOST", settings.app_host)
    port = int(os.getenv("PORT", settings.app_port))
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
