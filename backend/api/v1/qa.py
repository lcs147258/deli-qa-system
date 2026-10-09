# backend/api/v1/qa.py
# 问答接口路由。前端 static/index.html 的 JSON 契约逐字保持：
#   POST /api/create_session、POST /api/query、WS /api/stream、
#   GET|DELETE /api/history/{sid}、GET /api/sources、GET /health。

import asyncio
import json
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel

from backend.agents.qa import QAService
from backend.config import get_settings
from backend.core.logger import get_logger
from backend.dependencies import get_faq, get_mysql_client

logger = get_logger(__name__)
router = APIRouter()

# ── 问候语（高频问候规则秒回，不进流水线、不写历史、不耗 LLM；话术口语化）──
GREETING_PATTERNS = [
    {"pattern": r"^(你好|您好|hi|hello)",
     "response": "你好呀～我是智慧出品的客服，想了解什么产品或门店问题，直接问我就行！"},
    {"pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
     "response": "我是智慧出品熟食店的客服，产品价格、食安操作、门店运营这些事儿都可以问我～"},
    {"pattern": r"^(在吗|在不在|有人吗)",
     "response": "在的在的，您请说～"},
    {"pattern": r"^(干嘛呢|你在干嘛|做什么)",
     "response": "在呢，随时待命～有产品或门店的问题直接说！"},
]


def check_greeting(query: str) -> Optional[str]:
    text = (query or "").strip()
    for item in GREETING_PATTERNS:
        if re.match(item["pattern"], text, re.IGNORECASE):
            return item["response"]
    return None


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


_qa_service: Optional[QAService] = None


def _get_qa_service() -> QAService:
    """问答流水线单例（首次调用才加载模型）。"""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service


# ── 会话 ──────────────────────────────────────────────────────

@router.post("/api/create_session")
async def create_session():
    return {"session_id": str(uuid.uuid4())}


@router.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = await asyncio.to_thread(
            get_mysql_client().fetch_recent_history, session_id, 5)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.warning("api.get_history_failed", error=str(e))
        return {"session_id": session_id, "history": []}


@router.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    ok = await asyncio.to_thread(get_mysql_client().clear_history, session_id)
    if ok:
        return {"status": "success", "message": "历史记录已清除"}
    raise HTTPException(status_code=500, detail="清除历史记录失败")


@router.get("/api/sources")
async def get_sources():
    return {"sources": get_settings().valid_sources}


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


# ── 非流式查询：问候 / FAQ 直答；RAG 请求切换 WebSocket ────────

@router.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    greeting = check_greeting(request.query)
    if greeting:
        return {
            "answer": greeting,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }

    settings = get_settings()
    try:
        answer = await asyncio.to_thread(
            get_faq().answer, request.query, settings.faq_bm25_threshold)
    except Exception as e:
        logger.warning("api.faq_check_failed", error=str(e))
        answer = None
    if answer:
        # FAQ 命中也要落会话历史（与流水线内 save_memory 同口径）
        try:
            await asyncio.to_thread(
                get_mysql_client().insert_conversation,
                session_id, request.query, answer)
        except Exception:
            pass
        return {
            "answer": answer,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }

    return {
        "answer": "请使用WebSocket接口获取流式响应",
        "is_streaming": True,
        "session_id": session_id,
        "processing_time": time.time() - start_time,
    }


# ── WebSocket 流式问答 ────────────────────────────────────────

@router.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id") or str(uuid.uuid4())
            start_time = time.time()

            async def send(payload: dict):
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(payload)

            await send({"type": "start", "session_id": session_id})

            # 问候语：单 token + end（不进流水线）
            greeting = check_greeting(query)
            if greeting:
                await send({"type": "token", "token": greeting, "session_id": session_id})
                await send({"type": "end", "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time})
                break

            # 跑问答流水线：转发 RAG/闲聊生成的 LLM token
            accumulated = ""
            final = {"answer": "", "sources": [], "answer_mode": "", "confidence": 0.0}
            async for event, payload in _get_qa_service().astream_answer(
                query, session_id, source_filter=source_filter,
            ):
                if event == "token":
                    accumulated += payload
                    await send({"type": "token", "token": payload,
                                "session_id": session_id})
                elif event == "done":
                    final = payload

            # FAQ 命中 / 固定兜底话术不产生 LLM token；RAG 的来源后缀也在此补发。
            # 规则：最终答案以流式正文为前缀则补差值，否则整段补发。
            answer = final.get("answer", "")
            if answer.startswith(accumulated):
                suffix = answer[len(accumulated):]
                if suffix:
                    await send({"type": "token", "token": suffix,
                                "session_id": session_id})
            elif answer:
                await send({"type": "token", "token": answer,
                            "session_id": session_id})

            await send({"type": "end", "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time})
            break

    except WebSocketDisconnect as e:
        logger.info("ws.disconnected", code=e.code, reason=e.reason)
    except Exception as e:
        logger.error("ws.error", error=str(e), exc_info=True)
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception:
            pass
