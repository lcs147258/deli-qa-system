# backend/core/llm_factory.py
# 大模型统一工厂：DeepSeek OpenAI 兼容端点（默认模型 deepseek-chat）。
# 模型/密钥/端点全部走 .env.local，切换厂商（如回 DashScope）只改配置不改代码。
# 实现选择：本系统是固定流水线而非 Agent，LLM 只做"照检索结果写回答"，
# 直接用 openai SDK（AsyncOpenAI）即可，不引入 langchain 模型抽象层。
#
# messages 统一为 OpenAI 原生 dict 格式：
#   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

from typing import AsyncIterator, Optional

import httpx
import openai

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 绕过系统代理：Windows HTTPS_PROXY 会让私有兼容端点 TLS 握手失败
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)
_HTTP_SYNC_CLIENT = httpx.Client(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)


class LLMClient:
    """openai SDK 轻封装：非流式 achat/chat + 流式 astream。按 temperature 缓存。"""

    def __init__(self, temperature: float):
        settings = get_settings()
        self.model = settings.llm_model
        self.temperature = temperature
        self.max_tokens = settings.llm_max_tokens
        common = {
            "api_key": settings.llm_api_key,
            "base_url": settings.llm_base_url,
            "max_retries": 1,
        }
        self._async_client = openai.AsyncOpenAI(
            **common, http_client=_HTTP_ASYNC_CLIENT)
        self._sync_client = openai.OpenAI(
            **common, http_client=_HTTP_SYNC_CLIENT)
        logger.info("llm_factory.client_initialized",
                    model=self.model, temperature=temperature)

    # ── 非流式 ───────────────────────────────────────────────

    async def achat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """同步版本（评估脚本等无事件循环场景使用）。"""
        resp = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    # ── 流式（仅 RAG / 闲聊回答用，逐 token 转发前端）──────────

    async def astream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── 工厂单例 ──────────────────────────────────────────────────

_clients: dict[float, LLMClient] = {}


def get_llm(agent_type: str = "qa", temperature: float = None,
            streaming: bool = False) -> LLMClient:
    """
    取 LLM 客户端。
    agent_type 仅保留语义标记（qa/intent），底层同一模型；
    streaming 不影响实例（achat/astream 是两个方法），参数保留以兼容调用点。
    """
    settings = get_settings()
    if temperature is None:
        temperature = settings.llm_temperature
    if temperature not in _clients:
        _clients[temperature] = LLMClient(temperature=temperature)
    return _clients[temperature]


def clear_cache() -> None:
    _clients.clear()
