# backend/agents/qa/service.py
# 熟食问答流水线（普通 async 服务层，无 LangGraph / 无 Agent 自主决策）。
#
# 系统本质是一条确定性流水线，每一步路由都由规则/小模型/阈值决定，LLM 只负责：
#   选检索策略（四选一）、Query 改写（HyDE/子查询/回溯）、照知识库写回答。
#
# 流程：
#   用户问题
#     ├─ ① FAQ 热路径（BM25 >= 0.85 直答，不走向量）
#     ├─ ② 意图判定（规则 -> 熟食 BERT 二分类）
#     │     └─ 通用 GENERAL            -> LLM 寒暄直答
#     └─ ③ 专业问题 -> LLM 四策略选择器（直接/HyDE/子查询/回溯）
#           -> BGE-M3 子块混合召回 -> 父块回升（Small-to-Big）-> BGE 父块精排
#                 ├─ 高置信 -> LLM 严格按知识库生成（流式，附参考来源）
#                 └─ 低置信 -> 固定转人工话术（不调 LLM）+ 入待补充队列
#     └─ ④ 会话历史落 MySQL conversations（最近 5 轮）
#
# 食品安全红线：低置信绝不调通用 LLM 硬答；不泄露内部配方/成本。

import asyncio
from datetime import datetime
from typing import AsyncIterator, Optional

from backend.agents.qa.prompts import (
    BACKTRACK_PROMPT,
    GENERAL_ANSWER_PROMPT,
    HYDE_PROMPT,
    MULTI_QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    SYSTEM_PROMPT,
)
from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.pending import enqueue_pending
from backend.core.query_classifier import LABEL_GENERAL, classify_query
from backend.core.reranker import rerank_parents, retrieve
from backend.core.strategy_selector import (
    BACKTRACK,
    DIRECT,
    HYDE,
    SUBQUERY,
    aselect_strategy,
)

logger = get_logger(__name__)

# ── 检索参数 ──────────────────────────────────────────────────
MAX_SUB_QUERIES     = 3
SUBQUERY_RECALL_K   = 8      # 每条子查询的子块召回数
SUBQUERY_PER_TOP_K  = 5      # 每条子查询回升+精排后保留的父块数（合并前）
GENERAL_MIN_PROB    = 0.85   # BERT 判"通用知识"的置信门槛，低于此按专业问题走 RAG

_WEEKDAYS_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# ── 规则集（熟食领域）──────────────────────────────────────────
_GENERAL_EXACT = {
    "你好", "您好", "hi", "hello", "嗨", "hey",
    "谢谢", "谢谢你", "感谢", "thanks", "thank you",
    "你是谁", "你叫什么", "你叫什么名字", "你是什么",
    "你能做什么", "你有什么功能", "你能帮我做什么",
    "再见", "拜拜", "bye",
}
_GENERAL_KEYWORDS = (
    "你是谁", "你叫什么", "你能做什么", "你有什么功能",
    "介绍一下你自己", "自我介绍",
    "今天天气", "天气怎么样", "讲个笑话", "说个故事",
    "今天几号", "今天是几号", "今天是星期", "现在几点",
    "现在时间", "当前时间", "星期几",
)
# 命中即确认专业问题，跳过 BERT（走 RAG 是食品安全场景的安全方向）
_SPECIALIZED_KEYWORDS = (
    "多少钱", "价格", "价目", "一斤", "半斤", "克重", "规格", "批发价", "零售价",
    "酱鸭", "卤味", "熟食", "凉菜", "猪头肉", "鸡爪", "桶子鸡", "调味料汁", "红油",
    "冷藏", "冷冻", "冷链", "温度", "消毒", "效期", "留样", "陈列", "保质期",
    "贮存", "加热", "中心温度", "报损", "卫生", "操作规范", "记录", "灭蝇", "洗手",
    "加盟", "配送", "订货", "门店", "巡店", "供应商", "证照", "营业", "员工",
)


def _is_general_by_rule(query: str) -> bool:
    q = query.strip().lower()
    if q in _GENERAL_EXACT:
        return True
    return any(kw in q for kw in _GENERAL_KEYWORDS)


def _is_specialized_by_rule(query: str) -> bool:
    q = query.strip().lower()
    return any(kw in q for kw in _SPECIALIZED_KEYWORDS)


def _current_time_str() -> str:
    now = datetime.now()
    return now.strftime(f"%Y年%m月%d日 {_WEEKDAYS_CN[now.weekday()]} %H:%M")


def _fallback_answer() -> str:
    """低置信固定话术：不调 LLM，杜绝食品安全幻觉。"""
    phone = get_settings().customer_service_phone
    return (
        "抱歉，知识库中暂未找到与您问题充分相关的可靠信息，"
        "为避免误导，我不能随意作答。\n\n"
        f"请联系人工客服咨询，电话：{phone}。"
    )


def _format_history_text(history: list[dict]) -> str:
    """MySQL 会话历史 [{'question','answer'}] -> 提示词文本。"""
    if not history:
        return "（无历史对话）"
    return "\n".join(
        f"用户：{h['question']}\nAI：{str(h['answer'])[:200]}" for h in history)


def _history_messages(history: list[dict]) -> list[dict]:
    """历史 -> OpenAI 原生多轮消息（最近 5 轮，放在 system 之后、当前问题之前）。"""
    messages = []
    for h in (history or [])[-5:]:
        messages.append({"role": "user", "content": str(h["question"])})
        messages.append({"role": "assistant", "content": str(h["answer"])[:500]})
    return messages


class QAService:
    """
    问答流水线门面。API 层只依赖：
      aanswer(...)        非流式，返回最终结果 dict
      astream_answer(...) 流式，yield ("token", str) / ("done", dict)
    """

    # ── 对外接口 ─────────────────────────────────────────────

    async def aanswer(self, query: str, session_id: str,
                      source_filter: Optional[str] = None,
                      tenant_id: str = None) -> dict:
        ctx = self._new_ctx(query, session_id, source_filter, tenant_id)
        await self._run(ctx, token_queue=None)
        return self._result(ctx)

    async def astream_answer(
        self, query: str, session_id: str,
        source_filter: Optional[str] = None,
        tenant_id: str = None,
    ) -> AsyncIterator[tuple[str, object]]:
        ctx = self._new_ctx(query, session_id, source_filter, tenant_id)
        queue: asyncio.Queue = asyncio.Queue()

        async def runner():
            try:
                await self._run(ctx, token_queue=queue)
            except Exception as e:
                logger.error("qa.pipeline_unexpected_error",
                             error=str(e)[:300], exc_info=True)
                ctx.update(answer=_fallback_answer(), sources=[],
                           answer_mode="fallback", confidence=0.0)
            finally:
                await queue.put(("__end__", None))

        task = asyncio.create_task(runner())
        while True:
            event, payload = await queue.get()
            if event == "__end__":
                break
            yield event, payload
        await task
        yield ("done", self._result(ctx))

    # ── 流水线主体 ───────────────────────────────────────────

    def _new_ctx(self, query, session_id, source_filter, tenant_id) -> dict:
        settings = get_settings()
        return {
            "query": query,
            "session_id": session_id,
            "tenant_id": tenant_id or settings.default_tenant_id,
            "course_id": source_filter,
            "history": [],
            "answer": "",
            "sources": [],
            "answer_mode": "",
            "confidence": 0.0,
            "ranked_chunks": [],
            "strategy": DIRECT,
        }

    @staticmethod
    def _result(ctx: dict) -> dict:
        return {
            "answer": ctx.get("answer", ""),
            "sources": ctx.get("sources", []),
            "answer_mode": ctx.get("answer_mode", ""),
            "confidence": ctx.get("confidence", 0.0),
        }

    async def _run(self, ctx: dict, token_queue: Optional[asyncio.Queue]) -> None:
        query = ctx["query"]

        # ① FAQ 热路径
        answer = await self._faq_answer(query)
        if answer:
            ctx.update(answer=answer, answer_mode="faq")
            logger.info("qa.faq_hit", query=query[:30])
            await self._save_memory(ctx)
            return

        # 载入多轮历史（MySQL，失败降级为无历史）
        ctx["history"] = await self._load_history(ctx["session_id"])

        # ② 意图判定：规则 -> BERT
        if not await self._is_specialized(ctx):
            await self._generate_general(ctx, token_queue)
            await self._save_memory(ctx)
            return

        # ③ 专业问题：四策略检索
        ctx["strategy"] = await aselect_strategy(query)
        await self._retrieve_by_strategy(ctx)

        # ④ 置信度路由
        settings = get_settings()
        if ctx["confidence"] >= settings.high_confidence_threshold:
            await self._generate_rag(ctx, token_queue)
        else:
            self._generate_fallback(ctx)
            await self._enqueue_pending(ctx)

        await self._save_memory(ctx)

    # ── ① FAQ ────────────────────────────────────────────────

    @staticmethod
    async def _faq_answer(query: str) -> Optional[str]:
        from backend.dependencies import get_faq
        try:
            faq = get_faq()
            return await asyncio.to_thread(faq.answer, query)
        except Exception as e:
            logger.warning("qa.faq_check_failed_degrade_rag", error=str(e))
            return None

    # ── ② 意图判定 ───────────────────────────────────────────

    async def _is_specialized(self, ctx: dict) -> bool:
        query = ctx["query"]

        if _is_general_by_rule(query):
            logger.info("classify.general_by_rule", query=query[:30])
            return False

        if _is_specialized_by_rule(query):
            logger.info("classify.specialized_by_rule", query=query[:30])
            return True

        loop = asyncio.get_running_loop()
        label, scores = await loop.run_in_executor(None, classify_query, query)
        if label == LABEL_GENERAL and scores[LABEL_GENERAL] >= GENERAL_MIN_PROB:
            logger.info("classify.general_by_bert",
                        query=query[:30], prob=round(scores[LABEL_GENERAL], 4))
            return False

        logger.info("classify.specialized_by_bert",
                    query=query[:30],
                    prob_general=round(scores[LABEL_GENERAL], 4))
        return True

    # ── ③ 四策略检索 ─────────────────────────────────────────

    async def _retrieve_by_strategy(self, ctx: dict) -> None:
        strategy = ctx["strategy"]
        query = ctx["query"]
        loop = asyncio.get_running_loop()
        settings = get_settings()

        if strategy == HYDE:
            retrieve_query = await self._hyde_document(ctx)
            ranked, confidence = await loop.run_in_executor(
                None,
                lambda: retrieve(retrieve_query, ctx["tenant_id"], ctx["course_id"],
                                 settings.recall_top_k, settings.rerank_top_k,
                                 settings.min_relevant_score),
            )
        elif strategy == BACKTRACK:
            base_query = await self._backtrack_query(ctx)
            ranked, confidence = await loop.run_in_executor(
                None,
                lambda: retrieve(base_query, ctx["tenant_id"], ctx["course_id"],
                                 settings.recall_top_k, settings.rerank_top_k,
                                 settings.min_relevant_score),
            )
        elif strategy == SUBQUERY:
            ranked, confidence = await self._retrieve_subqueries(ctx)
        else:  # DIRECT
            ranked, confidence = await loop.run_in_executor(
                None,
                lambda: retrieve(query, ctx["tenant_id"], ctx["course_id"],
                                 settings.recall_top_k, settings.rerank_top_k,
                                 settings.min_relevant_score),
            )

        ctx["ranked_chunks"] = [
            {"content": d.content, "score": d.score, "metadata": d.metadata}
            for d in ranked
        ]
        ctx["confidence"] = confidence
        logger.info("qa.retrieve_done", strategy=strategy,
                    ranked=len(ctx["ranked_chunks"]),
                    confidence=round(confidence, 4))

    async def _hyde_document(self, ctx: dict) -> str:
        """VAGUE/HYDE：生成假设文档扩充语义；失败回退原问题。"""
        from backend.core.llm_factory import get_llm
        try:
            llm = get_llm("qa", temperature=0.3)
            doc = await llm.achat([{"role": "user", "content": HYDE_PROMPT.format(
                history=_format_history_text(ctx["history"]), query=ctx["query"])}])
            doc = doc.strip()
            logger.info("qa.hyde_done", query=ctx["query"][:30], doc_len=len(doc))
            return doc or ctx["query"]
        except Exception as e:
            logger.warning("qa.hyde_failed_use_original", error=str(e)[:200])
            return ctx["query"]

    async def _backtrack_query(self, ctx: dict) -> str:
        """BACKTRACK：Step-back 改写为基础问题；失败回退原问题。"""
        from backend.core.llm_factory import get_llm
        try:
            llm = get_llm("qa", temperature=0.1)
            base = await llm.achat([{"role": "user", "content": BACKTRACK_PROMPT.format(
                query=ctx["query"])}], max_tokens=128)
            base = base.strip().split("\n")[0].strip()
            logger.info("qa.backtrack_done",
                        query=ctx["query"][:30], base_query=base[:50])
            return base or ctx["query"]
        except Exception as e:
            logger.warning("qa.backtrack_failed_use_original", error=str(e)[:200])
            return ctx["query"]

    async def _retrieve_subqueries(self, ctx: dict):
        """SUBQUERY：拆子查询 -> 各路父块检索 -> 合并去重 -> 对原问题统一父块精排。"""
        from backend.core.llm_factory import get_llm

        query = ctx["query"]
        last_ai = ctx["history"][-1]["answer"] if ctx["history"] else ""
        sub_queries = [query]
        try:
            llm = get_llm("qa", temperature=0.3)
            raw = await llm.achat([{"role": "user", "content":
                                    MULTI_QUERY_REWRITE_PROMPT.format(
                                        last_answer=last_ai[:500] or "（无上一轮回答）",
                                        query=query)}])
            sub_queries = [
                line.lstrip("0123456789.-）)、 ").strip()
                for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ][:MAX_SUB_QUERIES] or [query]
        except Exception as e:
            logger.warning("qa.subquery_rewrite_failed", error=str(e)[:200])
        logger.info("qa.subqueries", original=query[:30], queries=sub_queries)

        loop = asyncio.get_running_loop()

        async def one(sub_q: str):
            return await loop.run_in_executor(
                None,
                lambda: retrieve(sub_q, ctx["tenant_id"], ctx["course_id"],
                                 SUBQUERY_RECALL_K, SUBQUERY_PER_TOP_K,
                                 get_settings().min_relevant_score),
            )

        results = await asyncio.gather(*[one(q) for q in sub_queries])

        # 父块按内容去重（保留各路最高分）
        merged: dict[str, dict] = {}
        for docs, _ in results:
            for d in docs:
                key = d.content[:200]
                if key not in merged or d.score > merged[key]["score"]:
                    merged[key] = {"content": d.content,
                                   "score": d.score, "metadata": d.metadata}
        logger.info("qa.subquery_merged",
                    lanes=len(sub_queries), unique_parents=len(merged))
        return await loop.run_in_executor(
            None, rerank_parents, query, list(merged.values()))

    # ── ④ 生成 ───────────────────────────────────────────────

    async def _generate_rag(self, ctx: dict,
                            token_queue: Optional[asyncio.Queue]) -> None:
        from backend.core.llm_factory import get_llm

        settings = get_settings()
        chunks = ctx["ranked_chunks"]

        context_parts, sources = [], []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"【参考{i}】\n{chunk['content']}")
            name = chunk.get("metadata", {}).get("source_name", "业务文档")
            if name and name not in sources:
                sources.append(name)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += _history_messages(ctx["history"])
        messages.append({"role": "user", "content": RAG_ANSWER_PROMPT.format(
            context="\n\n".join(context_parts),
            query=ctx["query"],
            phone=settings.customer_service_phone,
        )})

        llm = get_llm("qa", temperature=settings.llm_temperature)
        try:
            body = await self._produce(llm, messages, token_queue)
            body = body.strip()
        except Exception as e:
            # LLM 故障（鉴权/限流/超时）fail-safe：不向前端抛技术错误，转人工
            logger.error("qa.rag_llm_failed_fallback", error=str(e)[:200])
            answer = _fallback_answer()
            ctx.update(answer=answer, sources=[], answer_mode="fallback")
            return

        final_answer = f"{body}\n\n📚 **参考来源**\n" + "\n".join(
            f"  • {s}" for s in sources)
        ctx.update(answer=final_answer, sources=sources, answer_mode="rag")
        logger.info("qa.rag_done", answer_len=len(final_answer),
                    sources=sources, confidence=round(ctx["confidence"], 4))

    async def _generate_general(self, ctx: dict,
                                token_queue: Optional[asyncio.Queue]) -> None:
        from backend.core.llm_factory import get_llm

        prompt = GENERAL_ANSWER_PROMPT.format(
            current_time=_current_time_str(),
            history=_format_history_text(ctx["history"]),
            query=ctx["query"],
        )
        llm = get_llm("qa", temperature=0.3)
        try:
            answer = (await self._produce(
                llm, [{"role": "user", "content": prompt}], token_queue)).strip()
        except Exception as e:
            logger.error("qa.general_llm_failed_fallback", error=str(e)[:200])
            answer = "您好，我是智慧出品熟食店的客服，有产品或门店的问题随时问我～"

        ctx.update(answer=answer, sources=[],
                   answer_mode="general", confidence=1.0)
        logger.info("qa.general_done", answer_len=len(answer))

    @staticmethod
    async def _produce(llm, messages: list[dict],
                       token_queue: Optional[asyncio.Queue]) -> str:
        """流式/非流式统一出口：流式时逐 token 入队并累积全文。"""
        if token_queue is None:
            return await llm.achat(messages)
        parts = []
        async for token in llm.astream(messages):
            parts.append(token)
            await token_queue.put(("token", token))
        return "".join(parts)

    def _generate_fallback(self, ctx: dict) -> None:
        answer = _fallback_answer()
        logger.info("qa.fallback",
                    confidence=round(ctx["confidence"], 4),
                    query=ctx["query"][:30])
        ctx.update(answer=answer, sources=[], answer_mode="fallback")

    # ── 旁路：待补充队列 + 会话落库 ────────────────────────────

    @staticmethod
    async def _enqueue_pending(ctx: dict) -> None:
        try:
            await asyncio.to_thread(
                enqueue_pending, ctx["query"], ctx.get("answer", ""),
                float(ctx.get("confidence", 0.0)))
        except Exception as e:
            logger.warning("qa.enqueue_pending_failed", error=str(e))

    @staticmethod
    async def _load_history(session_id: str) -> list[dict]:
        from backend.dependencies import get_mysql_client
        try:
            client = get_mysql_client()
            if getattr(client, "available", False):
                return await asyncio.to_thread(
                    client.fetch_recent_history, session_id, 5)
        except Exception as e:
            logger.warning("qa.load_history_failed", error=str(e))
        return []

    @staticmethod
    async def _save_memory(ctx: dict) -> None:
        from backend.dependencies import get_mysql_client
        try:
            client = get_mysql_client()
            if getattr(client, "available", False):
                await asyncio.to_thread(
                    client.insert_conversation,
                    ctx["session_id"], ctx["query"], ctx.get("answer", ""))
                logger.info("qa.memory_saved", session_id=ctx["session_id"])
        except Exception as e:
            logger.warning("qa.save_memory_failed", error=str(e))
