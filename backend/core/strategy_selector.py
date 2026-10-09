# backend/core/strategy_selector.py
# 检索策略选择器（对齐课程四策略）：
#   直接检索 DIRECT / 假设问题检索 HYDE / 子查询检索 SUBQUERY / 回溯问题检索 BACKTRACK
# 由 LLM 对专业问题四选一；任何异常都兜底为 DIRECT（检索链路永远可用）。
# 规则快判作为前置：明确的短事实问题不花 LLM 调用，直接 DIRECT。

from backend.agents.qa.prompts import STRATEGY_SELECT_PROMPT
from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)

DIRECT = "DIRECT"
HYDE = "HYDE"
SUBQUERY = "SUBQUERY"
BACKTRACK = "BACKTRACK"

# LLM 返回中文策略名 -> 内部常量
_NAME_MAP = [
    ("回溯", BACKTRACK),
    ("子查询", SUBQUERY),
    ("假设", HYDE),
    ("hyde", HYDE),
    ("直接", DIRECT),
]

# 明确事实型信号：含具体业务对象 + 短问题，无需 LLM 选策略
_FACT_KEYWORDS = (
    "多少钱", "价格", "价目", "一斤", "半斤", "克重", "规格", "批发价", "零售价",
    "几点", "营业时间", "电话", "地址", "几个品牌", "几家", "多少家",
)


def _normalize(raw: str) -> str:
    text = (raw or "").strip().lower()
    for key, strategy in _NAME_MAP:
        if key in text:
            return strategy
    return DIRECT


def _rule_fast_path(query: str) -> str | None:
    """规则快判：短事实问题直接 DIRECT；明显指代型模糊问题直接 HYDE。"""
    q = query.strip()
    if len(q) <= 12 and any(kw in q for kw in _FACT_KEYWORDS):
        return DIRECT
    vague_hints = ("没懂", "不懂", "不太懂", "啥意思", "什么意思", "看不懂", "再讲讲")
    if len(q) <= 10 and any(kw in q for kw in vague_hints):
        return HYDE
    return None


async def aselect_strategy(query: str) -> str:
    """
    异步选择检索策略，返回 DIRECT / HYDE / SUBQUERY / BACKTRACK。
    1) 规则快判命中 -> 直接返回，省一次 LLM 调用
    2) LLM 四选一 -> 名称归一化
    3) 任何异常 -> DIRECT（绝不因选策略失败阻断问答）
    """
    fast = _rule_fast_path(query)
    if fast:
        logger.info("strategy.rule_fast_path", query=query[:30], strategy=fast)
        return fast

    try:
        llm = get_llm("intent", temperature=0.1)
        raw = await llm.achat(
            [{"role": "system",
              "content": "你是一个有用的助手，能够根据用户输入的 Prompt 严格执行并返回可靠的结果"},
             {"role": "user",
              "content": STRATEGY_SELECT_PROMPT.format(query=query)}],
            max_tokens=64,
        )
        strategy = _normalize(raw)
        logger.info("strategy.llm_selected", query=query[:30],
                    raw=raw.strip()[:30], strategy=strategy)
        return strategy
    except Exception as e:
        logger.warning("strategy.select_failed_default_direct", error=str(e)[:200])
        return DIRECT
