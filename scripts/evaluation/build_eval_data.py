# 该脚本用于: 跑一遍新架构的 RAG 通道, 生成 RAGAS 需要的评估数据 (rag_evaluate_data.json).
# 为什么需要这一步: RAGAS 要的是"问题 + 系统生成的答案 + 检索到的上下文 + 标准答案"四元组,
# 后两者必须由真实链路跑出来 —— 这正是评估"检索好不好、生成有没有幻觉"的依据.
# 链路与生产一致(backend/agents/qa/service.py 的四策略检索 + RAG 生成):
#   BGE-M3 子块混合检索 -> 父块回升 -> bge-reranker 父块精排
#   -> 拼【参考i】上下文 -> SYSTEM_PROMPT+RAG_ANSWER_PROMPT 生成(非流式).

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # 项目根

from backend.agents.qa.prompts import RAG_ANSWER_PROMPT, SYSTEM_PROMPT
from backend.config import get_settings
from backend.core.llm_factory import get_llm
from backend.core.logger import configure_logging, get_logger
from backend.core.query_classifier import classify_query
from backend.core.reranker import retrieve

configure_logging()
logger = get_logger("build_eval_data")

HERE = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(HERE, "eval_questions.json")
OUT_FILE = os.path.join(HERE, "rag_evaluate_data.json")


def main():
    settings = get_settings()
    questions = json.load(open(IN_FILE, encoding="utf-8"))
    llm = get_llm("qa", temperature=settings.llm_temperature, streaming=False)

    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        t0 = time.time()
        # 1. 意图分类（仅日志，与生产 classify_query_node 同一函数）
        label, scores = classify_query(q)
        # 2. 与生产同一检索入口：混合召回 + 精排
        docs, confidence = retrieve(
            q, settings.default_tenant_id, None,
            settings.recall_top_k, settings.rerank_top_k,
            settings.min_relevant_score)
        contexts = [d.content for d in docs]
        # 3. 与 service.py RAG 生成完全相同的拼装方式
        context_text = "\n\n".join(f"【参考{j}】\n{c}"
                                   for j, c in enumerate(contexts, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RAG_ANSWER_PROMPT.format(
                context=context_text, query=q,
                phone=settings.customer_service_phone)},
        ]
        answer = llm.chat(messages).strip()
        results.append({
            "question": q,
            "answer": answer,
            "context": context_text,    # 字符串格式（兼容旧版）
            "context_list": contexts,   # ragas 要求 retrieved_contexts 必须是 list
            "ground_truth": item["ground_truth"],
        })
        logger.info(f"[{i}/{len(questions)}] {q} -> 分类:{label} "
                    f"置信:{confidence:.3f} 上下文:{len(contexts)}块 "
                    f"答案:{len(answer)}字 耗时:{time.time()-t0:.1f}s")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评估数据已生成: {OUT_FILE}  共 {len(results)} 条")


if __name__ == "__main__":
    main()
