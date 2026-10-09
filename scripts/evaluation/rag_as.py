# 该脚本用于: 通过RAGAS框架 对RAG系统做评分, 主要考察: 检索部分 和 生成部分.
# 一共有4个指标, 分别是:
#   生成部分: 忠实度(faithfulness), 答案相关性(answer_relevancy).
#   检索部分: 上下文召回率(context_recall), 上下文精确率(context_precision).
#
# 【熟食版适配说明】
#   课件原版用本地 Ollama(qwen2.5:7b) 做评估模型 —— 本机未部署 Ollama, 改为使用
#   项目同一套 OpenAI 兼容端点（.env.local，与生产 LLM 同一配置）:
#     评估 LLM   : conf.llm_model（自动跟随 .env.local）
#     评估嵌入   : EMBEDDING_MODEL（要求该端点提供 /embeddings）
#   注: RAGAS 0.2.6 的 evaluate 只接受 LangChain 的 ChatModel/Embeddings 接口,
#       因此离线评估保留 langchain-openai 适配（生产链路不含任何 langchain）。
#   数据来自 build_eval_data.py（真实跑一遍链路生成 answer 与 context）.

# todo 1.导包
import json
import os
import sys

from ragas import evaluate                     # RAGAS 评估入口
from ragas.metrics import (                    # 4 个指标
    faithfulness,           # 忠实度: 答案是否只基于上下文（幻觉检测）
    answer_relevancy,       # 答案相关性: 答案是否切题
    context_precision,      # 上下文精确度: 检索到的内容有多少是真正有用的
    context_recall          # 上下文召回率: 标准答案所需信息有没有被检索到
)
from datasets import Dataset                   # RAGAS 需要的数据格式
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # 用 OpenAI 兼容接口接端点模型

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import get_settings

conf = get_settings()

HERE = os.path.dirname(os.path.abspath(__file__))

# todo 2. 加载评估数据集（由 build_eval_data.py 跑真实链路生成）
with open(os.path.join(HERE, 'rag_evaluate_data.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'加载的评估样本数量: {len(data)}')

# todo 3. 转换数据格式 -> RAGAS 要求的 Dataset
eval_data = {
    'question': [item['question'] for item in data],
    'answer': [item['answer'] for item in data],          # 系统生成的答案
    'contexts': [item['context_list'] for item in data],  # 检索到的上下文（必须是分块列表）
    'ground_truth': [item['ground_truth'] for item in data],   # 人工标准答案
}
dataset = Dataset.from_dict(eval_data)
print(f'转换后的数据集: {dataset}')

# todo 4. 配置评估模型（端点模型, 不再依赖 Ollama）
llm = ChatOpenAI(
    model=conf.llm_model,   # qwen3.7-plus（flash 免费额度已尽）
    api_key=conf.llm_api_key,
    base_url=conf.llm_base_url,
    temperature=0,                     # 评估要稳定, 不要发挥
    timeout=120,
)
# 注意: DeepSeek 官方端点不提供 /embeddings；若 LLM 端点无 embedding 能力,
#       需把 .env.local 临时切到支持 embedding 的 OpenAI 兼容端点再跑本脚本。
EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-v3")
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=conf.llm_api_key,
    base_url=conf.llm_base_url,
    check_embedding_ctx_length=False,  # 非 OpenAI 官方模型, 关掉 tiktoken 分块逻辑
)

# answer_relevancy 默认要模型一次生成 3 个问题(strictness=3), 而很多兼容端点不支持 n>1,
# 这里降到 1, 保证能跑通（代价是该指标的鲁棒性略降, 报告里注明）.
answer_relevancy.strictness = 1

# todo 5. 执行评估
print('\n开始评估（约 10-15 分钟, 要做上百次模型调用）...')
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
)

# todo 6. 输出与保存
print(f'\nRAGAS评估结果: {result}')
out_csv = os.path.join(HERE, 'ragas_result.csv')
result.to_pandas().to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'逐题明细已保存: {out_csv}')

# 汇总指标存 JSON（方便写进 README / 面试材料）
# ragas 的 EvaluationResult 没有 .items(), 从 to_pandas() 的数值列算均值
summary = {c: float(result.to_pandas()[c].mean()) for c in result.to_pandas().columns
           if result.to_pandas()[c].dtype.kind in 'fi'}
with open(os.path.join(HERE, 'ragas_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f'汇总指标已保存: {os.path.join(HERE, "ragas_summary.json")}')
print(json.dumps(summary, ensure_ascii=False, indent=2))
