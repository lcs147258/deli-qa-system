# DeliRAG · 智慧出品熟食连锁智能问答系统

基于 **FAQ 热路径 + 轻量分层 RAG 流水线** 的熟食连锁企业知识问答系统，
面向**加盟商与终端顾客**两类外部用户（门店内部不使用），
回答产品规格价格、加盟与批发政策、食品安全与储存、冷链配送等问题。

> **第二版（pgvector-lite，2026-10）**：检索层从 Milvus + etcd + MinIO 三件套
> 换成 **ParadeDB 单容器**（pgvector 稠密 + pg_search 稀疏 + SQL 内 RRF 融合），
> embedding / reranker 支持**本地模型与 SiliconFlow 云 API 双后端**，整机可在
> 4GB 内存的云服务器上以纯 CPU 运行。编排层已去除 LangGraph，改为 `QAService` 显式流水线。
> 课程原始版本（`rag_qa/` + `mysql_qa/` + `new_main.py`）已整体移除，
> 旧文档归档在 `_归档_旧版本文档/`（确认无误后可整目录删除）。
> 面试速查见 `docs/面试速查_DeliRAG.md`。

---

## 一、系统架构

```mermaid
flowchart TD
    U(["用户 / 浏览器"]) --> API["FastAPI · backend/main.py<br/>REST + WebSocket 流式"]

    API --> FAQ["① faq_check<br/>BM25 + jieba + softmax"]
    FAQ -->|"≥ 0.85 命中"| HIT(["直接返回<br/>零 LLM 调用"])
    FAQ -->|"未命中"| CLS

    subgraph G["问答流水线 · backend/agents/qa/service.py"]
        CLS["② 意图分类<br/>规则 → BERT 二分类 → LLM 策略选择"]
        CLS -->|GENERAL| GEN["闲聊直答，不检索"]
        CLS -->|SPECIALIZED + 策略| STR["③ 四策略<br/>DIRECT / HYDE / SUBQUERY / BACKTRACK"]
        STR --> RET

        RET["④ retrieve · 父子块双路召回<br/>pgvector HNSW 稠密 + pg_search BM25 稀疏<br/>SQL 内 RRF 融合 → bge-reranker 精排"]
        RET -->|"top1 ≥ 0.45"| RAG["⑤ generate_rag<br/>严格依据知识库 + 来源标注"]
        RET -->|"top1 < 0.45"| FB["⑥ 固定话术转人工（不调 LLM）"]
        FB --> PEND["⑦ 入待补充队列 knowledge_pending"]
    end

    HIT --> MEM["⑧ save_memory<br/>MySQL 会话历史"]
    GEN --> MEM
    RAG --> MEM
    PEND --> MEM
    MEM --> OUT(["返回答案"])

    subgraph OFF["离线支撑 · scripts/"]
        ING["build_knowledge_base.py<br/>加载 → 父子分块 1200/300/64 → Contextual RAG → 编码"]
        PSTORE[("ParadeDB 单容器 · kb_chunks<br/>HNSW(vector) + paradedb BM25 索引")]
        ING --> PSTORE
    end
    PSTORE -.->|"在线检索"| RET
```

**设计取向三句话**：成本上省（FAQ 命中零大模型调用）、安全上收（低置信拒答不硬编）、
检索上分策略（不同问题类型走不同检索路径，不是一个检索法打天下）。

---

## 二、技术栈

| 层 | 技术 |
|---|---|
| 编排 | **`QAService` 显式流水线**（去 LangGraph；四策略 DIRECT / HYDE / SUBQUERY / BACKTRACK） |
| 接入 | FastAPI + Uvicorn + WebSocket 流式（`start / token / end`） |
| 大模型 | **DeepSeek `deepseek-chat`**（OpenAI 兼容端点，`.env.local` 可换厂商） |
| FAQ 热路径 | MySQL 8（`faq` 表）+ Redis 7（Cache-Aside + embedding 内容哈希缓存，`deli:` 前缀隔离）+ rank_bm25 + jieba 领域词 |
| 向量库 | **ParadeDB 0.25.11（PostgreSQL 17）单容器**：`pgvector` 0.8.4 稠密向量（HNSW，cosine）+ `pg_search` 0.25.11 稀疏检索（Lindera 中文分词 BM25），**双路 RRF 融合在一条 SQL 内完成**（k=60） |
| Embedding | **双后端可切换**：本地 `bge-small-zh-v1.5`（512 维，sentence-transformers）/ SiliconFlow `bge-m3`（1024 维，免费额度） |
| 精排 | **双后端可切换**：本地 `bge-reranker-large`（CrossEncoder）/ SiliconFlow `bge-reranker-v2-m3`（HTTP 失败自动 RRF 分数兜底降级） |
| 意图分类 | 规则层 + **bert-base-chinese 微调二分类**（`models/classifier/bert_query_classifier`） |
| 文档入库 | md 按标题切 → **父子分块 1200/300/64**（子块携带父块全文冗余）→ 可选 **Contextual RAG** |
| 中间件 | Docker Compose 三容器：**ParadeDB + Redis + MySQL**（旧版 etcd / MinIO / Milvus 已移除） |

**端口**：ParadeDB/PostgreSQL `5434` · MySQL `3307` · Redis `6380`（6379 被别的项目占用）· Web `8082` · MCP `8001`

**资源画像**：本地模式全链路 CPU 可跑（embedding 仅 95MB / 512 维）；切 SiliconFlow 后端后
服务器只需跑分类器小模型，embedding 与 rerank 全部走云 API，适配 2核4GB 轻量服务器。

---

## 三、目录结构

```
deli_qa_system/
├── backend/                       # 应用代码（分层）
│   ├── main.py                    # FastAPI 入口：路由装配 + 启动预热 FAQ
│   ├── config.py                  # pydantic-settings 配置中心（.env.local 优先于 OS env）
│   ├── dependencies.py            # 依赖装配：FAQ / MySQL 单例
│   ├── api/v1/qa.py               # 路由：会话 / 查询 / WS 流式 / 历史 / 健康检查
│   ├── agents/qa/                 # ★ 问答流水线
│   │   ├── service.py             # QAService：分类 → 策略 → 检索 → 闸门 → 生成（astream_answer）
│   │   └── prompts.py             # 提示词（含食品安全红线）
│   ├── core/
│   │   ├── knowledge_base.py      # psycopg3 PG 客户端：双路召回 SQL / 父块提升 / upsert（单例）
│   │   ├── embeddings.py          # ★ embedding 双后端（Local bge-small / SiliconFlow bge-m3）+ 两层缓存
│   │   ├── fusion.py              # ★ RRF / WeightedRRF / CombMNZ 融合器
│   │   ├── reranker.py            # reranker 双后端 + retrieve() 一体化入口（含云 API 失败降级）
│   │   ├── query_classifier.py    # BERT 二分类（推理）
│   │   ├── strategy_selector.py   # LLM 四策略选择（失败默认 DIRECT）
│   │   ├── llm_factory.py         # 模型工厂（按用途缓存，禁止散落 init_chat_model）
│   │   ├── memory.py              # 滑动窗口会话裁剪
│   │   └── pending.py             # 低置信问题入待补充队列
│   ├── faq/                       # FAQ 通道：mysql_client / redis_client / bm25_search / preprocess
│   ├── ingest/                    # 文档加载（含 OCR）+ 切分
│   └── mcp/knowledge_base_server.py  # 把检索暴露为 MCP 工具（Streamable HTTP）
├── scripts/
│   ├── init_pg.py                 # ★ 建扩展 + kb_chunks 表与索引（HNSW + paradedb BM25）
│   ├── build_knowledge_base.py    # 文档入库全流程（含 docs/ 安全拦截）
│   ├── init_faq_data.py           # FAQ CSV → MySQL（增量同步）
│   ├── train_classifier.py        # 重训 BERT 分类器
│   └── evaluation/                # RAGAS 评估集、跑分脚本、评估报告
├── data/                          # ★ 唯一入库扫描根
│   ├── faq/deli_faq.csv           # 96 条问答对（+ 来源台账）
│   └── {product,safety,ops,franchise,supply}_data/   # 五类知识文档
├── models/classifier/bert_query_classifier/   # 微调分类器权重
├── training/                      # 分类器训练集与生成提示词
├── docs/                          # 业务素材：访谈提纲、内部配方（★永不入库）、设计规范
└── static/                        # 前端页面
```

---

## 四、快速开始

```bash
# 0. 环境：Python 3.11（CUDA 可选，纯 CPU 也能跑）· 依赖见 requirements-win.txt
#    .env.local 需填好 LLM_API_KEY / LLM_BASE_URL；用云 embedding/rerank 再填 SF_API_KEY

# 1. 中间件（ParadeDB + Redis + MySQL 三容器）
docker compose up -d

# 2. 建表与索引（pgvector HNSW + pg_search Lindera BM25）
python scripts/init_pg.py --force

# 3. 导入 FAQ（幂等，只补缺失的问题）
python scripts/init_faq_data.py
#    ⚠️ 新增 FAQ 后必须清 Redis 语料缓存，否则 BM25 还用旧语料：
#    redis-cli -p 6380 DEL deli:qa_original_questions deli:qa_tokenized_questions

# 4. 知识库入库（--no-context 可跳过 Contextual RAG 加速）
python scripts/build_knowledge_base.py
#    当前语料：82 个子块（franchise 9 / ops 18 / product 28 / safety 10 / supply 17）

# 5. 训练意图分类器（可选，权重已随项目提供）
python scripts/train_classifier.py --data training/deli_classifier_500.json --epochs 3

# 6. 启动 Web 服务
python 启动服务.py                                        # 封装脚本（含环境自检）
#    或：python -m uvicorn backend.main:app --host 127.0.0.1 --port 8082

# 7. 环境自检
python verify_env.py
```

打开 `http://127.0.0.1:8082`。可选启动 MCP 检索服务：`python -m backend.mcp.knowledge_base_server`

**切云后端**：`.env.local` 设 `RERANK_BACKEND=siliconflow` / `EMBEDDING_BACKEND=siliconflow`。
注意 embedding 换 bge-m3 后维度变为 1024，必须重建表（`init_pg.py --force`）并全量重新入库。

---

## 五、关键参数（`backend/config.py`）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `faq_bm25_threshold` | **0.85** | FAQ 命中阈值（精确率优先：误报是错误答案，漏报只是多一次 LLM） |
| `recall_candidate_limit` | 50 | 每路在 PG 内先取的候选池大小，RRF 融合后再截断 |
| `recall_top_k` / `rerank_top_k` | 10 / 3 | 融合后召回数 / 精排后保留数 |
| `fusion_strategy` / `rrf_k` | `rrf` / 60 | 双路融合策略（可选 weighted_rrf / combmnz）与 RRF 常数 |
| `fusion_dense_weight` / `fusion_sparse_weight` | 0.6 / 0.4 | weighted_rrf 下的两路权重 |
| `min_relevant_score` | 0.05 | 精排分低于此值视为召回噪声丢弃 |
| `high_confidence_threshold` | **0.45** | 精排 top1 低于此值 → 固定话术转人工，**不生成**（换云 reranker 需重新标定） |
| `parent_chunk_size` / `child_chunk_size` / `chunk_overlap` | 1200 / 300 / 64 | 父子分块：子块精准命中，命中后提升到父块喂 LLM |
| `embedding_backend` / `embedding_dim` | local / 512 | local=bge-small-zh 512；siliconflow=bge-m3 1024（改维度必须重建库） |
| `rerank_backend` | local | local=bge-reranker-large；siliconflow=bge-reranker-v2-m3 |
| `contextual_rag` | true | 入库时给每个 chunk 加一句上下文描述 |
| `valid_sources` | product/safety/ops/franchise/supply | 五类知识过滤 |

---

## 六、数据来源与合规

| 数据 | 来源 | 状态 |
|---|---|---|
| 产品规格/价格/保质期 | 业务方价目表与产品资料 | 真实数据 |
| 食品安全/冷链/操作规范 | GB 31654-2021、GB 31605-2020、《餐饮服务食品安全操作规范》**要点整理稿** | 公开标准二次整理 |
| FAQ 96 条 | 业务方材料 + 行业通行做法 | 每条来源见 `data/faq/deli_faq_来源台账.md` |
| 分类器训练集 | LLM 按领域提示词生成 + 两轮人工质检 | 合成数据，已标注 |

**三条铁律**：

1. **内部配方绝不入库**：现拌凉菜精确配方是商业机密，归档在 `docs/内部配方_不入知识库.md`。
   `build_knowledge_base.py` 显式拦截 `docs/` 路径（`if "docs" in resolved.parts: raise`），
   物理隔离 —— 顾客问"你们黄瓜怎么拌的"套不出秘方。
2. **食品安全零幻觉**：保质期、温度、留样义务绝不能靠模型记忆回答。
   低置信走固定话术 + 人工电话，绝不硬答。
3. **价格与保质期标注时效**：答案为销售指导价 / 行业参考值，注明"以门店当日公示为准"。

---

## 七、实测数据

**第二版回归（2026-10-09，本地 bge-small-zh + bge-reranker-large，82 子块）**：
四道标准题全部符合预期 —— 猪头肉价格 0.9557 命中、冷藏温度 0.9677 命中、
酱牛腱配方红线 0.1024 正确转人工、公司品牌 0.4687 过闸命中品牌矩阵。
环境自检 53/54 项通过（唯一失败项为 LLM 账户余额不足 402，充值后复测）。

**第一版 RAGAS 基线（Milvus + BGE-M3，15 题真实链路，评估器 LLM，第二版待充值后复跑）**：

| 指标 | 全量 15 题 | 有检索语料的 13 题 |
|---|---|---|
| faithfulness 忠实度 | 0.730 | **0.843** |
| answer_relevancy 相关性 | 0.645 | 0.745 |
| context_precision 精确率 | 0.655 | 0.679 |
| context_recall 召回率 | 0.678 | 0.757 |

- 核心 7 题（价格、保质期、加热、冷链、健康证、采购）**faithfulness 全是 1.00**，相关性 0.94+
- 全量被拉低是因为 3 道题知识库无对应语料，系统**正确转人工**被指标记 0 —— 这是红线生效，不是故障
- 逐题明细见 `scripts/evaluation/评估报告_RAGAS.md`

---

## 八、已知限制与后续规划

**限制（诚实清单）**：

1. `data/franchise_data/` 语料较少，部分加盟类问题仍会转人工
2. **长价目表切分不友好**：300 字符子块切大表可能切散，拖低 context_recall
3. BM25 是词汇匹配：语义相近但意图不同的问题可能误命中（靠 0.85 高阈值压住）
4. 拒答闸门只看精排 top1 分数，未校验"检回内容是否含问题关键实体"
5. 荤菜保质期为行业估计值，待业务方确认
6. SiliconFlow 双后端代码已就绪，bge-m3 1024 维链路与 0.45 阈值标定待 API key 到位后联调

**规划**：

- [ ] SiliconFlow 后端联调与阈值重标定（v2-m3 分数尺度不同，0.45 不能直接沿用）
- [ ] 4GB 云服务器部署验证（CPU torch + 2GB swap + 阿里云镜像加速）
- [ ] 表格类文档专门切分（按行建 chunk）
- [ ] 检索相关性闸门（关键实体校验）
- [ ] FAQ 语料扩充至 150+（问题面见 `docs/竞品问题挖掘清单.md`）
- [ ] **角色权限（上线阻塞项）**：加盟商可见批发价与加盟政策，终端顾客不可见；
      当前系统**无任何鉴权**，两类用户共用同一套知识库
- [ ] 知识时效性治理：价格/政策类文档的版本与生效日期管理

---

## 九、工程侧要点（值得一看的实现）

- **融合下推到一条 SQL**：dense CTE（HNSW `<=>` 余弦）+ sparse CTE（pg_search）
  FULL OUTER JOIN 后按 `1/(60+rank)` 算 RRF，双路 ANN 都在库内完成，只回传 top-k
- **pg_search 0.25 的真实语法坑**（与网上旧文档完全不同）：
  中文 BM25 用 `content::pdb.lindera(chinese)`（schema 是 **pdb** 不是 paradedb），
  建索引走 `CREATE INDEX ... USING paradedb`（已无 `create_bm25` 存储过程）；
  `@@@ '冷藏温度'` 默认按**相邻短语**匹配会 0 命中，必须 `boolean(should => term(...)[])`
  做 bag-of-words OR，查询词要在库内用同一 Lindera 分词器先切
- **云后端降级**：SiliconFlow rerank 超时/429 时自动用 RRF 排名映射兜底分，
  embedding 失败则降级为纯 BM25 单路召回，不向前端抛技术错误
- **embedding 两层缓存**：进程内 dict + Redis（key 为内容 sha256），换后端不污染旧缓存
- **同步推理不阻塞事件循环**：编码、rerank、BM25、MySQL 全走 `asyncio.to_thread`
- **全链路降级**：MySQL / Redis / ParadeDB / LLM 任一不可用都有降级路径
- **入库幂等**：`ON CONFLICT (id) DO UPDATE`，子块冗余携带 `parent_content` 支撑父块提升
- **两个真实坑**：BGE 权重必须放纯 ASCII 路径（中文路径 sentencepiece 读不了）；
  embedding 换模型维度变化（512→1024）必须重建表并全量重灌，向量列维度不兼容

---

## 十、相关文档

- `docs/面试速查_DeliRAG.md` — 面试拆解（主链路 / 追问应答 / 翻车点清单）
- `scripts/evaluation/评估报告_RAGAS.md` — 评估逐题明细（第一版基线）
- `docs/DESIGN-SPEC.md` / `docs/REJECT-LIST.md` — 前端设计规范与否决清单
- `.trae/documents/deli_rag_refactor_plan.md` — 第一版重构的规划与决策记录
