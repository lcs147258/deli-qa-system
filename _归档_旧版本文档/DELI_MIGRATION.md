# DeliRAG 迁移手册（教育版 → 熟食连锁企业问答）

> 本项目由 `my_qa_system`（教育版）按课程要求改造而来，架构、14 个手写文件练习完全不变，
> 领域换成"熟食工厂 + 自营批发市场门店 + 连锁卤味门店"。
> 本手册是熟食版的**总账**：改了什么、还差什么、坑在哪。教育版细节看 `my_qa_system` 的文档。

---

## 一、两个项目的关系

| | 教育版 `my_qa_system` | 熟食版 `deli_qa_system`（本项目） |
|---|---|---|
| 定位 | 课程原始交付，**保留不动**，作对照参考 | 按老师要求改造后的交付 |
| 架构 | 双通道：FAQ(MySQL+Redis+BM25) → RAG(Milvus+BGE-M3+重排) | 完全相同，代码层不迁移逻辑 |
| 14 个手写文件 | 空壳（另做） | 空壳（**在这里写**，领域点见第四节） |
| MySQL 库 | `subjects_kg`（表 `jpkb`） | **`deli_qa`**（表 `faq`） |
| Milvus | 库 `itcast`，collection `edurag_final` | **库 `deli`**，collection **`deli_final`** |
| 中间件容器 | edurag-* 5 个容器 | **复用同一批容器**，只换库名 |

## 二、M0 已完成（2026-09-24）

1. **fork**：`my_qa_system` → `deli_qa_system`，排除了 models/volumes/__pycache__/.idea/logs。
2. **模型处理**（共 6.4G，全部零重复存储）：
   - `bert_query_classifier`（391M）：**真实复制**。M3 重训时新模型写在这里，
     不会穿透覆盖教育版。
   - 其余 4 个基础模型（bge-m3 / bge-reranker-large / bert-base-chinese /
     nlp_bert_document-segmentation）：**junction** 指向教育版真实目录，两项目共用同一份文件。
3. **config.ini 已改为熟食版**：`deli_qa` 库、`deli`/`deli_final`、
   `valid_sources = ["product","safety","ops","franchise","supply"]`，注释里写明共用中间件的口径。
4. **new_main.py 补上了 httpx keep-alive 修复**（`max_keepalive_connections=0`）。
   ⚠️ **重要发现**：教育版 `my_qa_system/new_main.py` 里这个修复**丢了**（只在 `E:\edurag_ref` 有），
   教育版闲置后首次请求会复发 40 秒卡顿——需要把同样的 diff 应用回教育版（见第五节 D 档）。
5. **教育版数据已从本项目清除**：FAQ CSV、ai_data 文档、BERT 训练集、RAGAS 评估集，
   换成下面的熟食版数据结构。

## 三、知识体系：5 类目录（RAG 通道的数据骨架）

`config.ini` 的 `valid_sources` 与 `rag_qa/data/` 目录**一一对应**——
入库时 source 元数据 = 目录名去掉 `_data` 后缀（document_processor 的规则）。

| 目录 | 中文 | 内容来源 | 当前状态 |
|---|---|---|---|
| `product_data/` | 产品知识 | **必须来自伯父**（产品单/价格/保质期） | ✅ 已有《产品与规格手册》（xlsx+价目表+折页整理，2026-09-27） |
| `safety_data/` | 食品安全 | 公开国标 + 公司制度 | ✅ 已有 GB 31654-2021 整理稿 |
| `ops_data/` | 门店运营 | 公司 SOP + 国标依据 | ✅ 已有《餐饮服务食品安全操作规范》整理稿；SOP 待访谈 |
| `franchise_data/` | 加盟与批发 | **必须来自伯父** | ⬜ 待访谈 |
| `supply_data/` | 供应链与仓储 | 公开国标 + 公司做法 | ✅ 已有 GB 31605-2020 整理稿；公司做法待访谈 |

> ⚠️ **内部配方不入库**（2026-09-27 决定）：xlsx Sheet2/3 的现拌凉菜精确配方是商业机密，
> 已归档到 `docs/内部配方_不入知识库.md`，**绝不放进 `rag_qa/data/`**——
> 否则顾客问"你们家的黄瓜怎么拌的"就能套出秘方。将来做店员/加盟商角色权限时再议。

**铁律：目录名前缀必须与 valid_sources 完全一致**，拼写不一致时 source 元数据对不上，
Milvus 过滤**静默失效**（不报错）。

## 四、14 个手写文件里的领域点（其余 10 个文件照 DESIGN_SPEC 原规格写）

| 文件 | 熟食版差异 |
|---|---|
| `mysql_client.py` | 表名 `faq`，字段 `category / question / answer`；CSV 表头"分类,问题,答案" |
| `strategy_selector.py` | 策略提示词里的 4 个 few-shot 示例换成熟食问题（原为"AI学费/JAVA大纲"） |
| `preprocess.py`（可选加分） | jieba 加载自定义词典：卤味/酱鸭/桶子鸡/效期/报损/冷链/真空包装……提升 BM25 分词 |
| 其余 11 个 | 与领域无关，按 `my_qa_system/DESIGN_SPEC.md` 原规格写 |

## 五、改动总账（四档）

**A. 配置（✅ 已完成）**：config.ini 如第二节所述。

**B. 数据（部分完成）**：

| 数据 | 落点 | 状态 |
|---|---|---|
| FAQ 问答对 | `mysql_qa/data/deli_faq.csv` | ✅ **v0.1 已有 40 条**（28 产品 + 7 食品安全 + 5 兜底），来源见同目录《deli_faq_来源台账.md》；缺口：荤菜保质期、营业时间、加盟批发细节，访谈后扩到 150+ |
| 知识库文档×5 类 | `rag_qa/data/*_data/` | ✅ 3 份国标整理稿 + **产品与规格手册**（2026-09-27）；franchise 与公司 SOP 待访谈成稿 |
| BERT 训练集 500 条 | `rag_qa/classify_data/deli_classifier_500.json` | ✅ **已生成并两轮质检**（150 通用 + 350 专业，6 子主题各 ≥50，8 条边界样本全对，零重复）。修复了「你们」捷径：通用知识补入 7 条含"你们"的闲聊、专业咨询 180 条去前缀改写——否则分类器会学成"含你们=专业咨询"的假规则 |
| RAGAS 评估集与跑分 | `rag_qa/rag_assessment/` | ✅ **已完成**：15 题（含 2 道诚实性测试）经真实链路生成 → RAGAS 四指标：**precision 0.933 / recall 0.800 / faithfulness 0.803（排除拒答 0.889）/ relevancy 0.796（排除拒答 0.920）**。报告见 `评估报告_RAGAS.md`。评估暴露两个改进项：长表格分块、检索相关性闸门 |

**C. 手写文件领域点**：见第四节，写的时候直接按熟食版写。

**D. 已提供文件的文案编辑（我给 diff，你动手改，逻辑不动）**：

| 文件 | 位置 | 改什么 |
|---|---|---|
| `app.py` | 60–73 行 | 4 条问候语："黑马程序员/IT教育" → 品牌话术 |
| `static/index.html` | 6、549 行 | 标题"传智教育EduRAG智慧问答系统" → 品牌名（定了叫什么告诉我） |
| `static/index.html` | 快捷回复按钮 | 3 个 AI 问题 → 熟食问题（推荐："卤味常温能放多久？""怎么加盟？""今天有什么新品？"） |
| `static/index.html` | 595 行 | label"学科类别" → "知识分类" |
| `new_main.py` | 320–342 行 | CLI 文案"学科类别" → "知识分类"（低优先） |
| `old_main.py` | 82、148 行 | "AI课程"示例、"高薪就业"退出语（低优先） |
| `App.jsx` | — | 与后端路由不匹配，**不修**；只删 153 行残留的"喧嚣"二字 |
| `my_qa_system/new_main.py`（教育版） | 44 行 | **把 httpx 修复补回去**（diff 问我要） |

## 六、里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M0 准备** ✅ | fork + 配置 + 国标整理稿 + 访谈提纲 + 竞品清单 + 生成提示词 | 本文档 + `docs/` 两个文档 + `data/` 3 份整理稿 |
| **M1 FAQ 通道** ✅ | **代码已由 AI 代写并验收通过（2026-09-27）**：base/config.py、base/logger.py、mysql_qa 5 件 + init_faq_data.py。建库 deli_qa + 表 faq + 导入 56 条 + BM25 端到端测试 | 命中（猪头肉 0.906、你们鸡来源 0.994）、Redis 答案缓存二次命中、未命中正确降级 need_rag=True、日志 handler=2（守卫生效）、缓存键带 deli: 前缀与教育版隔离 |

> **M1 实测发现（面试素材）**：softmax 全量归一化在"同类问题多"的语料上会稀释概率——
> "五香料汁多少钱"与库内问题几乎一字不差，仅得 0.583（56 条里 20+ 条都含"多少钱"）。
> 实测过 top5 softmax（0.937 可救回）但会把"鸡胗新鲜吗"也放行（0.922，答非所问）。
> **最终维持全量 softmax + 0.85**：漏报只是多一次 LLM 调用且 RAG 能答对，
> 误报是免费但错误的答案——FAQ 通道必须精确率优先。提高命中率的正道是 FAQ 语料运营
> （每个意图多写几种用户真实问法），而非调阈值。
| **M2 RAG 通道** ✅ | **代码已代写并验收（2026-09-27）**：prompts.py、document_processor.py、vector_store.py、rag_qa/main.py。建 Milvus `deli` 库 + `deli_final` 集合，入库 **36 个子块**（product 15 / safety 8 / ops 7 / supply 6） | 检索验收：无过滤返回相关父块；`source='safety'` 过滤精确生效；价格查询命中产品手册；集成链路 Q2 返回带引用答案（60℃/8℃ 危险温度区间） |
| **M3 智能层** ✅ | **代码已代写并验收（2026-09-27）**：query_classifier.py、strategy_selector.py（熟食 few-shot 示例）。用 500 条熟食语料重训 BERT：**10.5 秒训完（GPU）、验证集准确率 0.97、边界样本 7/8**（唯一失手：`帮我想个朋友圈卖卤味的文案` 被判专业咨询——最难的营销文案边界，后续补样本） | 策略选择器三查三中（直接/子查询/HyDE）；集成：Q1 FAQ 命中、Q2 子查询路径返回带引用答案 |
| **M4 收尾** | rag_qa/main.py + 文案 + RAGAS 评估集 + 可选 jieba 词典 | 全链路演示 + 四指标数字 |

## 七、坑清单（每一条都写进对应验收）

1. **Redis 旧缓存**：M1 导入新 FAQ 后必须删 `qa_original_questions` / `qa_tokenized_questions`
   两个键（与教育版共用 Redis），否则 BM25 语料还是 467 道教育题。
2. **目录命名**：`{source}_data` 与 valid_sources 不一致 → 过滤静默失效（见第三节铁律）。
3. **食品安全零幻觉**：保质期/储存温度/留样义务绝不能靠模型记忆回答。rag_prompt 的
   "信息不足转人工 {phone}" 兜底必须保留；RAGAS faithfulness 重点抽测 safety 类问题。
4. **留样别套错**：普通零售门店一般不强制留样，FAQ 里"要不要留样"的答案以公司口径为准，
   不要把集体用餐的要求写成门店义务。
5. **知识时效性**：价格/加盟政策写进文档时标注"以门店当日公示为准 / 以公司最新通知为准"。
6. **模型路径**：bge 系模型走 ASCII 路径（`E:\edurag_models` 或本项目 junction），
   从中文路径直接加载会踩 sentencepiece 的坑（教育版已实测）。分类器用本项目
   `rag_qa/models/bert_query_classifier`（重训后的新模型），别用 `E:\edurag_models` 里那个。
7. C 盘空间保持 50G 以上（Docker 数据在 C 盘）。
8. **【M2 实测 bug】查询侧 fp16 未转 float32**：GPU+fp16 下 BGE-M3 查询向量是 float16，
   入库侧参考实现转了 float32、查询侧没转，Milvus 报
   `vector type must be the same, field VECTOR_FLOAT, search info VECTOR_FLOAT16`。
   读写的类型转换必须对称。（已修复: vector_store.hybrid_search_with_rerank 加 np.asarray float32）
9. **【M3 实测 bug·教育版同样存在】流式生成器被当同步字符串用**：
   `new_main.call_dashscope` 是流式生成器（yield），但 `new_rag_system` 的
   HyDE/子查询/回溯三个策略对返回值调 `.strip()` → `'generator' object has no attribute 'strip'`
   → 被 except 吞掉 → 检索返回空 → 表现为"信息不足"。**只在策略选中这三条时触发**，
   课程演示全用直接检索所以从未暴露；最终答案生成那条路靠"生成器透传"侥幸工作。
   已修: 新增 `_llm_text()` 兼容两种注入（生成器则拼接、字符串原样），三个策略调用点改用它。
10. **已知数据缺口（防误判为 bug）**："如何减少门店卤味的报损"这类问题会兜底
    —— 知识库里没有"方法类"内容（ops 整理稿只有报损制度）。HyDE 生成的假设答案其实
    质量很好，是**数据缺口**不是链路故障。补齐靠访谈提纲"当天没卖完怎么处理"问深一层。
11. **【M4 实测 bug·教育版同样存在】同步 RAG 链路阻塞 async 事件循环**：
    提供的 `app.py` 在 async WebSocket 端点里直接 `for` 迭代同步生成器
    `qa_system.query(...)`, 而该生成器内部有大模型调用/模型推理等阻塞操作
    （实测首次策略选择一次调用就要 23.5 秒）→ 阻塞 uvicorn 事件循环 →
    服务端无法响应 WebSocket 心跳 → 客户端 ~40 秒后判定超时断开
    （现象: 收到 start 后一直没有 token, 然后 connection closed）.
    修法: 新增 `_aiter_sync()` 桥接器, 把同步生成器的每一步放进线程池
    (`asyncio.to_thread`) 执行, 事件循环保持空闲. 修复后 WS 全链路验收通过.

12. **【前端实测】CommonMark 中文加粗不生效（static/index.html）**：
    RAG 答案里"应在**高于60℃或低于8℃**的条件下"的 `**` 原样露出——CommonMark 的强调规则
    要求 `**` 左侧为空白/标点才能"开启"，**中文夹 `**` 恰好两端都是汉字，判定失败**；
    而大模型写中文加粗（强调数字）时天生就是这个形状，所以这类问题会反复出现。
    修法：渲染前用正则把成对 `**…**` 先转 `<strong>`，再交给 marked（`renderMarkdown()`）。
13. **【前端实测】监听器挂在网络初始化之后存在操作空窗期**：
    `DOMContentLoaded` 里先 `await createNewSession()` / `await loadSources()` 才
    `addEventListener('click', sendMessage)` —— 初始化期间（含两次网络请求）用户点击发送毫无反应。
    修法：监听器提前到网络初始化之前。

## 前端重设计（2026-09-27）

按用户的 `ai-webpage-design` skill 流程重做 `static/index.html`（课程自带页面视觉较差）：

- 规范落盘：`docs/DESIGN-SPEC.md`（设计令牌/布局/功能清单）+ `docs/REJECT-LIST.md`（否决清单）
- 调性：**中式现代 · 红黑米白**（用户选定）；禁止紫蓝渐变/发光（用户勾选）
- 结构：新增**欢迎态**（品牌字标 + 快捷问题），提问后收敛为对话流；移除三个"开发中"假按钮与 Font Awesome 依赖
- 全部 10 项原有功能保留并逐条验收（会话/分类/非流式/流式/Markdown/历史/清空/搜索/主题/防重复提交）

## M4 验收数据（2026-09-27 实测, WebSocket 全链路）

| 查询 | 首 token | 总耗时 | 说明 |
|---|---|---|---|
| 精品猪头肉多少钱一斤 | **0.03s** | 0.03s | FAQ 通道, 零 LLM 调用 |
| 门店里卤味陈列和存放温度有什么要求 | 38.53s | 42.94s | RAG 首次（含 LLM 端点冷启动 ~23.5s） |
| 卤味冷藏后怎么加热 | 12.24s | 14.15s | RAG 第二次（冷启动已过） |

**优化方向（面试可讲）**：① 策略选择与查询编码可并行；② reranker 首次 predict 有 CUDA 初始化开销, 可服务启动时预热；③ LLM 端点冷启动属外部因素, 可加"启动自检调用"。

## 八、下一步（本周）

1. **你**：按《docs/伯父访谈提纲.md》问伯父（先问产品和门店两组，30 分钟），把现成资料发我
2. **你**：继续手写 `base/config.py` 和 `base/logger.py`（与领域无关，教育版规格直接用）
3. **我**：收到你的访谈记录后，起草 `deli_faq.csv`、product/franchise/SOP 三份文档、
   生成 BERT 训练集并抽查
4. **我**：给出 `app.py`/`index.html` 的逐行文案 diff 和教育版 httpx 修复 diff
