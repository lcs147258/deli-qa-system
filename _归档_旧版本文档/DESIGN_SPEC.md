# EduRAG 14 个待实现文件 —— 设计规格

> 这份文档给你**接口和设计意图**，不给代码。
> 你的任务：按规格自己实现。卡住了再去对照
> `04-代码/随堂代码/integrated_qa_system/` 里的参考实现。
>
> **为什么要这样**：面试官会问"这个函数为什么这么设计"，只有你自己写过才答得出来。
> 直接复制粘贴的代码，你记不住，也扛不住追问。

---

## ⚠️ 先搞清楚 V1 / V2 的关系（不然你会困惑"我写的代码到底跑没跑"）

项目有两套 RAG 核心，**你要写的是 V1，跑 Web 界面用的是 V2**：

| | 文件 | 谁写的 | 特点 |
|---|---|---|---|
| **V1** | `rag_qa/core/rag_system.py` | ⬜ **你写**（第 13 个文件） | 基础版，无多轮历史、无流式 |
| **V2** | `rag_qa/core/new_rag_system.py` | ✅ 已提供 | 增加多轮对话 + 流式输出 |

对应两个入口：

| 入口 | 用的版本 | 用途 |
|---|---|---|
| `old_main.py` | **V1** ← 你写的 | **你测试自己代码就用这个** |
| `new_main.py` / `app.py` | V2 | Web 界面走这里 |

**关键点：V2 并不是完全独立的。** `new_rag_system.py` 里 import 了
`RAGPrompts`、`QueryClassifier`、`StrategySelector`、`VectorStore` —— 这 4 个都是你要写的文件。
所以**你写完这 4 个之后，启动 Web 服务时跑的就是你的代码**（只有 `rag_system.py` 被 V2 替换了）。

**所以验证顺序是**：

```bash
# 1. 先验证你写的 V1 全流程
python old_main.py                      # ← 用你的 rag_system.py

# 2. 再启动 Web 服务（用 V2，但复用了你写的 4 个核心模块）
PORT=8081 python app.py
```

`rag_qa/__init__.py` 里当前导入的是 V2：
```python
# from core.rag_system import RAGSystem        # V1（你写的）
from core.new_rag_system import RAGSystem      # V2（已提供）
```
等你想让 Web 服务也跑你自己的 V1 时，把这两行的注释对调即可 —— **这本身就是一个很好的实验**（对比 V1/V2 的行为差异）。

---

## ⚠️ 模型路径必须用 ASCII（否则 reranker 一定加载失败）

这是我实测出来的坑，**你不处理的话 `bge-reranker-large` 100% 加载不了**。

**现象**：
```
OSError: Unable to load vocabulary from file.
Please check that the provided vocabulary is accessible and not corrupted.
```
但你去检查文件，`sentencepiece.bpe.model` 明明存在（5MB）。

**原因**：本项目所在路径含中文（`E:\三个项目\05_EduRAG 项目\...`）。Windows 下
`sentencepiece` 的 C++ 层使用**窄字符 API**，无法正确处理含中文的路径 →
打开文件失败 → 报"文件不存在"。

**实测对比**（原始文件完全相同，只是访问路径不同）：

| 访问路径 | 结果 |
|---|---|
| `E:\三个项目\...\my_qa_system\rag_qa\models\bge-reranker-large` | ❌ 加载失败 |
| `E:\edurag_models\bge-reranker-large`（junction 中转） | ✅ 成功，打分 0.997 |

**为什么只有 reranker 中招**：`bge-m3` 目录里有 `tokenizer.json`，走的是快速分词器（纯 Python 实现，能处理 Unicode 路径）；
而 `bge-reranker-large` **只有** `sentencepiece.bpe.model`，必须走 C++ 实现的慢速分词器，就撞上了这个 bug。

**解决方案（已帮你做好）**：

已经建好了 `E:\edurag_models\` 目录，里面是 5 个模型的 junction（目录链接，**不额外占用空间**）：

```
E:\edurag_models\
├── bge-m3                                        -> 实际模型目录
├── bge-reranker-large                            -> 实际模型目录
├── bert-base-chinese                             -> 实际模型目录
├── bert_query_classifier                         -> 实际模型目录
└── nlp_bert_document-segmentation_chinese-base   -> 实际模型目录
```

`config.ini` 里已经加好了配置项：
```ini
[model]
dir = E:\edurag_models
```

**你要做的**：写代码加载模型时，**从 `Config().MODEL_DIR` 拼路径，不要用项目内的相对路径**。

```python
# ✅ 正确
model_path = os.path.join(Config().MODEL_DIR, 'bge-reranker-large')
CrossEncoder(model_path, device='cuda')

# ❌ 错误 —— 会报 "Unable to load vocabulary from file"
model_path = os.path.join(project_root, 'rag_qa', 'models', 'bge-reranker-large')
CrossEncoder(model_path, device='cuda')
```

> **顺带一个知识点**（面试可以讲）：模型、数据这类需要被 C/C++ 库读取的文件，
> 放在纯 ASCII 路径下是生产环境的常见约定。中文路径、空格、特殊字符都可能
> 在某个底层库上踩坑，而且报错信息往往误导（说"文件不存在"，但文件明明在）。

---

## ⚠️ 写代码时必须避开的 7 个坑（来自参考实现的实测）

我把参考实现完整跑通了，**下面这些坑它全踩了**。你在写的时候就避开，
既省时间，又能在面试时讲"我知道这里有个坑，因为…"。
详细记录见 `04-代码\实操手册_参考实现跑通全记录.md`。

| # | 坑 | 你要怎么写 |
|---|---|---|
| ① | BGE-M3 在 GPU+fp16 下输出 **float16**，Milvus 要求 **float32** | 入库和查询**两侧**都写 `np.asarray(vec, dtype=np.float32)` |
| ② | 排序时若两个**分数相同**，Python 会比较 `Document` 对象 → 崩溃 | 排序**必须**写 `key=lambda pair: pair[0]`，不要靠元组隐式比较 |
| ③ | 模型放中文路径下，`sentencepiece` 读不了（报错说"文件不存在"，但文件在） | 用 `Config().MODEL_DIR`（`E:\edurag_models`） |
| ④ | `from base import logger` 拿到的是**模块**不是实例 | 用 `from base.logger import logger`、`from base.config import Config` |
| ⑤ | Redis 写入用 `json.dumps`，读取直接 `get` 没 `json.loads` → **第二次提问格式就变了** | 读写序列化方式必须**严格对称** |
| ⑥ | `pymysql.connect` 没写 `port`/`charset` → 连错库 + 中文乱码 | 显式写 `port=Config().MYSQL_PORT, charset='utf8mb4'` |
| ⑦ | 重排后**没有分数阈值兜底** → 知识库外的问题照样返回内容 | 加阈值：实测相关 0.83~0.995、无关 0.0001~0.003，阈值取 0.1~0.3 即可 |

**其中 ⑤ 和 ② 特别值得你在面试时讲**，因为它们"第一次都正常，第二次/特定情况才崩"，
属于最难排查的一类 bug。具体现象和数据见实操手册的第四章。

---

## 目录与依赖关系

```
第 1 层  base/config.py        配置中心（所有模块都依赖它）
         base/logger.py        日志

第 2 层  mysql_qa/utils/preprocess.py     文本预处理（jieba 分词）
         mysql_qa/db/mysql_client.py      MySQL 客户端
         mysql_qa/cache/redis_client.py   Redis 客户端
         mysql_qa/retrieval/bm25_search.py  BM25 检索 + 阈值判定
         mysql_qa/main.py                  FAQ 问答系统（组装上面四个）

第 3 层  rag_qa/core/prompts.py            提示词模板
         rag_qa/core/document_processor.py 文档加载 + 父子分块
         rag_qa/core/vector_store.py       Milvus 混合检索 + 重排
         rag_qa/core/query_classifier.py   BERT 意图分类器
         rag_qa/core/strategy_selector.py  LLM 选检索策略
         rag_qa/core/rag_system.py          RAG 主流程

第 4 层  rag_qa/main.py                     RAG 系统入口（CLI）
```

**建议实现顺序**：严格从上到下。每一层都依赖上一层，跳着写会一直报错。

---

# 第 1 层：基础设施

## 1. `base/config.py` —— 配置中心（39 字节空壳）

**职责**：读 `config.ini`，暴露成属性，全项目共用一份实例。

**必须实现的接口**：

```python
class Config:
    def __init__(self, config_file=config_file_path): ...

# 模块级单例，其他模块都 `from base.config import config` 然后 config.XXX
config = Config()
```

**必须暴露的属性**（名字不能改，其他模块直接引用）：

| 分组 | 属性名 |
|---|---|
| MySQL | `MYSQL_HOST` `MYSQL_PORT` `MYSQL_USER` `MYSQL_PASSWORD` `MYSQL_DATABASE` |
| Redis | `REDIS_HOST` `REDIS_PORT` `REDIS_PASSWORD` `REDIS_DB` |
| 日志 | `LOG_FILE` |
| Milvus | `MILVUS_HOST` `MILVUS_PORT` `MILVUS_DATABASE` `MILVUS_COLLECTION` |
| LLM | `LLM_MODEL` `DASHSCOPE_API_KEY` `DASHSCOPE_BASE_URL` |
| 检索 | `PARENT_CHUNK_SIZE` `CHILD_CHUNK_SIZE` `CHUNK_OVERLAP` `RETRIEVAL_K` `CANDIDATE_M` |
| 应用 | `VALID_SOURCES` `CUSTOMER_SERVICE_PHONE` |
| **模型路径** | **`MODEL_DIR`** ← ⚠️ 见下方说明，不要漏 |

**设计要点**：

1. 用 `configparser` 读 ini，`config.get('mysql', 'host', fallback='localhost')` 形式给默认值
2. **每个配置项都要支持环境变量覆盖**：`os.getenv('MYSQL_HOST', self.config.get(...))`。这是为了容器化部署时不用改文件
3. `VALID_SOURCES` 存的是 JSON 字符串，要 `json.loads()` 成 list
4. 数字类型（port、chunk_size）要 `int()` 转换
5. `config_file_path` 要用 `os.path.dirname(__file__)` 相对定位，**不要写死绝对路径**（参考实现里 `new_rag_system.py` 就写死了 `D:\workspace\...`，换机器必崩 — 别学这个）

**验收**：`python -c "from base.config import config; print(config.MYSQL_PORT, config.VALID_SOURCES)"` 能打印出 `3307 ['ai','java','test','ops','bigdata']`

---

## 2. `base/logger.py` —— 日志（32 字节空壳）

**职责**：统一 logger，控制台 + 文件双输出。

**必须实现的接口**：

```python
def setup_logger(log_file=log_file): 
    """返回一个配置好的 logging.Logger"""
```

**⚠️ 最容易漏的一点（漏了必报 ImportError）**：

`base/__init__.py` 里写的是 `from logger import logger`，所以你的 `logger.py`
**除了定义函数，还必须在模块末尾创建一个模块级实例**：

```python
logger = setup_logger()      # ← 必须有这一行
```

这样其他模块 `from base.logger import logger` 才能拿到。参考实现的最后一行就是这个。

**设计要点**：

1. 用 `logging.getLogger('EduRAG')` 拿同一个 logger（**关键**：多次调用不能重复加 handler，否则日志会重复打印 N 遍）
2. 两个 handler：`StreamHandler`（控制台）+ `FileHandler`（写文件，注意 `encoding='utf-8'`，否则中文乱码）
3. `FileHandler` 要保证 `logs/` 目录存在：`os.makedirs(os.path.dirname(log_file), exist_ok=True)`
4. 格式建议：`'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
5. **判重用 `if not logger.handlers:`** —— 已有 handler 就跳过，不要重复添加
6. 日志路径从 `Config().LOG_FILE` 取，再拼成绝对路径

**验收**：连续调用两次 `setup_logger()`，日志只打印一次；`from base.logger import logger` 不报错。

---

# 第 2 层：FAQ 问答系统（MySQL + Redis + BM25）

## 3. `mysql_qa/utils/preprocess.py` —— 文本预处理（35 字节空壳）

**职责**：把一句话切成词列表，供 BM25 使用。

**必须实现的接口**：

```python
def preprocess_text(text): 
    """输入 str，返回 list[str]"""
```

**设计要点**：

1. `jieba.lcut(text.lower())` — 中文分词 + 转小写
2. 要过滤掉空白字符（`if token.strip()`）
3. **思考题**（面试会问）：为什么要转小写？为什么要过滤空白？如果不过滤会怎样？
4. **注意一致性**：建库时和查询时必须用**同一个**预处理函数，否则 BM25 的词表对不上

**验收**：`preprocess_text("什么是 机器学习?")` → `['什么', '是', '机器', '学习', '?']`

---

## 4. `mysql_qa/db/mysql_client.py` —— MySQL 客户端（50 字节空壳）

**职责**：建表、灌数据、取答案。

**必须实现的接口**：

```python
class MySQLClient:
    def __init__(self): ...                    # 连库，数据库不存在要能自动创建
    def create_table(self): ...                # 建 jpkb 表
    def insert_data(self, csv_path): ...       # 从 CSV 灌数据
    def fetch_questions(self): ...             # 取出所有问题（给 BM25 建索引）
    def fetch_answer(self, question): ...      # 按问题查答案
    def close(self): ...
```

**表结构**（`jpkb` = 精品课知识库）：
- `id` 主键自增
- `question` 问题文本
- `answer` 答案文本
- 建议给 `question` 加索引

**设计要点**：

1. 用 `pymysql`，连接参数**必须显式指定 port 和 charset**：

```python
pymysql.connect(
    host=Config().MYSQL_HOST,
    port=Config().MYSQL_PORT,          # ⚠️ 必须写！你的 MySQL 在 3307
    user=Config().MYSQL_USER,
    password=Config().MYSQL_PASSWORD,
    database=Config().MYSQL_DATABASE,
    charset='utf8mb4'                  # ⚠️ 必须写！否则中文乱码
)
```

> **⚠️ 这两行是你必须加的，参考实现里没有。**
> 课程作者的 MySQL 在默认的 3306，所以不写 port 也对；**但你的 3306 被本机原生 MySQL80 占了，
> 教程的 Docker MySQL 映射到了 3307**。不写 port 就会连到原生 MySQL，密码对不上，直接连接失败。
> 同理不写 charset，中文会变问号。**这两处别照抄参考实现。**

2. **`__init__` 里要先连到 server（不指定 database）建库，再连到库** — 或者用 `CREATE DATABASE IF NOT EXISTS`
   （本项目的 `subjects_kg` 已经由 Docker 自动创建好了，你可以直接连）
3. `insert_data` 用 `executemany` 批量插入，比逐条快几十倍
4. **防重复插入**：判断表里已有数据就跳过，否则每次跑都插一遍
5. CSV 路径 = `mysql_qa/data/JP学科知识问答.csv`，用 `os.path.dirname(__file__)` 定位
6. `fetch_answer` 要 `fetchone()` 并处理查不到的情况（返回 None）
7. 异常要 `raise` 出去，不要吞掉 —— 让调用方知道连接失败了

**验收**：`SHOW TABLES` 能看到 `jpkb`，`SELECT COUNT(*)` 等于 CSV 行数；中文显示正常（不是问号）。

---

## 5. `mysql_qa/cache/redis_client.py` —— Redis 缓存（37 字节空壳）

**职责**：缓存分词结果、全部问题、以及命中答案。

**必须实现的接口**：

```python
class RedisClient:
    def __init__(self): ...
    def set_data(self, key, value): ...      # 存（JSON 序列化）
    def get_data(self, key): ...             # 取（JSON 反序列化）
    def get_answer(self, query): ...         # 按 query 直接取缓存答案
```

**要缓存的三种 key**（这是性能关键，理解了才算懂这个项目）：

| key | value | 为什么 |
|---|---|---|
| `qa_original_questions` | 原始问题列表 | 避免每次启动都查 MySQL 重建 |
| `qa_tokenized_questions` | **分词后**的问题列表 | 分词很慢，缓存下来省时间 |
| `answer:{query}` | 命中的答案 | 高频问题直接命中，连 BM25 都不用算 |

**设计要点**：

1. `redis.Redis(host, port, password=..., db=..., decode_responses=True)`
2. `decode_responses=True` 让返回的是 str 而不是 bytes，省去手动 decode
3. 用 `json.dumps/loads` 存 list，不能直接存 Python 对象
4. **`get_answer` 返回的是字符串**（因为高频答案直接给用户），和 `get_data` 返回反序列化对象**不一样** — 注意别混
5. Redis 挂了不应该让整个系统崩：建议 try/except 兜底，降级为不用缓存

**验收**：`redis-cli -a 1234 keys *` 能看到上述 key。

---

## 6. `mysql_qa/retrieval/bm25_search.py` —— BM25 检索（67 字节空壳）★核心

**职责**：全项目最巧妙的一层 —— 用 BM25 判断"这个问题是不是高频问题"。

**必须实现的接口**：

```python
class BM25Search:
    def __init__(self, redis_client, mysql_client): ...
    def _load_data(self): ...                      # 加载问题库（优先读 Redis）
    def _softmax(self, scores): ...                # 分数归一化
    def search(self, query, threshold=0.85): ...   # 返回 (answer, need_rag, similarity)
```

**返回值约定**（很关键，上层靠这个分流）：
- `search()` 返回三元组 `(answer, need_rag, similarity)` 或相似结构
- 命中（分数 ≥ 阈值）→ 返回答案，标记不需要走 RAG
- 未命中 → 返回 None，标记需要走 RAG

**核心算法 —— 为什么用 softmax**：

BM25 原始分数是**无界的**（可能 0～20+），没法直接和 0.85 比。所以：
1. `BM25Okapi.get_scores(tokenized_query)` 拿到所有候选问题的原始分数
2. **用 softmax 把分数转成概率分布**（总和为 1）
3. 取 `argmax` 对应的概率，与阈值 0.85 比较

**这意味着 0.85 的含义是"这个问题占全部候选问题的概率质量 85%"**，不是"文本相似度 85%"。

**设计要点**：

1. `_softmax` 要**减去最大值再 exp**（防溢出）：`exp(x - max) / sum(exp(x - max))`。这是数值稳定的标准做法，面试常问
2. `_load_data` 的优先级：先查 Redis 缓存的分词结果 → 没有才查 MySQL 并分词 → 写回 Redis
3. BM25 语料只在启动时构建一次（`BM25Okapi(tokenized_corpus)`），**不要每次查询重建**，否则极慢
4. 要维护 `问题 → 答案` 的映射，命中后能取到答案
5. **冷启动问题**：如果 Redis 和 MySQL 都空，要给出明确报错，不能静默失败

**验收**：问一个 CSV 里有的高频问题 → 秒回；问一个库里没有的问题 → 返回需要走 RAG。

---

## 7. `mysql_qa/main.py` —— FAQ 问答系统（46 字节空壳）

**职责**：把上面四个模块组装成 `MySQLQASystem`。

**必须实现的接口**：

```python
class MySQLQASystem:
    def __init__(self): ...            # 创建 MySQLClient / RedisClient / BM25Search
    def query(self, query): ...        # 主入口：返回 (answer, need_rag)

def main(): ...                        # 命令行测试入口
```

**设计要点**：

1. `query()` 里先查 Redis 缓存答案（`get_answer`）→ 命中直接返回，**这一步能挡掉大量重复提问**
2. 未命中缓存 → 走 `BM25Search.search()`
3. 命中 → 把答案写回 Redis 缓存 → 返回
4. 未命中 → 返回 `(None, True)` 表示"交给 RAG"
5. `__init__` 里要建表 + 灌数据（幂等），否则第一次跑没数据
6. 加耗时日志：`logger.info(f'查询处理耗时 {elapsed:.2f}秒')` — 这是面试要讲的数据

**验收**：`python mysql_qa/main.py` 能进入交互，问高频问题秒回。

---

# 第 3 层：RAG 系统

## 8. `rag_qa/core/prompts.py` —— 提示词模板（38 字节空壳）

**职责**：四个 prompt 模板，都用 `@staticmethod`。

**必须实现的接口**：

```python
class RAGPrompts:
    @staticmethod
    def rag_prompt(): ...          # 主问答模板，含 {context} {question} {history} {phone}
    @staticmethod
    def hyde_prompt(): ...         # HyDE：让 LLM 先"编"一个假答案
    @staticmethod
    def subquery_prompt(): ...     # 子查询：把复杂问题拆成多个
    @staticmethod
    def backtracking_prompt(): ... # 回溯：把问题抽象成更通用的形式
```

**设计要点**：

1. `rag_prompt` 必须包含占位符 `{context}` `{question}`，多轮场景还要 `{history}`，兜底话术要 `{phone}`
2. **要约束输出**：指定语言（中文）、指定风格（专业）、要求"资料不足时明确说明"
   - 参考实现的讲义自己承认了缺陷："没有指定具体的领域""回答的语言风格没有指定，偶发小红书等非专业风格" —— 你在写的时候就应该修掉
3. **HyDE 的原理**（面试重点）：先让 LLM 生成一个假设答案，再用这个答案去检索。因为"答案"和"文档"的语义分布比"问题"和"文档"更接近，所以召回率更高
4. 每个 prompt 都要求 LLM 输出**结构化的东西**（策略选择要输出策略名，子查询要输出 JSON 列表）

**验收**：`RAGPrompts.rag_prompt().format(context='x', question='y', history='', phone='110')` 不报 KeyError。

---

## 9. `rag_qa/core/document_processor.py` —— 文档加载与父子分块（67 字节空壳）★核心

**职责**：扫描目录 → 加载文档 → 切成父子块 → 返回可入库的 Document 列表。

**必须实现的接口**：

```python
def load_documents_from_directory(directory_path): 
    """返回 List[Document]"""

def process_documents(directory_path, parent_chunk_size=..., 
                      child_chunk_size=..., chunk_overlap=...): 
    """返回父子块列表，子块带 parent_content 元数据"""
```

**父子分块（Parent-Child / Small-to-Big）—— 这是本项目的技术亮点**：

```
父块 (1200 字) ── 提供完整上下文，送给 LLM 用
   └── 子块 (300 字) ── 做向量检索用，匹配更精准
```

**为什么这么做**：
- 子块小 → 向量表达更聚焦，检索更准
- 但小块喂给 LLM 上下文不完整 → 所以命中后**回退到父块**送给 LLM
- 解决了 RAG 里"检索精度"和"上下文完整性"的矛盾

**必须写入的元数据**：

| 字段 | 说明 |
|---|---|
| `parent_id` | 父块 ID，格式如 `doc_{i}_parent_{j}` |
| `parent_content` | **父块正文**，检索命中后靠它回升 |
| `source` | 来源标识（学科：ai/java/test/ops/bigdata），用于过滤 |
| `timestamp` | 入库时间 |

**设计要点**：

1. 按**文件类型**分派 loader：PDF 用 `OCRPDFLoader`、DOCX 用 `OCRWordLoader`、PPT 用 `OCRPPTLoader`、图片用 `OCRIMGLoader`、md/txt 直接读
2. 这些 loader 已经写好了，在 `rag_qa/edu_document_loaders/`，直接 import 用
3. 中文切分要用 `ChineseRecursiveTextSplitter`（在 `rag_qa/edu_text_spliter/`），它定制了中文标点分隔符
4. **先切父块，再在父块内部切子块**，保证子块不会跨父块
5. ID 命名规则要能让 `vector_store` 反推出父块关系
6. 遍历目录要跳过临时文件（`.DS_Store`、`~$` 开头的 Office 临时文件）
7. **思考题**：为什么用 md5 做文档主键？（提示：幂等 upsert，重复入库不产生重复数据）

**验收**：打印返回列表，能看到子块的 `metadata` 里有 `parent_content`，且 `parent_content` 明显长于子块自身。

---

## 10. `rag_qa/core/vector_store.py` —— Milvus 混合检索 + 重排（41 字节空壳，参考实现 373 行）★最核心

**职责**：本项目最长、最难的文件。向量入库 + 稠密稀疏混合检索 + 重排 + 父块回升。

**必须实现的接口**：

```python
class VectorStore:
    def __init__(self, host=..., port=..., database=..., collection=..., ...): ...
    def _create_or_load_collection(self): ...
    def add_documents(self, documents): ...
    def hybrid_search_with_rerank(self, query, k=..., source_filter=None): ...
    def _get_unique_parent_docs(self, sub_chunks): ...
    def _doc_from_hit(self, hit): ...
```

**Milvus Collection Schema**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR(100) | 主键，`hashlib.md5(内容)` |
| `text` | VARCHAR(65535) | 子块正文 |
| `dense_vector` | FLOAT_VECTOR(dim=1024) | BGE-M3 稠密向量 |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BGE-M3 稀疏向量 |
| `parent_id` | VARCHAR | 父块 ID |
| `parent_content` | VARCHAR | 父块正文 |
| `source` | VARCHAR | 学科，用于过滤 |
| `timestamp` | VARCHAR/INT | 入库时间 |

**索引配置**：
- dense: `IVF_FLAT`, `metric_type="IP"`, `nlist=128`；检索参数 `nprobe=10`
- sparse: `SPARSE_INVERTED_INDEX`, `IP`, `drop_ratio_build=0.2`

**混合检索流程**（这是核心，务必理解每一层为什么）：

```
1. 用 BGE-M3 把 query 编码成 稠密向量 + 稀疏向量
2. 构造两个 AnnSearchRequest（dense 一个、sparse 一个）
   - 如果传了 source_filter，两个请求都要加过滤表达式
3. WeightedRanker 融合两路结果 → 取 TopK 子块
4. 通过 parent_id / parent_content 回升到父块，去重
5. CrossEncoder (bge-reranker-large) 对 (query, 父块) 逐对打分重排
6. 截取前 CANDIDATE_M 个父块返回
```

**设计要点与坑**：

1. **BGE-M3 的用法**：用 `BGEM3EmbeddingFunction`，返回 `{'dense': ..., 'sparse': ...}`。注意稠密 1024 维、稀疏是 scipy 的 `coo_array`，需要转成 Milvus 能接受的 `{index: value}` 字典
   - **路径要用 `os.path.join(Config().MODEL_DIR, 'bge-m3')`**（实测：BGE-M3 有 `tokenizer.json`，走快速分词器，中文路径也能加载；但为统一，建议全部走 ASCII 路径）
2. **`use_fp16`**：你有 RTX 3060 Ti（8G），建议开 fp16 省显存。但 CPU 上跑要关掉
3. **`WeightedRanker` 的权重**：参考实现代码里是 `(1.0, 0.7)`（稠密权重高），但讲义写的是 `(0.7, 1.0)` — **这两者矛盾**。你自己实测决定哪个好，并记录下来。这正好是你的参数实验素材
4. **父块去重要按内容或 ID 去重**，不要按对象内存地址（参考实现有这个缺陷，讲义自己也指出了）
5. **重排后建议加分数阈值过滤**（参考实现没有，导致可能返回完全不相关的文档）— 这是你的改进点
6. `metric_type` 用 `IP`（内积）时，BGE-M3 的向量**必须归一化**过，否则内积不等于余弦相似度
7. 入库前先判断集合是否存在：`utility.has_collection()`
8. 集合里要有数据才能检索，空集合检索会报错

**验收**：
- `collection.num_entities` 等于入库的子块数
- 检索一个明确的问题，返回的父块里确实包含答案相关内容
- 开日志看每步耗时，知道瓶颈在哪

---

## 11. `rag_qa/core/query_classifier.py` —— BERT 意图分类器（35 字节空壳，参考 330 行）★你的阶段四知识在这里兑现

**职责**：把 query 分成"通用知识"和"专业咨询"两类。通用知识直接问 LLM，省掉检索。

**必须实现的接口**：

```python
class QueryClassifier:
    def __init__(self, model_path=None): ...   # ⚠️ 默认用 Config().MODEL_DIR 拼 bert_query_classifier
    def load_model(self): ...
    def save_model(self): ...
    def preprocess_data(self, texts, labels): ...
    def create_dataset(self, encodings, labels): ...
    def train_model(self, data_file="..."): ...
    def compute_metrics(self, eval_pred): ...
    def evaluate_model(self, texts, labels): ...
    def predict_category(self, query): ...
```

> ⚠️ **模型路径要用 `Config().MODEL_DIR` 拼**，不要用项目内相对路径（中文路径会导致加载失败，详见开头那节）。
> 训练数据文件 `model_generic_5000.json` 在 `rag_qa/classify_data/`，这个用相对路径没问题（它是普通 JSON，不经过 C++ 库）。

**为什么用 BERT 小模型而不是直接问大模型**（面试必问）：
1. 延迟：BERT 推理 ~10ms，LLM 调用 ~1s
2. 成本：BERT 本地免费，LLM 按 token 收费
3. 这个分类任务很简单（二分类），小模型足够

**训练要点（你的机器学习基础用在这里）**：

| 项 | 建议值 | 说明 |
|---|---|---|
| 基座 | `bert-base-chinese` | 已在 `rag_qa/models/` |
| `num_labels` | 2 | 通用知识=0，专业咨询=1 |
| `learning_rate` | 2e-5 | 太大不收敛，太小太慢 |
| `epochs` | 3 | BERT 微调 3-5 轮足够，多了过拟合 |
| `batch_size` | 8~16 | 8G 显存用 8 |
| `weight_decay` | 0.01 | 正则化，防过拟合 |
| `warmup_steps` | ~20 | 学习率预热 |
| `metric_for_best_model` | `eval_loss` 或 `accuracy` | 决定选哪个 checkpoint |
| `load_best_model_at_end` | True | 训练完自动加载最优 |

**必须输出的评估结果**：`classification_report`（每类 P/R/F1）+ `confusion_matrix`（混淆矩阵）。**这两个是你面试要讲的数据。**

**设计要点**：

1. 训练数据：`rag_qa/classify_data/model_generic_5000.json`，5000 条，划分 8:2
2. 用 `Trainer` + `TrainingArguments`（不要手写训练循环）
3. 用 `BertForSequenceClassification` + `BertTokenizer`
4. **`predict_category` 要做长度截断**（`max_length=128` 之类），否则长 query 会报错
5. `predict_category` 用 `torch.no_grad()` + `model.eval()` 推理
6. 训练完记得 `save_model()`，否则每次启动都要重训
7. **思考题**：为什么用 `eval_loss` 选最优模型而不是 `accuracy`？（提示：小数据集上 accuracy 波动大）

**验收**：训练完打印出 `classification_report`，准确率应该在 0.85 以上（参考实现做到 90%+）。

---

## 12. `rag_qa/core/strategy_selector.py` —— 检索策略选择器（42 字节空壳）

**职责**：让 LLM 判断该用哪种检索策略。

**必须实现的接口**：

```python
class StrategySelector:
    def __init__(self): ...
    def call_dashscope(self, prompt): ...       # 调 LLM
    def _get_strategy_prompt(self): ...         # 策略选择的 prompt
    def select_strategy(self, query): ...       # 返回策略名
```

**四种策略**：

| 策略 | 适用场景 | 原理 |
|---|---|---|
| `直接检索` | 明确的问题 | 直接用原 query 检索 |
| `假设问题检索` (HyDE) | 短问题、抽象问题 | 先生成假答案再检索，拉近 query 和文档的语义距离 |
| `子查询检索` | 复合问题（含"和""以及"） | 拆成多个子问题分别检索再合并 |
| `回溯问题检索` | 过于具体的问题 | 把问题抽象化（"这个API怎么调" → "API 调用方法"） |

**设计要点**：

1. 用 OpenAI SDK 调 DashScope 兼容端点：`OpenAI(api_key=..., base_url=...)`
2. `temperature=0.1` — 分类任务要低温度，保证稳定
3. prompt 里要**列出四个策略名并说明适用场景**，要求 LLM 只输出策略名
4. **必须有异常兜底**：LLM 返回了不认识的名字、或调用失败 → 默认返回"直接检索"。**不能让策略选择失败导致整个系统崩**
5. 解析返回时要 strip + 去掉可能的引号和标点

**验收**：`select_strategy("AI课程有哪些")` 返回四个策略之一。

---

## 13. `rag_qa/core/rag_system.py` —— RAG 主流程（44 字节空壳，参考 289 行）★组装层

**职责**：把检索、策略、prompt、LLM 串起来。

**必须实现的接口**：

```python
class RAGSystem:
    def __init__(self, vector_store, llm): ...
    def _retrieve_with_hyde(self, query, source_filter=None): ...
    def _retrieve_with_subqueries(self, query, source_filter=None): ...
    def _retrieve_with_backtracking(self, query, source_filter=None): ...
    def retrieve_and_merge(self, query, source_filter=None, strategy=None): ...
    def generate_answer(self, query, source_filter=None): ...
```

**主流程**：

```
generate_answer(query, source_filter)
  ├─ QueryClassifier.predict_category(query)
  │    ├─ "通用知识" → 直接调 LLM，不检索
  │    └─ "专业咨询" ↓
  ├─ StrategySelector.select_strategy(query) → 策略名
  ├─ retrieve_and_merge(query, source_filter, strategy)
  │    └─ 按策略分派到对应 _retrieve_with_xxx
  │         └─ 每个都调 vector_store.hybrid_search_with_rerank()
  ├─ 把检索到的父块拼成 context
  ├─ RAGPrompts.rag_prompt().format(context, question, history, phone)
  └─ 调 LLM 生成答案（流式）
```

**设计要点与坑**：

1. **`source_filter` 必须传给所有分支** — 参考实现只在"直接检索"分支传了，导致 HyDE/子查询/回溯三条路径下**学科过滤失效**。这是个真 bug，你要修掉。（发现并修复它 = 面试加分项）
2. `_retrieve_with_subqueries` 要对每个子查询都检索，然后合并去重
3. **去重要按 `doc.page_content`**，不要按对象内存地址（参考实现的缺陷）
4. 子查询分支开销大（每个子查询都跑一遍 hybrid search + rerank），要在日志里记耗时
5. `generate_answer` 返回**生成器**（流式），上层 `app.py` 用 `for chunk in ...` 迭代
   - 如果返回字符串，前端流式就没法工作
   - **注意分支返回类型要一致**：参考实现"通用知识"分支返回字符串、其他分支返回生成器，这是不一致的坑，你要统一
6. 上下文拼装要**加分隔符标明边界**（讲义建议："上下文和问题的内容，增加符号表示边界"）
7. 上下文长度要控制（讲义经验值：父块 + 多轮对话 + prompt 模板 < 3 万字）

**验收**：调 `generate_answer` 能拿到逐块返回的生成器；传 `source_filter` 时四条策略都真正生效（自己验证！）。

---

# 第 4 层：入口

## 14. `rag_qa/main.py` —— RAG 系统 CLI 入口（45 字节空壳，参考 167 行）

**职责**：命令行入口，两种模式。

**必须实现的接口**：

```python
def main(query_mode=True, directory_path="data"): ...
```

**两种模式**：
1. `--data-processing` → 数据处理模式：扫目录、加载文档、切父子块、入库 Milvus
2. 默认 → 交互查询模式：读用户输入，调 RAGSystem 回答

**设计要点**：

1. 用 `argparse` 解析参数
2. 数据处理模式要做成**幂等**的（重复跑不产生重复数据）
3. 交互模式要能优雅退出（Ctrl+C / 输入 quit）
4. 要打印耗时（入库耗时、查询耗时）
5. **注意**：`main()` 里的 CLI 模式是开发调试用的；生产入口是 `app.py` 的 FastAPI 服务

**验收**：`python rag_qa/main.py --data-processing` 能把 `rag_qa/data/ai_data/` 里的文档灌进 Milvus；`python rag_qa/main.py` 能进交互问答。

---

# 实现顺序与验收清单

按这个顺序做，每做完一个先验收再往下：

- [ ] 1. `base/config.py` — 能读出配置
- [ ] 2. `base/logger.py` — 不重复打印
- [ ] 3. `mysql_qa/utils/preprocess.py` — 分词正确
- [ ] 4. `mysql_qa/db/mysql_client.py` — 表建好、数据灌进去
- [ ] 5. `mysql_qa/cache/redis_client.py` — key 能读写
- [ ] 6. `mysql_qa/retrieval/bm25_search.py` — 高频命中、长尾不命中
- [ ] 7. `mysql_qa/main.py` — **FAQ 通道端到端通了**（第一个里程碑）
- [ ] 8. `rag_qa/core/prompts.py` — 模板能 format
- [ ] 9. `rag_qa/core/document_processor.py` — 父子块元数据正确
- [ ] 10. `rag_qa/core/vector_store.py` — **能入库、能检索出相关父块**（最难的一关）
- [ ] 11. `rag_qa/core/query_classifier.py` — 训练出准确率
- [ ] 12. `rag_qa/core/strategy_selector.py` — 四种策略能选
- [ ] 13. `rag_qa/core/rag_system.py` — **RAG 通道端到端通了**（第二个里程碑）
- [ ] 14. `rag_qa/main.py` — 两个模式都能跑

---

# 做完之后一定要做的实验（面试素材来源）

跑通只是及格，**这些实验才是你面试能讲的东西**：

| 实验 | 改什么 | 记录什么 |
|---|---|---|
| 1 | BM25 阈值 0.85 → 0.6 / 0.7 | 命中率、误命中（答非所问）怎么变 |
| 2 | `child_chunk_size` 300 → 600 | 检索质量、上下文长度、耗时 |
| 3 | `retrieval_k` 3 → 5 / 10 | 召回率、重排耗时、最终效果 |
| 4 | `WeightedRanker` 权重反转 | 哪类 query 变好、哪类变差 |
| 5 | 关掉 rerank | 重排到底贡献了多少 |
| 6 | 四种策略两两对比 | 哪类问题适合哪种策略 |
| 7 | `nlist` / `nprobe` 调大调小 | 召回率 vs 延迟的取舍 |

**每组实验记录成表格**：改了什么 / 指标变化 / 我的结论。做完这份表，你的简历和面试就有了不可编造的内容。

---

# 最后

**卡住时的顺序**：
1. 先看这份规格的"设计要点"
2. 再看参考实现对应文件（只看到能理解为止，不要整段复制）
3. 还不行就带着**具体报错**问我

**不要跳过验收步骤**。每个文件验收通过再往下，比写完 14 个再一起调错要快得多 —— 一次调 14 个文件的 bug 是灾难。
