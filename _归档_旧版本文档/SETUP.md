# EduRAG 项目 —— 环境操作手册

> 这份文档记录**已经帮你搭好的东西**，以及你每天怎么用。
> 生成时间：2026-09-16

---

## 一、当前状态总览

| 组件 | 状态 | 地址 / 路径 |
|---|---|---|
| Milvus 向量库 | ✅ 运行中 | `localhost:19530`，库 `itcast`（已创建） |
| etcd（Milvus 元数据） | ✅ 运行中 | 容器内 `etcd:2379` |
| MinIO（Milvus 对象存储） | ✅ 运行中 | `localhost:9000`，控制台 `localhost:9001` |
| Redis | ✅ 运行中 | `localhost:6379`，密码 `1234` |
| MySQL | ✅ 运行中 | `localhost:3307`，`root/123456`，库 `subjects_kg` |
| Python 环境 | ✅ 就绪 | `E:\venvs\edurag`（Python 3.11.15） |
| PyTorch | ✅ 就绪 | `2.10.0+cu126`，GPU 实算已验证 |
| GPU | ✅ 可用 | RTX 3060 Ti，8GB，sm_86 |
| 模型文件 | ✅ 就绪 | BGE-M3 / bge-reranker-large / bert-base-chinese / **已训练好的分类器** |
| 14 个待实现文件 | ⬜ **待你写** | 见 `DESIGN_SPEC.md` |

**两个需要注意的地方**：

1. **MySQL 端口是 3307，不是课程默认的 3306。** 因为你本机 3306 已被原生 MySQL80 服务占用。`config.ini` 里已经改成 3307 了。
2. **本机 8080 端口被别的项目占了**（`food-kafka-ui` 容器）。EduRAG 的 Web 服务默认用 8080，启动时要改端口：
   ```bash
   PORT=8081 python app.py
   ```

---

## 二、目录结构

```
E:\三个项目\05_EduRAG 项目\04-代码\
├── integrated_qa_system_模板.zip          ← 原始模板（14 个空壳）
├── 随堂代码\integrated_qa_system\          ← 【参考实现】完整版，卡住了对照这里
├── _bootstrap_my_qa_system.py             ← 初始化脚本（已跑过）
├── _link_models.py                        ← 模型链接脚本（已跑过）
├── _make_requirements_win.py              ← 依赖清单生成脚本（已跑过）
└── my_qa_system\                          ← 【你的工作目录】在这里写代码
    ├── DESIGN_SPEC.md                     ← ★ 14 个文件的设计规格，动手前先读
    ├── SETUP.md                           ← 本文件
    ├── docker-compose.yml                 ← 中间件编排
    ├── requirements-win.txt               ← Windows 可用依赖清单
    ├── config.ini                         ← 已配好（MySQL 3307）
    ├── app.py                             ← Web 服务（已提供）
    ├── new_main.py / old_main.py          ← 集成系统 V2 / V1（已提供）
    ├── base\config.py                     ← ⬜ 你要写
    ├── base\logger.py                     ← ⬜ 你要写
    ├── mysql_qa\...                       ← ⬜ 你要写（5 个文件）
    ├── rag_qa\core\...                    ← ⬜ 你要写（6 个文件）
    ├── rag_qa\main.py                     ← ⬜ 你要写
    ├── rag_qa\edu_document_loaders\       ← 已提供（OCR 加载器）
    ├── rag_qa\edu_text_spliter\           ← 已提供（中文切分器）
    ├── rag_qa\rag_assessment\             ← 已提供（RAGAS 评估脚本）
    ├── rag_qa\models\                     ← 模型（junction 链接到参考实现，不占额外空间）
    ├── static\                            ← 前端页面（已提供）
    └── volumes\                           ← Docker 数据卷（别删）
```

---

## 三、日常操作

### 3.1 启动 / 停止中间件

```bash
cd "E:/三个项目/05_EduRAG 项目/04-代码/my_qa_system"

docker compose up -d        # 启动（开机后跑一次）
docker compose ps           # 看状态，等 health 变 healthy
docker compose logs -f milvus   # 看 Milvus 日志（Ctrl+C 退出）
docker compose stop         # 停止（保留数据）
docker compose down         # 停止并删除容器（数据卷 volumes/ 还在）
docker compose up -d        # 再次启动
```

**验证中间件是否就绪**：

```bash
curl http://localhost:9091/healthz                                  # Milvus → OK
docker exec edurag-redis redis-cli -a 1234 --no-auth-warning ping    # Redis → PONG
docker exec edurag-mysql mysql -uroot -p123456 -e "SHOW DATABASES;"  # MySQL → 有 subjects_kg
```

### 3.2 激活 Python 环境

```bash
# Git Bash
source /e/venvs/edurag/Scripts/activate

# 或者直接用绝对路径（不用激活）
/e/venvs/edurag/Scripts/python.exe xxx.py
```

### 3.3 运行项目

```bash
cd "E:/三个项目/05_EduRAG 项目/04-代码/my_qa_system"

# ① 数据入库（把文档灌进 Milvus）—— 要先实现完代码
python rag_qa/main.py --data-processing

# ② 命令行问答测试
python rag_qa/main.py

# ③ 启动 Web 服务（注意端口，8080 被占）
PORT=8081 python app.py
#   然后浏览器打开 http://localhost:8081
```

---

## 四、你还需要做的两件事

### ① 填 DashScope API Key ⚠️ 必须

项目用阿里云百炼（DashScope）的 `qwen-plus` 做生成。你需要：

1. 去 https://bailian.console.aliyun.com/ 注册，开通服务
2. 创建 API Key
3. 填进 `my_qa_system/config.ini`：

```ini
[llm]
model = qwen-plus
dashscope_api_key = sk-你的真实key      ← 改这里
dashscope_base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
```

**注意**：参考实现的 `config.ini` 里那个 key 是别人的（而且已失效），别直接用。

### ② 确认 GPU 能用（已帮你验证过，但你自己会一次更放心）

```bash
/e/venvs/edurag/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
应该输出 `True NVIDIA GeForce RTX 3060 Ti`。

---

## 五、你的下一步

按 `DESIGN_SPEC.md` 的顺序动手。第一个里程碑是**第 7 个文件**（`mysql_qa/main.py`）——做完它，FAQ 通道就能端到端跑通了。

**建议的节奏**：

| 步骤 | 做什么 | 预计 |
|---|---|---|
| 1 | 读 `DESIGN_SPEC.md` 第 1-2 层，理解 FAQ 通道的完整数据流 | 0.5 天 |
| 2 | 写 `base/config.py` + `base/logger.py`，各自验收 | 0.5 天 |
| 3 | 写 `mysql_qa/` 的 5 个文件，逐层验收 | 1-2 天 |
| 4 | 做完 `mysql_qa/main.py`，**FAQ 通道端到端跑通**（第一个里程碑） | 0.5 天 |
| 5 | 写 `rag_qa/core/` 的 6 个文件（`vector_store.py` 最难，留足时间） | 3-5 天 |
| 6 | 写 `rag_qa/main.py`，**RAG 通道端到端跑通** | 0.5 天 |
| 7 | 启动 Web 服务，看到界面 | 0.5 天 |
| 8 | 做参数实验，写实验记录表（面试素材） | 1-2 天 |

---

## 六、常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `Can't connect to MySQL server` | 中间件没启动 | `docker compose up -d` 然后等 health |
| MySQL 连接被拒 | 用了 3306 | 改用 **3307**（本机 3306 是原生 MySQL） |
| **`OSError: Unable to load vocabulary from file`** | **模型用了中文路径** | **改用 `E:\edurag_models\...`（见下方专门说明）** |
| `Milvus 连接失败` / `db not found` | Milvus 没起好 / itcast 库丢了 | 等 healthy；库已建好，若重建容器需再建一次 |
| 端口 8080 被占用 | 别的项目占了 | `PORT=8081 python app.py` |
| `CUDA out of memory` | 8G 显存不够 | 把 reranker 或 embedding 放 CPU：`device='cpu'` |
| 中文乱码 | 编码问题 | 确认 MySQL 用 `utf8mb4`；文件用 UTF-8 读写 |
| `ModuleNotFoundError: No module named 'base'` | 没在项目根目录跑 | `cd` 到 `my_qa_system` 再执行 |
| 模型下载卡住 | 已全部就绪 | 不需要下载，模型已链接好 |

### ⚠️ 模型路径必须用 ASCII —— 这是本机特有的坑

**现象**：加载 `bge-reranker-large` 时报
`OSError: Unable to load vocabulary from file`，但你去查文件，`sentencepiece.bpe.model` 明明在（5MB）。

**原因**：本项目路径含中文（`E:\三个项目\...`）。Windows 下 `sentencepiece` 的 C++ 层
用窄字符 API，处理不了中文路径，于是"找不到文件"。

**实测证据**（同一份文件，只是访问路径不同）：

| 访问路径 | 结果 |
|---|---|
| `E:\三个项目\...\my_qa_system\rag_qa\models\bge-reranker-large` | ❌ 失败 |
| `E:\edurag_models\bge-reranker-large` | ✅ 成功，打分 0.997 |

**为什么只有 reranker 中招**：`bge-m3` 有 `tokenizer.json` 走快速分词器（Python 实现，支持 Unicode）；
`bge-reranker-large` 只有 `sentencepiece.bpe.model`，必须走 C++ 慢速分词器。

**已经帮你解决**：`E:\edurag_models\` 下建好了 5 个模型的 junction（零额外空间），
`config.ini` 里也加好了配置项：

```ini
[model]
dir = E:\edurag_models
```

**你的代码里要用**：
```python
model_path = os.path.join(Config().MODEL_DIR, 'bge-reranker-large')   # ✅
# 不要用 os.path.join(project_root, 'rag_qa', 'models', ...)          # ❌ 会失败
```

**重启电脑后**：中间件不会自动启动，需要跑一次 `docker compose up -d`。

---

## 七、重要提醒

1. **`volumes/` 目录别删** —— Milvus、MySQL、Redis 的数据都在里面。删了数据就没了（要重建也行，重新入库即可）。
2. **不要动别的项目的容器** —— 你机器上还有 `food-spark-*`、`frn-baseline` 等别的项目在跑，那些不是本项目的，别误停。
3. **模型是链接过去的** —— `my_qa_system/rag_qa/models` 是指向参考实现的 junction。**不要删除参考实现目录**，否则模型就断了。
4. **改动 `config.ini` 要重启服务** —— 配置是启动时读的。
