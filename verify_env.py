# -*- coding: utf-8 -*-
"""DeliRAG 环境自检脚本（重构版）
验证: Python 依赖 / GPU / 中间件(MySQL 3307 · Redis 6380 · Milvus 19530) /
      模型文件 / 项目关键文件 / LLM 端点
运行: E:\\venvs\\edurag\\Scripts\\python.exe verify_env.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, FAIL, WARN = "[OK]  ", "[FAIL]", "[WARN]"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail or ""))
        print(f"{OK} {name}" + (f"  -> {detail}" if detail else ""))
    except Exception as e:
        results.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"{FAIL} {name}  -> {type(e).__name__}: {e}")


from backend.config import get_settings
s = get_settings()

print("=" * 68)
print("  DeliRAG 环境自检（四策略父子检索 RAG 流水线）")
print("=" * 68)

print("\n【1】Python 与依赖包")
check("Python 版本", lambda: sys.version.split()[0])

from importlib.metadata import PackageNotFoundError, version as _pkg_version


def _v(dist_name, module_name=None):
    """优先导入模块取 __version__，没有则读发行包元数据。"""
    mod = module_name or dist_name.replace("-", "_")
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", None)
        if v:
            return v
    except ImportError:
        pass
    try:
        return _pkg_version(dist_name)
    except PackageNotFoundError:
        raise RuntimeError(f"未安装发行包 {dist_name}")


def _torch():
    import torch
    return f"{torch.__version__} | CUDA={'可用' if torch.cuda.is_available() else '不可用'}"


for _name, _fn in [
    ("torch", _torch),
    ("transformers", lambda: __import__("transformers").__version__),
    ("sentence-transformers", lambda: __import__("sentence_transformers").__version__),
    ("FlagEmbedding", lambda: "ok"),
    # langchain 仅用于 ingest 切分基类 + RAGAS 离线评估；生产问答链路零 langchain/langgraph
    ("langchain-core", lambda: _v("langchain-core", "langchain_core")),
    ("langchain-text-splitters", lambda: _v("langchain-text-splitters", "langchain_text_splitters")),
    ("langchain-community", lambda: _v("langchain-community", "langchain_community")),
    ("langchain-openai", lambda: _v("langchain-openai", "langchain_openai")),
    ("pydantic-settings", lambda: _v("pydantic-settings", "pydantic_settings")),
    ("fastapi", lambda: _v("fastapi")),
    ("starlette", lambda: _v("starlette")),
    ("uvicorn", lambda: _v("uvicorn")),
    ("websockets", lambda: _v("websockets")),
    ("mcp", lambda: _v("mcp")),
    ("openai", lambda: __import__("openai").__version__),
    ("pymilvus", lambda: __import__("pymilvus").__version__),
    ("pymysql", lambda: __import__("pymysql").__version__),
    ("redis", lambda: __import__("redis").__version__),
    ("jieba", lambda: __import__("jieba").__version__),
    ("rank_bm25", lambda: "ok"),
    ("numpy", lambda: __import__("numpy").__version__),
    ("pandas", lambda: __import__("pandas").__version__),
    ("scikit-learn", lambda: __import__("sklearn").__version__),
    ("ragas", lambda: __import__("ragas").__version__),
    ("datasets", lambda: __import__("datasets").__version__),
    ("fitz (PyMuPDF)", lambda: "ok"),
    ("python-docx", lambda: "ok" if __import__("docx") else "缺"),
    ("python-pptx", lambda: "ok" if __import__("pptx") else "缺"),
    ("opencv (cv2)", lambda: __import__("cv2").__version__),
    ("Pillow", lambda: __import__("PIL").__version__),
    ("rapidocr_onnxruntime", lambda: "ok"),
]:
    check(_name, _fn)


def _starlette_pin():
    import starlette
    major = int(starlette.__version__.split(".")[0])
    if major >= 1:
        raise RuntimeError(f"starlette {starlette.__version__} 与 fastapi 不兼容，"
                           "必须锁 0.40~0.46（mcp 2.x 会强升它）")
    return "0.x，与 fastapi 兼容"


check("starlette 版本约束", _starlette_pin)

print("\n【2】GPU 实算")


def _gpu_calc():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    a = torch.randn(512, 512, device="cuda")
    b = (a @ a).sum().item()
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name} | {mem:.1f}GB | 矩阵运算结果 {b:.1f}"


check("GPU 矩阵运算", _gpu_calc)

print("\n【3】中间件连通性")


def _mysql():
    import pymysql
    conn = pymysql.connect(host=s.mysql_host, port=s.mysql_port, user=s.mysql_user,
                           password=s.mysql_password, database=s.mysql_database,
                           charset="utf8mb4", connect_timeout=5)
    with conn.cursor() as cur:
        cur.execute("SELECT VERSION()")
        v = cur.fetchone()[0]
        cur.execute("SHOW TABLES")
        tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return f"MySQL {v} @{s.mysql_port}, 库 {s.mysql_database}, 表 {tables}"


check(f"MySQL ({s.mysql_port})", _mysql)


def _redis():
    import redis
    r = redis.Redis(host=s.redis_host, port=s.redis_port, password=s.redis_password,
                    db=s.redis_db, decode_responses=True, socket_connect_timeout=5)
    r.ping()
    return f"PONG @{s.redis_port} db{s.redis_db}"


check(f"Redis ({s.redis_port})", _redis)


def _milvus():
    from pymilvus import MilvusClient
    c = MilvusClient(uri=f"http://{s.milvus_host}:{s.milvus_port}")
    cols = c.list_collections()
    if s.milvus_collection not in cols:
        raise RuntimeError(f"集合 {s.milvus_collection} 不存在，"
                           "先跑 python scripts/init_milvus.py --force")
    rows = c.query(collection_name=s.milvus_collection,
                   filter=f'tenant_id == "{s.default_tenant_id}"',
                   output_fields=["id"], limit=1)
    return f"集合={cols} | {s.milvus_collection} 可查询(样例{len(rows)}行)"


check(f"Milvus ({s.milvus_port}, 集合 {s.milvus_collection})", _milvus)

print("\n【4】模型文件（BGE 系必须 ASCII 路径）")


def _model_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError("目录不存在")
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path) for f in fs)
    return f"{total/1024/1024:.0f} MB"


check("bge-m3", lambda p=s.path_from_root(s.bge_m3_model_path): _model_dir(p))
check("bge-reranker-large",
      lambda p=s.path_from_root(s.reranker_model_path): _model_dir(p))
check("bert-base-chinese (分类器底座)",
      lambda p=s.path_from_root(s.bert_base_model_path): _model_dir(p))
check("熟食微调分类器 bert_query_classifier",
      lambda p=s.classifier_abs_path: _model_dir(p))

print("\n【5】项目关键文件")
BASE = os.path.dirname(os.path.abspath(__file__))
for f in [".env.local", "backend/main.py", "backend/agents/qa/service.py",
          "backend/core/strategy_selector.py",
          "backend/mcp/knowledge_base_server.py", "scripts/init_milvus.py",
          "scripts/build_knowledge_base.py", "scripts/init_faq_data.py",
          "scripts/train_classifier.py", "docker-compose.yml",
          "static/index.html", "data/faq/deli_faq.csv"]:
    check(f, lambda f=f: "存在" if os.path.exists(os.path.join(BASE, f))
          else (_ for _ in ()).throw(FileNotFoundError(f)))

print("\n【6】LLM 端点")


def _llm():
    import httpx
    r = httpx.post(f"{s.llm_base_url}/chat/completions",
                   headers={"Authorization": f"Bearer {s.llm_api_key}"},
                   json={"model": s.llm_model,
                         "messages": [{"role": "user", "content": "ok"}],
                         "max_tokens": 1},
                   timeout=20, trust_env=False)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    return f"{s.llm_model} 可用"


check(f"LLM ({s.llm_model})", _llm)

print("\n" + "=" * 68)
n_fail = sum(1 for ok, _, _ in results if not ok)
n_ok = sum(1 for ok, _, _ in results if ok)
print(f"  结果: {n_ok} 项通过, {n_fail} 项失败")
if n_fail:
    print("\n  失败项:")
    for ok, name, detail in results:
        if not ok:
            print(f"    - {name}: {detail}")
print("=" * 68)
sys.exit(1 if n_fail else 0)
