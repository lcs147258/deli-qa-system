# 知识库文档目录（RAG 通道语料）

5 个子目录对应 `config.ini` 的 `valid_sources`，**目录名前缀必须与配置完全一致**
（source 元数据 = 目录名去掉 `_data` 后缀，不一致会导致 Milvus 过滤静默失效）。

| 目录 | 中文 | 状态 |
|---|---|---|
| `product_data/` | 产品知识 | ⬜ 待业务方访谈 |
| `safety_data/` | 食品安全 | ✅ 已有 GB 31654-2021 整理稿 |
| `ops_data/` | 门店运营 | ✅ 已有《餐饮服务食品安全操作规范》整理稿；公司 SOP 待访谈 |
| `franchise_data/` | 加盟与批发 | ⬜ 待业务方访谈 |
| `supply_data/` | 供应链与仓储 | ✅ 已有 GB 31605-2020 整理稿；公司做法待访谈 |

文档格式：Markdown / Word / PDF 均可（加载器已支持）；Markdown 优先（切分效果最好）。
