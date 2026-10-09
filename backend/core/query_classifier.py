# backend/core/query_classifier.py
# 熟食语料微调的 BERT 查询二分类器（通用知识 / 专业咨询）。
# 迁移自 rag_qa/core/query_classifier.py 的推理部分（训练移至 scripts/train_classifier.py）：
# tokenizer 与权重同在 models/classifier/bert_query_classifier，直接从该目录加载。
# 架构位置：问答流水线意图判定的模型通道；规则快判逻辑在 agents/qa/service.py。

from typing import Optional

import torch

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 与训练标签严格一致：通用知识->0, 专业咨询->1
LABEL_GENERAL = "general"
LABEL_SPECIALIZED = "specialized"
_ID_TO_LABEL = {0: LABEL_GENERAL, 1: LABEL_SPECIALIZED}


class QueryClassifier:
    """BERT 二分类单例。classify() 返回英文标签与两类概率。"""

    _instance: Optional["QueryClassifier"] = None

    def __init__(self):
        from transformers import BertForSequenceClassification, BertTokenizer

        settings = get_settings()
        model_dir = settings.classifier_abs_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("classifier.loading", model_dir=model_dir, device=str(self.device))

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("classifier.loaded")

    @classmethod
    def get_instance(cls) -> "QueryClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, query: str) -> tuple[str, dict]:
        """
        :return: (label, {"general": p0, "specialized": p1})
        """
        encoding = self.tokenizer(
            query, truncation=True, padding=True,
            max_length=128, return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
        label = _ID_TO_LABEL[int(probs.index(max(probs)))]
        scores = {LABEL_GENERAL: float(probs[0]), LABEL_SPECIALIZED: float(probs[1])}
        logger.info(
            "classifier.predicted",
            query=query[:30], label=label,
            general=round(scores[LABEL_GENERAL], 4),
            specialized=round(scores[LABEL_SPECIALIZED], 4),
        )
        return label, scores


def classify_query(query: str) -> tuple[str, dict]:
    """便捷入口。模型不可用时降级为 specialized（走 RAG 更安全），不阻断主流程。"""
    try:
        return QueryClassifier.get_instance().classify(query)
    except Exception as e:
        logger.error("classifier.unavailable_degrade_specialized", error=str(e))
        return LABEL_SPECIALIZED, {LABEL_GENERAL: 0.0, LABEL_SPECIALIZED: 0.0}
