# scripts/train_classifier.py
# 熟食查询分类器重训：training/deli_classifier.jsonl（{"query","label"}）
#   底座 bert-base-chinese（ASCII 路径）→ 输出 models/classifier/bert_query_classifier
# 用法：python scripts/train_classifier.py
#       python scripts/train_classifier.py --data training/deli_classifier_500.json --epochs 3

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from backend.config import get_settings
from backend.core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger("train_classifier")

LABEL_MAP = {"通用知识": 0, "专业咨询": 1}


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}


def main(data_file: str, epochs: int):
    settings = get_settings()
    # 底座必须 ASCII 路径（sentencepiece）；输出目录含中文路径无碍（纯 Python 保存）
    base_model = settings.path_from_root(settings.bert_base_model_path)
    if not Path(base_model).exists():
        base_model = "bert-base-chinese"
    output_dir = settings.classifier_abs_path

    with open(data_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    texts = [r["query"] for r in rows]
    labels = [LABEL_MAP[r["label"]] for r in rows]
    logger.info("dataset.loaded", samples=len(rows))

    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertForSequenceClassification.from_pretrained(
        base_model, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_t, val_t, train_y, val_y = train_test_split(
        texts, labels, test_size=0.2, random_state=42)
    train_enc = tokenizer(train_t, truncation=True, padding=True,
                          max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_t, truncation=True, padding=True,
                        max_length=128, return_tensors="pt")
    train_ds = _Dataset(train_enc, train_y)
    val_ds = _Dataset(val_enc, val_y)

    training_args = TrainingArguments(
        output_dir="./bert_results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir="./bert_logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        fp16=False,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("model.saved", output_dir=output_dir)

    preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    print(classification_report(val_y, preds,
                                target_names=["通用知识", "专业咨询"]))
    print(confusion_matrix(val_y, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(
        Path(get_settings().path_from_root("training/deli_classifier.json"))))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args.data, args.epochs)
