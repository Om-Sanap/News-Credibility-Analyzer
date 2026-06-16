# train_hf_from_true_false.py
# Usage:
#   python train_hf_from_true_false.py --true_csv true.csv --false_csv false.csv --model_id distilroberta-base

import argparse, os, json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def main(args):
    df_true  = pd.read_csv(args.true_csv)
    df_false = pd.read_csv(args.false_csv)

    # Ensure there is a 'text' column; if not, use the first column as text
    if "text" not in df_true.columns:
        df_true = df_true.rename(columns={df_true.columns[0]: "text"})
    if "text" not in df_false.columns:
        df_false = df_false.rename(columns={df_false.columns[0]: "text"})

    df_true["label"]  = "real"
    df_false["label"] = "fake"

    df = pd.concat([df_true[["text","label"]], df_false[["text","label"]]], ignore_index=True).dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)
    raw = DatasetDict({"train": train_ds, "test": test_ds})

    labels = sorted(raw["train"].unique("label"))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(args.model_id)
    max_len = args.max_len

    def encode_labels(example):
        example["labels"] = label2id[example["label"]]
        return example

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    data = raw.map(encode_labels)
    data = data.map(tokenize, batched=True, remove_columns=[c for c in data["train"].column_names if c not in ("input_ids","attention_mask","labels")])
    data.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    accuracy  = evaluate.load("accuracy")
    f1        = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall    = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy":  accuracy.compute(predictions=y_pred, references=y_true)["accuracy"],
            "f1":        f1.compute(predictions=y_pred, references=y_true, average="weighted")["f1"],
            "precision": precision.compute(predictions=y_pred, references=y_true, average="weighted")["precision"],
            "recall":    recall.compute(predictions=y_pred, references=y_true, average="weighted")["recall"],
        }

    args_out = "hf_fake_news"
    os.makedirs(args_out, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=args_out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",       # <- renamed in v5
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        bf16=args.bf16,
        metric_for_best_model="f1",
        greater_is_better=True,
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tok,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    save_dir = os.path.join(args_out, "best")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "labels.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)

    print("Saved model to", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_csv", type=str, required=True)
    parser.add_argument("--false_csv", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="distilroberta-base",
                        help="Try 'xlm-roberta-base' for multilingual data")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    main(args)
