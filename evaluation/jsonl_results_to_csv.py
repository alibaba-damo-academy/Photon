#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import argparse
from collections import defaultdict

# 与评估脚本一致
TASK = {
    'task1': ['Anatomical_observation', 'Pathological_observation'],
    'task2': ['Abnormality_feature', 'Abnormality_position', 'Abnormality_type', 'Diagnosis'],
    'task3': ['Diameter', 'Size', 'Thickness'],
    'task4': ['Arterial wall calcification', 'Atelectasis', 'Bronchiectasis', 'Cardiomegaly', 'Consolidation',
              'Coronary artery wall calcification', 'Emphysema', 'Hiatal hernia', 'Interlobular septal thickening',
              'Lung nodule', 'Lung opacity', 'Lymphadenopathy', 'Medical material', 'Mosaic attenuation pattern',
              'Peribronchial thickening', 'Pericardial effusion', 'Pleural effusion', 'Pulmonary fibrotic sequela'],
    'task5': ['b', 'c', 'd', 'e', 'f', 'g', 'h'],
    'task6': ['b', 'c', 'd', 'e', 'f', 'g', 'h'],
}

_spc_re = re.compile(r"\s+")
def norm_space(s: str) -> str:
    return _spc_re.sub(" ", str(s or "").strip())

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}:{ln}\n{e}")
            obj["_line_no"] = ln
            items.append(obj)
    return items

def first_video(rec) -> str:
    vids = rec.get("videos", [])
    return str(vids[0]).strip() if vids else ""

def get_question(rec) -> str:
    msgs = rec.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return norm_space(m.get("content", ""))
    return norm_space(rec.get("Question", ""))

def task_and_sub_from_label_rec(rec):
    t_id = int(rec["Task"])
    s_id = int(rec["Subtask"])
    task_name = f"task{t_id}"
    sub_name = TASK[task_name][s_id - 1]
    return task_name, sub_name

def qtype_from_label_rec(rec) -> str:
    qt = rec.get("QuestionType", None)
    if qt is not None:
        q = str(qt).strip().lower()
        if q.startswith("open"):
            return "open"
        if q.startswith("close"):
            return "close"
    task_name, _ = task_and_sub_from_label_rec(rec)
    return "open" if task_name in {"task1", "task2", "task3", "task4"} else "close"

def pred_text_from_pred_rec(rec) -> str:
    if rec.get("response", None) is not None:
        return str(rec["response"]).strip()
    msgs = rec.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return str(m.get("content", "")).strip()
    if "Pred" in rec:
        return str(rec["Pred"]).strip()
    return ""

def answer_from_pred_rec(rec) -> str:
    # 评估用 Answer，按你的要求：来自预测文件中的 labels
    if "labels" in rec and rec["labels"] is not None:
        return str(rec["labels"]).strip()
    if "Answer" in rec and rec["Answer"] is not None:
        return str(rec["Answer"]).strip()
    return ""

def write_csv(rows, path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser(
        description="按顺序对齐标签与预测（行对行）。标签仅补充 Task/Subtask/QuestionType；一预测→一行。"
    )
    ap.add_argument("--labels_jsonl", default='/mnt/bj/fangchengyu.fcy/Datasets/3D-RAD/test.jsonl', help="标签 jsonl 文件路径")
    ap.add_argument("--preds_jsonl", default='/mnt/bj/fangchengyu.fcy/Photon/output/3D-RAD/3B-FINAL/v1-20251107-185924/fp32/infer_result/20251110-081345.jsonl', help="预测 jsonl 文件路径")
    ap.add_argument("--out_root",    default="3D-RAD-Results", help="输出根目录（会写到 results/{model}/...）")
    ap.add_argument("--model",       default="3B", help="模型名（用于 results/{model}）")
    ap.add_argument("--save_debug", action="store_true",
                    help="CSV 附加 Video/Question/LabelFromPred 便于排查")
    args = ap.parse_args()

    labels = read_jsonl(args.labels_jsonl)
    preds  = read_jsonl(args.preds_jsonl)

    n_lab = len(labels)
    n_pred = len(preds)
    n = min(n_lab, n_pred)
    if n_lab != n_pred:
        print(f"[Warn] 行数不一致：labels={n_lab}, preds={n_pred}。将仅处理前 {n} 行。")

    buckets_open  = defaultdict(list)  # (task_name, sub_name) -> rows
    buckets_close = defaultdict(list)

    for lab, pr in zip(labels[:n], preds[:n]):
        task_name, sub_name = task_and_sub_from_label_rec(lab)
        qtype = qtype_from_label_rec(lab)

        pred_txt = pred_text_from_pred_rec(pr)
        ans_pred = answer_from_pred_rec(pr)  # 评估 Answer 来自预测文件 labels（你的要求）

        if qtype == "open":
            row = {"Pred": pred_txt, "Answer": ans_pred}
            if args.save_debug:
                row["Video"] = first_video(pr)
                row["Question"] = get_question(lab)
                row["LabelFromPred"] = ans_pred
            buckets_open[(task_name, sub_name)].append(row)
        else:
            correct = int(norm_space(pred_txt).lower() == norm_space(ans_pred).lower())
            row = {"Correct": correct}
            if args.save_debug:
                row["Pred"] = pred_txt
                row["Answer"] = ans_pred
                row["Video"] = first_video(pr)
                row["Question"] = get_question(lab)
                row["LabelFromPred"] = ans_pred
            buckets_close[(task_name, sub_name)].append(row)

    out_root = os.path.join(args.out_root, args.model)
    os.makedirs(out_root, exist_ok=True)

    for (task_name, sub_name), rows in buckets_open.items():
        path = os.path.join(out_root, task_name, sub_name, "eval_open_vqa.csv")
        fields = ["Pred", "Answer"] + (["Video", "Question", "LabelFromPred"] if args.save_debug else [])
        write_csv(rows, path, fields)

    for (task_name, sub_name), rows in buckets_close.items():
        path = os.path.join(out_root, task_name, sub_name, "eval_close_vqa.csv")
        fields = ["Correct"] + (["Pred", "Answer", "Video", "Question", "LabelFromPred"] if args.save_debug else [])
        write_csv(rows, path, fields)

    print(f"完成。输出根目录: {out_root}")
    print(f"已处理行数: {n}（labels={n_lab}, preds={n_pred}）。")

if __name__ == "__main__":
    main()