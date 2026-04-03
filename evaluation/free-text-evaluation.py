import json
import re
from collections import defaultdict

# --------------------------
# 配置
# --------------------------
file1_path = "/mnt/beijing_fast/fangchengyu.fcy/Photon/pretrained_weights/DeeptumorVQA-Freetext/infer_result/20260324-072712.jsonl"  # 预测结果
file2_path = "/mnt/beijing/fangchengyu.fcy/Datasets/DeepTumorVQA_1.0/val_free_text.jsonl"  # 第二个文件（题目类型和更多信息）

# 需要使用 MRA 的数值类子类型
NUMERIC_MRA_SUBTYPES = {
    "lesion volume measurement",
    "organ HU measurement",
    "organ volume measurement",
    "largest lesion diameter",
    "largest lesion slice",
    "lesion count by location",
    "lesion counting",
    "organ aggregation",
    "tumor organ HU difference",
}

# MRA 阈值集合：{0.50, 0.55, ..., 0.95}
MRA_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# --------------------------
# 工具函数
# --------------------------
_num_pattern = re.compile(
    r"""
    [+-]?                # 可选符号
    (?:
        (?:\d+\.\d*)|    # 12. 或 12.34
        (?:\.\d+)|       # .34
        (?:\d+)          # 整数
    )
    (?:[eE][+-]?\d+)?    # 科学计数
    """,
    re.VERBOSE,
)

def parse_number(x):
    """从字符串或数字中解析第一个浮点数；无法解析返回 None。"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        x = str(x)
    m = _num_pattern.search(x)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def mra_score(pred, label, eps=1e-6):
    """
    单题 MRA 分数（0~1，步长 0.05）。解析数值，计算相对误差，并在各阈值下打分后取均值。
    任一侧非数值则返回 0.0。
    """
    y_hat = parse_number(pred)
    y = parse_number(label)
    if y_hat is None or y is None:
        return 0.0
    denom = max(abs(y), eps)
    rel_err = abs(y_hat - y) / denom
    hits = sum(1 for theta in MRA_THRESHOLDS if rel_err < (1.0 - theta))
    return hits / len(MRA_THRESHOLDS)

# --------------------------
# 读取数据
# --------------------------
with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
    data1 = [json.loads(line) for line in f1]
    data2 = [json.loads(line) for line in f2]

assert len(data1) == len(data2), "两个文件行数不一致"

# --------------------------
# 统计
# 非数值子类型：ACC（correct/total）
# 数值子类型：MRA（sum_mra/total）
# --------------------------
stats = {}
for d1, d2 in zip(data1, data2):
    q_type = d2.get("question type")
    q_subtype = d2.get("question subtype")
    pred = d1.get("response")
    label = d1.get("labels")

    key = (q_type, q_subtype)
    if key not in stats:
        if q_subtype in NUMERIC_MRA_SUBTYPES:
            stats[key] = {"metric": "MRA", "sum_mra": 0.0, "total": 0}
        else:
            stats[key] = {"metric": "ACC", "correct": 0, "total": 0}

    if q_subtype in NUMERIC_MRA_SUBTYPES:
        score = mra_score(pred, label)
        stats[key]["sum_mra"] += score
        stats[key]["total"] += 1
    else:
        is_correct = (pred == label)
        stats[key]["correct"] = stats[key].get("correct", 0) + (1 if is_correct else 0)
        stats[key]["total"] += 1

# --------------------------
# question type 出现顺序（按第二文件）
# --------------------------
q_type_order = []
for d in data2:
    q_type = d.get("question type")
    if q_type not in q_type_order:
        q_type_order.append(q_type)

# 按 question type 顺序 + subtype 名称排序
sorted_keys = sorted(stats.keys(), key=lambda x: (q_type_order.index(x[0]), x[1]))

# --------------------------
# 汇总与打印
# type 级别平均：子类型简单平均（ACC 用 accuracy，MRA 用平均 MRA）
# Total Average：跨所有子类型的简单平均（macro-subtype，表格口径）
# --------------------------
type_metric_sum = defaultdict(float)
type_subtype_count = defaultdict(int)
all_subtype_scores = []  # 用于 Total Average（macro-subtype）

print("=== 每个 subtype 的指标 ===")
for (q_type, q_subtype) in sorted_keys:
    stat = stats[(q_type, q_subtype)]
    if stat["metric"] == "MRA":
        total = stat["total"]
        avg_mra = (stat["sum_mra"] / total) if total else 0.0
        print(f"{q_type} - {q_subtype}: MRA = {avg_mra:.2%} (样本 {total})")
        type_metric_sum[q_type] += avg_mra
        type_subtype_count[q_type] += 1
        all_subtype_scores.append(avg_mra)   # 收集子类型分数
    else:
        correct = stat["correct"]
        total = stat["total"]
        acc = correct / total if total else 0.0
        print(f"{q_type} - {q_subtype}: ACC = {acc:.2%} ({correct}/{total})")
        type_metric_sum[q_type] += acc
        type_subtype_count[q_type] += 1
        all_subtype_scores.append(acc)       # 收集子类型分数

print("\n=== 每个 question type 的平均得分（子类型简单平均） ===")
for q_type in q_type_order:
    if type_subtype_count[q_type] > 0:
        avg_score = type_metric_sum[q_type] / type_subtype_count[q_type]
        print(f"{q_type}: {avg_score:.2%}  (基于 {type_subtype_count[q_type]} 个 subtype)")

# 总平均：跨所有子类型等权（macro-subtype，表格口径）
if all_subtype_scores:
    total_avg = sum(all_subtype_scores) / len(all_subtype_scores)
    print(f"\n=== Total Average (macro-subtype) ===\n{total_avg:.2%}")