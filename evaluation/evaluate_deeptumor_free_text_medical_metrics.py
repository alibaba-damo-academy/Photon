import json

# --------------------------
# 配置：输入文件路径
# --------------------------
file1_path = "/mnt/bj/fangchengyu.fcy/ms-swift/output/FINAL/DEEPTUMOR_FREE_TEXT/3B-140280280/v0-20250916-170427/fp32/infer_result/20250917-062818.jsonl"  # 预测结果
file2_path = "/mnt/bj/fangchengyu.fcy/Datasets/DeepTumorVQA_1.0/val_free_text.jsonl"  # 第二个文件（题目类型和更多信息）

# 目标子类型
TARGET_SUBTYPES = {
    "liver_lesion_existence",
    "kidney_tumor_existence",
    "pancreatic_lesion_existence",
}

# --------------------------
# 工具函数
# --------------------------
def to_bool(x):
    """将label或预测解析为布尔值（存在=1/True，不存在=0/False）。无法解析返回None。"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return True if x > 0 else False if x == 0 else None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return to_bool(x[0])
    s = str(x).strip().lower().strip(" .,:;!?，。；：！？")
    pos = {"yes","true","present","positive","exist","exists","有","是","存在","阳性"}
    neg = {"no","false","absent","negative","not exist","无","否","不存在","阴性","未见"}
    if s in pos: return True
    if s in neg: return False
    if s == "1": return True
    if s == "0": return False
    return None

def is_recognize(q_type):
    return isinstance(q_type, str) and q_type.lower().startswith("recognition")

def safe_div(n, d):
    return n / d if d else 0.0

# --------------------------
# 读取数据
# --------------------------
with open(file1_path, "r", encoding="utf-8") as f1, open(file2_path, "r", encoding="utf-8") as f2:
    data1 = [json.loads(line) for line in f1]
    data2 = [json.loads(line) for line in f2]

assert len(data1) == len(data2), "两个文件行数不一致"

# --------------------------
# 统计混淆矩阵
# --------------------------
counts = {sub: {"TP":0,"TN":0,"FP":0,"FN":0,"SKIP":0} for sub in TARGET_SUBTYPES}

for d1, d2 in zip(data1, data2):
    q_type = d2.get("question type")
    q_subtype = d2.get("question subtype")
    if not is_recognize(q_type) or q_subtype not in TARGET_SUBTYPES:
        continue
    y_true = to_bool(d1.get("labels"))
    y_pred = to_bool(d1.get("response"))
    if y_true is None or y_pred is None:
        counts[q_subtype]["SKIP"] += 1
        continue
    if y_true and y_pred: counts[q_subtype]["TP"] += 1
    elif not y_true and not y_pred: counts[q_subtype]["TN"] += 1
    elif not y_true and y_pred: counts[q_subtype]["FP"] += 1
    elif y_true and not y_pred: counts[q_subtype]["FN"] += 1

# --------------------------
# 计算指标
# --------------------------
print("=== Recognize 子类型结果 ===")
macro_sens, macro_spec, macro_acc = [], [], []
micro_tp = micro_tn = micro_fp = micro_fn = 0

for sub, c in counts.items():
    TP, TN, FP, FN = c["TP"], c["TN"], c["FP"], c["FN"]
    used = TP + TN + FP + FN
    sens = safe_div(TP, TP+FN)
    spec = safe_div(TN, TN+FP)
    acc  = safe_div(TP+TN, used)
    print(f"{sub}: TP={TP}, TN={TN}, FP={FP}, FN={FN}, SKIP={c['SKIP']}, Used={used}")
    print(f"  Sensitivity = {sens:.2%}, Specificity = {spec:.2%}, Accuracy = {acc:.2%}")
    if used > 0:
        macro_sens.append(sens); macro_spec.append(spec); macro_acc.append(acc)
    micro_tp += TP; micro_tn += TN; micro_fp += FP; micro_fn += FN

# 宏平均
if macro_sens:
    print("\n=== Macro Average（子类型等权） ===")
    print(f"Sensitivity = {sum(macro_sens)/len(macro_sens):.2%}")
    print(f"Specificity = {sum(macro_spec)/len(macro_spec):.2%}")
    print(f"Accuracy    = {sum(macro_acc)/len(macro_acc):.2%}")

# 微平均
total_used = micro_tp + micro_tn + micro_fp + micro_fn
if total_used > 0:
    print("\n=== Micro Average（样本加权） ===")
    print(f"Sensitivity = {safe_div(micro_tp, micro_tp+micro_fn):.2%}")
    print(f"Specificity = {safe_div(micro_tn, micro_tn+micro_fp):.2%}")
    print(f"Accuracy    = {safe_div(micro_tp+micro_tn, total_used):.2%}")