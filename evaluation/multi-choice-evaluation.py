import json
from collections import defaultdict


# 文件路径
file1_path = "/mnt/beijing_fast/fangchengyu.fcy/Photon/pretrained_weights/DeeptumorVQA-Multichoice/infer_result/20260324-072145.jsonl"  # 第一个文件（预测结果）
file2_path = "/mnt/beijing/fangchengyu.fcy/Datasets/DeepTumorVQA_1.0/val_multi_choice.jsonl"  # 第二个文件（题目类型和更多信息）

# 读取两个文件
with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
    data1 = [json.loads(line) for line in f1]
    data2 = [json.loads(line) for line in f2]

assert len(data1) == len(data2), "两个文件行数不一致"

# （q_type, q_subtype） -> 统计 correct/total
stats = {}
for d1, d2 in zip(data1, data2):
    q_type = d2.get("question type")
    q_subtype = d2.get("question subtype")
    pred = d1.get("response")
    label = d1.get("labels")

    key = (q_type, q_subtype)
    if key not in stats:
        stats[key] = {"correct": 0, "total": 0}
    
    if pred == label:
        stats[key]["correct"] += 1
    stats[key]["total"] += 1

# 获取 question type 出现顺序（保持第二文件中的顺序）
q_type_order = []
for d in data2:
    q_type = d.get("question type")
    if q_type not in q_type_order:
        q_type_order.append(q_type)

# 按 question type 顺序 + subtype 名称排序
sorted_keys = sorted(stats.keys(), key=lambda x: (q_type_order.index(x[0]), x[1]))

# 按大类统计准确率
type_acc_sum = defaultdict(float)  # q_type -> 累积准确率之和
type_subtype_count = defaultdict(int)  # q_type -> subtype 数量

print("=== 每个 subtype 的准确率 ===")
all_subtype_accs = []  # 用于 Total Average（macro-subtype）
for (q_type, q_subtype) in sorted_keys:
    correct = stats[(q_type, q_subtype)]["correct"]
    total = stats[(q_type, q_subtype)]["total"]
    acc = correct / total if total else 0.0
    print(f"{q_type} - {q_subtype}: {acc:.2%} ({correct}/{total})")

    type_acc_sum[q_type] += acc
    type_subtype_count[q_type] += 1
    all_subtype_accs.append(acc)

print("\n=== 每个 question type 的平均准确率 ===")
for q_type in q_type_order:
    if type_subtype_count[q_type] > 0:
        avg_acc = type_acc_sum[q_type] / type_subtype_count[q_type]
        print(f"{q_type}: {avg_acc:.2%}  (基于 {type_subtype_count[q_type]} 个 subtype 平均计算)")

# 仅按表格口径：跨所有 subtype 等权的 Total Average
if all_subtype_accs:
    total_avg = sum(all_subtype_accs) / len(all_subtype_accs)
    print(f"\n=== Total Average (macro-subtype) ===\n{total_avg:.2%}")