from datasets import load_dataset
import random
import os
from collections import defaultdict

# 加载数据集
dataset = load_dataset("ByteDance-Seed/Code-Contests-Plus", "default")

# 获取训练集
train_data = dataset["train"]

# 随机选取一个case
random_case = random.choice(train_data)
case_id = random_case['id']

# 创建结果目录
output_dir = f"/data/lyy/code_similarity/results/{case_id}"
os.makedirs(output_dir, exist_ok=True)

# 创建主信息文件
main_file = os.path.join(output_dir, f"case_{case_id}_info.txt")
with open(main_file, "w", encoding="utf-8") as f:
    # 写入case信息
    f.write("==== CASE INFORMATION ====\n")
    f.write(f"ID: {case_id}\n")
    f.write(f"Title: {random_case['title']}\n")
    f.write(f"Description: {random_case['description']}\n")
    f.write(f"Time Limit: {random_case['time_limit']}\n")
    f.write(f"Memory Limit: {random_case['memory_limit']}\n")
    f.write("\n")
    f.write(f"Correct Submissions: {len(random_case['correct_submissions'])}\n")
    f.write(f"Incorrect Submissions: {len(random_case['incorrect_submissions'])}\n")

# 按语言分组提交
correct_by_lang = defaultdict(list)
incorrect_by_lang = defaultdict(list)

# 分组正确提交
for submission in random_case['correct_submissions']:
    lang = submission['language']
    correct_by_lang[lang].append(submission)

# 分组错误提交
for submission in random_case['incorrect_submissions']:
    lang = submission['language']
    incorrect_by_lang[lang].append(submission)

# 保存按语言分类的正确提交
for lang, submissions in correct_by_lang.items():
    lang_file = os.path.join(output_dir, f"case_{case_id}_correct_{lang}.txt")
    with open(lang_file, "w", encoding="utf-8") as f:
        f.write(f"==== CORRECT SUBMISSIONS IN {lang.upper()} (Total: {len(submissions)}) ====\n\n")
        for i, submission in enumerate(submissions, 1):
            f.write(f"--- Correct Submission {i} ({submission['language']}) ---")
            f.write("\n\n")
            f.write(submission['code'])
            f.write("\n\n")
    print(f"已保存 {lang} 语言的正确提交到: {lang_file}")

# 保存按语言分类的错误提交
for lang, submissions in incorrect_by_lang.items():
    lang_file = os.path.join(output_dir, f"case_{case_id}_incorrect_{lang}.txt")
    with open(lang_file, "w", encoding="utf-8") as f:
        f.write(f"==== INCORRECT SUBMISSIONS IN {lang.upper()} (Total: {len(submissions)}) ====\n\n")
        for i, submission in enumerate(submissions, 1):
            f.write(f"--- Incorrect Submission {i} ({submission['language']}) ---")
            f.write("\n\n")
            f.write(submission['code'])
            f.write("\n\n")
    print(f"已保存 {lang} 语言的错误提交到: {lang_file}")

# 打印总结信息
print(f"\nCase ID: {case_id}")
print(f"总正确提交数: {len(random_case['correct_submissions'])}")
print(f"总错误提交数: {len(random_case['incorrect_submissions'])}")
print(f"正确提交语言种类: {list(correct_by_lang.keys())}")
print(f"错误提交语言种类: {list(incorrect_by_lang.keys())}")
print(f"\n所有文件已保存到目录: {output_dir}")
print(f"Case信息已保存到: {main_file}")