import json
import random

# 设置随机种子以确保结果可复现
random.seed(42)

def create_test_subset(input_file, output_file, num_cases=10):
    """
    从输入JSON文件中提取指定数量的case创建测试子集
    
    参数:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
        num_cases (int): 要提取的case数量
    """
    try:
        # 读取原始数据
        print(f"正在读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取所有case
        cases = data.get('cases', [])
        print(f"原始数据中共有 {len(cases)} 个case")
        
        # 如果case数量不足，直接使用全部case
        if len(cases) <= num_cases:
            print(f"警告: 原始数据中的case数量 ({len(cases)}) 小于请求的数量 ({num_cases})，将使用全部case")
            selected_cases = cases
        else:
            # 随机选择num_cases个case
            selected_cases = random.sample(cases, num_cases)
            print(f"已随机选择 {len(selected_cases)} 个case")
        
        # 创建新的数据结构，保留原始数据的其他字段
        subset_data = {
            'language': data.get('language', 'unknown'),
            'cases': selected_cases
        }
        
        # 复制原始数据中的其他字段
        for key, value in data.items():
            if key not in subset_data:
                subset_data[key] = value
        
        # 保存测试子集
        print(f"正在保存测试子集到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, ensure_ascii=False, indent=2)
        
        print("测试子集创建成功！")
        print(f"输出文件: {output_file}")
        print(f"包含 {len(selected_cases)} 个case")
        
        # 统计每个case的提交数量
        total_submissions = 0
        for idx, case in enumerate(selected_cases):
            case_id = case.get('case', f'case_{idx}')
            correct_submissions = case.get('correct_submission', [])
            submission_count = len(correct_submissions)
            total_submissions += submission_count
            print(f"  Case {case_id}: {submission_count} 个正确提交")
        
        print(f"\n总共 {total_submissions} 个正确提交")
        
    except Exception as e:
        print(f"创建测试子集时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "/data/lyy/code_similarity/datasets_old/dataset_test_py3.json"
    output_file = "/data/lyy/code_similarity/datasets_old/dataset_test_py3_subset.json"
    
    # 创建测试子集
    create_test_subset(input_file, output_file, num_cases=10)