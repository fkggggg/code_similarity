import os
import json
from datasets import load_dataset, Dataset
from collections import defaultdict

# 配置路径
OUTPUT_DIR = '/data/lyy/code_similarity/datasets'
DATASET_CACHE_DIR = '/data/lyy/.cache/huggingface/datasets'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset_from_cache():
    """使用datasets库从本地缓存加载数据集"""
    try:
        dataset = load_dataset("ByteDance-Seed/Code-Contests-Plus", "default")
        return dataset
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return None

def process_data(train_data):
    """处理数据并按语言分组"""
    print("开始处理数据...")
    # 按语言分组案例
    language_groups = defaultdict(list)
    
    # 处理每个问题
    for idx, problem in enumerate(train_data):
        if idx % 10 == 0:
            print(f"处理进度: {idx}/{len(train_data)}")
        
        try:
            # 获取问题ID和提交
            problem_id = problem.get('id', f'problem_{idx}')
            correct_submissions = problem.get('correct_submissions', [])
            incorrect_submissions = problem.get('incorrect_submissions', [])
            
            # 确保它们是列表类型
            if not isinstance(correct_submissions, list):
                correct_submissions = []
            if not isinstance(incorrect_submissions, list):
                incorrect_submissions = []
            
            # 收集所有提交并按原始语言分组（不进行标准化）
            lang_submissions = defaultdict(lambda: {'correct': [], 'incorrect': []})
            
            # 处理正确提交
            for sub in correct_submissions:
                if isinstance(sub, dict) and 'language' in sub:
                    lang = sub['language']  # 直接使用原始语言名称
                    lang_submissions[lang]['correct'].append(sub)
            
            # 处理错误提交
            for sub in incorrect_submissions:
                if isinstance(sub, dict) and 'language' in sub:
                    lang = sub['language']  # 直接使用原始语言名称
                    lang_submissions[lang]['incorrect'].append(sub)
            
            # 为每种语言创建案例
            for lang, subs in lang_submissions.items():
                if subs['correct'] or subs['incorrect']:
                    # 从提交中提取代码内容
                    correct_code_list = []
                    for sub in subs['correct']:
                        code_content = sub.get('code', '')
                        correct_code_list.append({'code': code_content})
                    
                    incorrect_code_list = []
                    for sub in subs['incorrect']:
                        code_content = sub.get('code', '')
                        incorrect_code_list.append({'code': code_content})
                    
                    case = {
                        'case': problem_id,
                        'correct_submission': correct_code_list,
                        'incorrect_submission': incorrect_code_list
                    }
                    language_groups[lang].append(case)
                    
        except Exception as e:
            print(f"处理第 {idx} 个问题时出错: {str(e)}")
    
    print(f"数据处理完成，共处理 {len(train_data)} 个问题，找到 {len(language_groups)} 种语言")
    return language_groups

def generate_json_files(language_groups):
    """生成JSON文件"""
    print("开始生成JSON文件...")
    for language, cases in language_groups.items():
        if not cases:
            continue
        
        # 构建JSON结构
        json_data = {
            'language': language,
            'cases': cases
        }
        
        # 保存文件
        output_file = os.path.join(OUTPUT_DIR, f'dataset_test_{language}.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"已生成 {output_file}，包含 {len(cases)} 个案例")
        except Exception as e:
            print(f"生成 {language} JSON文件时出错: {str(e)}")

def main():
    print("开始处理数据集并生成JSON...")
    
    # 从本地缓存加载数据集
    dataset_dict = load_dataset_from_cache()
    
    if dataset_dict is not None:
        # 检查是否为DatasetDict对象
        if hasattr(dataset_dict, 'keys'):
            print(f"数据集包含以下分割: {list(dataset_dict.keys())}")
            # 从train分割获取数据
            if 'train' in dataset_dict:
                train_data = dataset_dict['train']
                total_records = len(train_data)
                print(f"train分割包含 {total_records} 条记录")
                
                # 显示特征信息
                if hasattr(train_data, 'features'):
                    print(f"特征列表: {list(train_data.features.keys())}")
                
                # 处理完整数据集
                language_groups = process_data(train_data)
                
                # 生成JSON文件
                generate_json_files(language_groups)
                
                print("\n所有处理完成！")
                print(f"所有文件已保存到: {OUTPUT_DIR}")
            else:
                print("错误：数据集中不包含train分割")
        else:
            # 如果不是DatasetDict，尝试直接使用完整数据集
            total_records = len(dataset_dict)
            print(f"数据集中包含 {total_records} 条记录")
            
            # 处理完整数据集
            language_groups = process_data(dataset_dict)
            
            # 生成JSON文件
            generate_json_files(language_groups)
            
            print("\n所有处理完成！")
            print(f"所有文件已保存到: {OUTPUT_DIR}")
    else:
        print("无法加载数据集，程序退出")

if __name__ == "__main__":
    main()