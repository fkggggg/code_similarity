import json
import sys
import os
from CGED import BatchCalculate as CGED

def load_dataset(dataset_path):
    """
    加载数据集文件
    
    参数:
        dataset_path (str): 数据集文件路径
    
    返回:
        dict: 数据集内容
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_similarity_results(results_path):
    """
    加载相似度结果文件
    
    参数:
        results_path (str): 相似度结果文件路径
    
    返回:
        dict: 相似度结果内容
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_similarity_results(results, output_path):
    """
    保存相似度结果到文件
    
    参数:
        results (dict): 相似度结果
        output_path (str): 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def update_cged_scores(dataset_path, results_path, output_path):
    """
    更新相似度结果中的CGED分数（使用批量计算提高效率）
    
    参数:
        dataset_path (str): 数据集文件路径
        results_path (str): 原始相似度结果文件路径
        output_path (str): 更新后的结果输出路径
    """
    # 加载数据集和相似度结果
    dataset = load_dataset(dataset_path)
    results = load_similarity_results(results_path)
    
    # 获取语言信息
    language = dataset.get('language', 'python')
    # 如果是py3则映射为python
    if language == 'py3':
        language = 'python'
    # 如果是cpp则映射为c++
    elif language == 'cpp':
        language = 'cpp'
    
    # 获取结果列表
    results_list = results.get('results', [])
    
    print(f"开始更新 {len(results_list)} 个代码对的CGED分数...")
    
    # 准备批量计算的数据
    code_pairs = []
    language_pairs = []
    
    for i, result in enumerate(results_list):
        src_code = result['src']['code']
        dst_code = result['dst']['code']
        code_pairs.append((src_code, dst_code))
        language_pairs.append((language, language))
        # 显示准备数据的进度
        if (i + 1) % 50 == 0 or (i + 1) == len(results_list):
            print(f"  已准备 {i+1}/{len(results_list)} 个代码对数据")
    
    # 批量计算CGED分数
    print(f"正在进行批量CGED计算，共 {len(code_pairs)} 个代码对...")
    try:
        cged_scores = CGED(
            code_src_dst_list=code_pairs,
            language_src_dst_list=language_pairs,
            src_main_func_names = None,
            dst_main_func_names = None,
            src_update_cache=True,
            dst_update_cache=False,
            pdg_parallelism=10,
            nx_parallelism=60,
            nx_budget=20,
            verbose_level=1,  # 设置为1以显示更多进度信息
        )
        
        # 检查返回结果
        if not cged_scores or not isinstance(cged_scores, list):
            print("CGED计算返回结果异常")
            return
        elif len(cged_scores) != len(code_pairs):
            print(f"CGED计算结果数量不匹配: 期望 {len(code_pairs)}, 实际 {len(cged_scores)}")
            return
    except Exception as e:
        print(f"CGED批量计算出错: {e}")
        return
    
    # 更新结果中的CGED分数
    for i, (result, cged_score) in enumerate(zip(results_list, cged_scores)):
        result['similarity']['cged'] = cged_score
        if (i + 1) % 10 == 0 or (i + 1) == len(results_list):  # 每10个或最后一条打印一次进度
            print(f"  已更新 {i+1}/{len(results_list)} 个CGED分数")
    
    print(f"  所有CGED分数已更新完成")
    
    # 保存更新后的结果
    save_similarity_results(results, output_path)
    print(f"已保存更新后的结果到: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使用方法: python get_similarity_fix.py <dataset_path> <results_path> <output_path>")
        print("示例: python get_similarity_fix.py /data/lyy/code_similarity/datasets_old/dataset_test_py3_subset.json /data/lyy/code_similarity/similarity_results/20251128_132006/pos_pairs_similarity_results.json /data/lyy/code_similarity/pos_pairs_similarity_results_fixed.json")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    results_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 检查输入文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件 {dataset_path} 不存在")
        sys.exit(1)
    
    if not os.path.exists(results_path):
        print(f"错误: 结果文件 {results_path} 不存在")
        sys.exit(1)
    
    # 执行更新操作
    update_cged_scores(dataset_path, results_path, output_path)