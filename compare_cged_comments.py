import json
import random
import re
from CGED import BatchCalculate as CGED
from tqdm import tqdm

# 设置随机种子以确保可复现性
random.seed(42)

def extract_case_cged_values(file_path):
    """从txt文件中提取所有case的CGED值"""
    case_cged_map = {}
    current_case = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配case行
            case_match = re.search(r'处理case ([^:]+):', line)
            if case_match:
                current_case = case_match.group(1)
            
            # 匹配CGED值行
            cged_match = re.search(r'CGED: ([0-9.]+)', line)
            if cged_match and current_case:
                case_cged_map[current_case] = float(cged_match.group(1))
    
    return case_cged_map

def extract_random_code_pairs_from_case(case):
    """从单个case中随机提取代码对"""
    correct_submissions = case.get('correct_submission', [])
    code_list = [sub.get('code', '') for sub in correct_submissions if 'code' in sub]
    code_list = [code for code in code_list if code.strip()]
    
    # 确保至少有2个代码提交来生成代码对
    if len(code_list) >= 2:
        # 随机选择两个不同的下标
        idx1 = random.randint(0, len(code_list) - 1)
        # 确保idx2与idx1不同
        idx2 = random.randint(0, len(code_list) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(code_list) - 1)
        
        return [(code_list[idx1], code_list[idx2])], idx1, idx2
    return [], None, None

def find_matching_case_with_same_indices(case_id, data_no_comments, idx1, idx2):
    """在无注释数据集中查找匹配的case并使用相同下标提取代码对"""
    for case in data_no_comments.get('cases', []):
        if case.get('case') == case_id:
            correct_submissions = case.get('correct_submission', [])
            code_list = [sub.get('code', '') for sub in correct_submissions if 'code' in sub]
            code_list = [code for code in code_list if code.strip()]
            
            # 确保下标有效
            if idx1 is not None and idx2 is not None and len(code_list) > max(idx1, idx2):
                return [(code_list[idx1], code_list[idx2])]
    return []

def main():
    # 文件路径
    file_with_comments = '/data/lyy/code_similarity/datasets/dataset_test_py3.json'
    file_no_comments = '/data/lyy/code_similarity/datasets/dataset_test_py3_no_comments.json'
    txt_with_comments = '/data/lyy/code_similarity/with_comment.txt'
    txt_no_comments = '/data/lyy/code_similarity/no_commant.txt'
    output_file = '/data/lyy/code_similarity/cged_diff_results.txt'
    
    print("第一步：从txt文件中提取case的CGED值并找出差异...")
    # 从txt文件中提取CGED值
    with_comment_cged = extract_case_cged_values(txt_with_comments)
    no_comment_cged = extract_case_cged_values(txt_no_comments)
    
    # 找出CGED值不相等的case
    cased_with_different_cged = []
    for case_id in with_comment_cged:
        if case_id in no_comment_cged:
            cged_with = with_comment_cged[case_id]
            cged_without = no_comment_cged[case_id]
            if abs(cged_with - cged_without) > 1e-6:
                cased_with_different_cged.append({
                    'case_id': case_id,
                    'cged_with_comment': cged_with,
                    'cged_without_comment': cged_without,
                    'difference': abs(cged_with - cged_without)
                })
    
    print(f"在txt文件中找到 {len(cased_with_different_cged)} 个case的CGED值存在差异")
    for item in cased_with_different_cged:
        print(f"  case {item['case_id']}: 有注释CGED={item['cged_with_comment']:.4f}, 无注释CGED={item['cged_without_comment']:.4f}, 差异={item['difference']:.4f}")
    
    # 读取JSON数据
    print("\n第二步：读取JSON数据并在有差异的case中寻找具体代码对...")
    with open(file_with_comments, 'r', encoding='utf-8') as f:
        data_with_comments = json.load(f)
    
    with open(file_no_comments, 'r', encoding='utf-8') as f:
        data_no_comments = json.load(f)
    
    # 获取语言信息
    language = data_with_comments.get('language', 'python')
    if language == 'py3':
        language = 'python'
    
    # 遍历有差异的case进行详细分析
    max_differences = 5
    differences_found = False
    results_to_write = []
    processed_cases = 0
    
    try:
        # 为每个有差异的case进行详细分析
        for diff_item in cased_with_different_cged:
            if len(results_to_write) >= max_differences:
                print(f"已找到 {max_differences} 处差异，停止检测")
                break
                
            case_id = diff_item['case_id']
            # 在有注释数据集中查找对应的case
            target_case = None
            for case in data_with_comments.get('cases', []):
                if case.get('case') == case_id:
                    target_case = case
                    break
            
            if not target_case:
                print(f"  在有注释数据集中未找到case {case_id}")
                continue
                
            processed_cases += 1
            print(f"\n处理case {processed_cases}: {case_id}")
            print(f"txt文件中的CGED差异: {diff_item['difference']:.4f}")
            
            # 从有注释case中随机提取代码对（使用相同的随机下标）
            code_pairs_with_comments, idx1, idx2 = extract_random_code_pairs_from_case(target_case)
            if not code_pairs_with_comments:
                print("  无法提取足够的代码对")
                continue
                
            print(f"  使用随机下标 {idx1} 和 {idx2} 提取代码对")
            
            # 在无注释数据集中使用相同下标提取代码对
            code_pairs_no_comments = find_matching_case_with_same_indices(case_id, data_no_comments, idx1, idx2)
            
            if not code_pairs_no_comments:
                print(f"  警告：在无注释数据集中无法找到匹配的代码对")
                continue
                
            # 合并需要计算的代码对，用于批量计算
            all_code_pairs = []
            # 按照顺序：有注释的代码对，无注释的代码对
            all_code_pairs.append(code_pairs_with_comments[0])
            all_code_pairs.append(code_pairs_no_comments[0])
            
            print(f"  准备批量计算 {len(all_code_pairs)} 对代码的CGED相似度")
            
            # 计算当前代码对的CGED相似度
            cged_scores = CGED(
                code_origin_target_list=all_code_pairs,
                language_origin_target_list=[(language, language)] * len(all_code_pairs),
                origin_main_func_names=None,
                target_main_func_names=None,
                origin_update_cache=True,
                target_update_cache=False,
                nx_parallelism=60,
                nx_budget=20,
                use_tmp_cache=False,
                pdg_parallelism=10,
                verbose_level=0
            )
        
            # 检查结果长度是否匹配
            if len(cged_scores) != len(all_code_pairs):
                print(f"  警告：CGED返回的分数数量({len(cged_scores)})与代码对数量({len(all_code_pairs)})不匹配")
                continue
            
            # 比较代码对的CGED相似度
            cged_with = cged_scores[0]
            cged_without = cged_scores[1]
            
            print(f"  有注释CGED: {cged_with:.4f}")
            print(f"  无注释CGED: {cged_without:.4f}")
            
            # 检查是否有差异（考虑浮点精度问题）
            if abs(cged_with - cged_without) > 1e-6:
                print(f"  发现差异: {abs(cged_with - cged_without):.4f}")
                differences_found = True
                
                # 准备写入文件的内容
                results_to_write.append({
                    'case_id': case_id,
                    'indices': (idx1, idx2),
                    'with_comment_code1': code_pairs_with_comments[0][0],
                    'with_comment_code2': code_pairs_with_comments[0][1],
                    'without_comment_code1': code_pairs_no_comments[0][0],
                    'without_comment_code2': code_pairs_no_comments[0][1],
                    'cged_with_comment': cged_with,
                    'cged_without_comment': cged_without,
                    'difference': abs(cged_with - cged_without)
                })
                
                # 达到最大差异数量时退出循环
                if len(results_to_write) >= max_differences:
                    print(f"  已达到最大差异数量 {max_differences}，停止检测")
                    break
            else:
                print("  无显著差异")
            
        # 如果发现差异，写入文件
        if differences_found:
            print(f"\n总共发现 {len(results_to_write)} 处CGED相似度差异，写入文件: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"CGED相似度差异分析 - 从txt文件找到 {len(cased_with_different_cged)} 个case有整体CGED差异，详细分析了 {processed_cases} 个case\n")
                f.write("=" * 80 + "\n\n")
                
                for result in results_to_write:
                    f.write(f"Case: {result['case_id']} - 使用下标 {result['indices'][0]} 和 {result['indices'][1]}\n")
                    f.write(f"有注释CGED: {result['cged_with_comment']:.4f}\n")
                    f.write(f"无注释CGED: {result['cged_without_comment']:.4f}\n")
                    f.write(f"差异: {result['difference']:.4f}\n")
                    f.write("\n有注释代码1 (下标{result['indices'][0]}):\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['with_comment_code1'] + "\n")
                    f.write("-" * 40 + "\n\n")
                    
                    f.write("有注释代码2 (下标{result['indices'][1]}):\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['with_comment_code2'] + "\n")
                    f.write("-" * 40 + "\n\n")
                    
                    f.write("无注释代码1 (下标{result['indices'][0]}):\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['without_comment_code1'] + "\n")
                    f.write("-" * 40 + "\n\n")
                    
                    f.write("无注释代码2 (下标{result['indices'][1]}):\n")
                    f.write("-" * 40 + "\n")
                    f.write(result['without_comment_code2'] + "\n")
                    f.write("-" * 40 + "\n")
                    f.write("=" * 80 + "\n\n")
        else:
            print(f"\n在分析的 {processed_cases} 个case中，未发现代码对的CGED相似度存在差异")
            
    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()