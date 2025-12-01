import json
import re
import sys
import random
import os
import datetime
from itertools import combinations
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from TSED.TSED import Calculate as TSED
from CGED import BatchCalculate as CGED
from codebleu import calc_codebleu
import concurrent.futures
from functools import partial
import multiprocessing

# 从参考文件复制的分词和相似度计算函数
def tokenize_code(code_string):
    """
    将代码字符串分词，适用于BLEU和Jaccard计算
    简单分词：按空白符分割，保留标识符和关键字
    """
    # 使用正则表达式进行更合理的代码分词
    # 分割标识符、关键字、运算符等
    tokens = re.findall(r'\b\w+\b|[{}()\[\];:,./\\*&^%$#@!~`\-+=<>|]', code_string)
    # 过滤空字符串
    tokens = [token for token in tokens if token.strip()]
    return tokens

def bleu(code1, code2):
    """
    计算两段代码的BLEU分数
    
    参数:
        code1 (str): 第一段代码
        code2 (str): 第二段代码
    
    返回:
        float: BLEU分数，范围在0到1之间
    """
    # 分词
    reference_tokens = tokenize_code(code2)
    candidate_tokens = tokenize_code(code1)
    
    # 如果候选代码为空，返回0
    if not candidate_tokens:
        return 0.0
    
    # 使用平滑函数避免0分
    smoothie = SmoothingFunction().method1
    
    try:
        # 计算BLEU分数，使用code2作为参考，code1作为候选
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
        return bleu_score
    except:
        # 处理可能的异常
        return 0.0

def jaccard(code1, code2):
    """
    计算两段代码的Jaccard相似度
    Jaccard相似度 = 交集大小 / 并集大小
    
    参数:
        code1 (str): 第一段代码
        code2 (str): 第二段代码
    
    返回:
        float: Jaccard相似度，范围在0到1之间
    """
    # 分词
    tokens1 = set(tokenize_code(code1))
    tokens2 = set(tokenize_code(code2))
    
    # 如果两个集合都为空，返回1.0（完全相似）
    if not tokens1 and not tokens2:
        return 1.0
    
    # 如果任一集合为空，返回0.0（完全不相似）
    if not tokens1 or not tokens2:
        return 0.0
    
    # 计算交集和并集
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    # 计算Jaccard相似度
    jaccard_score = len(intersection) / len(union)
    return jaccard_score

def are_pairs_duplicate(pair1, pair2):
    """
    检查两个代码对是否重复（考虑顺序不敏感，即(A,B)和(B,A)视为重复）
    
    参数:
        pair1 (tuple): 第一个代码对(code1, code2)
        pair2 (tuple): 第二个代码对(code3, code4)
    
    返回:
        bool: 如果重复返回True，否则返回False
    """
    # 将两个代码字符串排序，然后比较
    sorted_pair1 = sorted(pair1, key=lambda x: x)
    sorted_pair2 = sorted(pair2, key=lambda x: x)
    return sorted_pair1 == sorted_pair2

def select_code_pairs(code_list, max_pairs=10, seed=42):
    """
    选择代码对，当代码对数量超过max_pairs时，随机选取max_pairs个
    
    参数:
        code_list (list): 代码列表
        max_pairs (int): 最大代码对数量
        seed (int): 随机种子，用于确保结果可复现
    
    返回:
        list: 选择的代码对列表
    """
    # 生成所有可能的代码对
    all_pairs = list(combinations(code_list, 2))
    
    # 如果代码对数量不超过最大值，直接返回
    if len(all_pairs) <= max_pairs:
        return all_pairs
    
    # 否则随机选取max_pairs个
    print(f"  代码对数量 {len(all_pairs)} 超过阈值 {max_pairs}，随机选取{max_pairs}个进行计算")
    random.seed(seed)  # 设置随机种子以保证结果可复现
    selected_pairs = random.sample(all_pairs, max_pairs)
    return selected_pairs

def calculate_all_similarities(code1, code2, language='python'):
    """
    计算所有相似度方法的分数
    
    参数:
        code1 (str): 第一段代码
        code2 (str): 第二段代码
        language (str): 代码语言
    
    返回:
        dict: 包含所有相似度分数的字典
    """
    # 计算BLEU分数
    bleu_score = bleu(code1, code2)
    
    # 计算Jaccard相似度
    jaccard_score = jaccard(code1, code2)
    
    # 计算TSED分数
    try:
        tsed_score = TSED(language, code1, code2, 1.0, 0.8, 1.0)
    except Exception as e:
        print(f"TSED计算出错: {e}")
        tsed_score = 0.0
    
    # 计算CGED分数
    try:
        cged_score = CGED(
            code_src_dst_list=[(code1, code2)],
            language_src_dst_list=[(language, language)],
            src_update_long_term_cache=True,
            dst_update_long_term_cache=False,
            pdg_parallelism = 10,
            nx_parallelism=60,
            nx_budget=20,
            verbose_level=0,   # set to 2 when debugging
        )
        # 获取返回的第一个相似度值
        if cged_score and isinstance(cged_score, list):
            cged_score = cged_score[0]
    except Exception as e:
        print(f"CGED计算出错: {e}")
        cged_score = 0.0
    
    # 计算CodeBLEU分数
    try:
        codebleu_result = calc_codebleu([code2], [code1], lang=language)
        codebleu_score = codebleu_result['codebleu']
    except Exception as e:
        print(f"CodeBLEU计算出错: {e}")
        codebleu_score = 0.0
    
    return {
        'bleu': bleu_score,
        'jaccard': jaccard_score,
        'tsed': tsed_score,
        'cged': cged_score,
        'codebleu': codebleu_score
    }

def _calculate_tsed_single(args):
    """
    计算单个TSED相似度的辅助函数
    
    参数:
        args (tuple): (index, language, code1, code2)
    
    返回:
        tuple: (index, tsed_score)
    """
    index, language, code1, code2 = args
    try:
        # 重新导入模块以确保在子进程中可用
        from TSED.TSED import Calculate as TSED
        tsed_score = TSED(language, code1, code2, 1.0, 0.8, 1.0)
        return index, tsed_score
    except Exception as e:
        print(f"TSED计算出错（对{index}）: {e}")
        return index, 0.0

def _calculate_codebleu_single(args):
    """
    计算单个CodeBLEU相似度的辅助函数
    
    参数:
        args (tuple): (index, language, code1, code2)
    
    返回:
        tuple: (index, codebleu_score)
    """
    index, language, code1, code2 = args
    try:
        # 重新导入模块以确保在子进程中可用
        from codebleu import calc_codebleu
        codebleu_result = calc_codebleu([code2], [code1], lang=language)
        return index, codebleu_result['codebleu']
    except Exception as e:
        print(f"CodeBLEU计算出错（对{index}）: {e}")
        return index, 0.0

def batch_calculate_similarities(code_pairs, language='python', num_processes=40, cged_params=None):
    """
    批量计算代码对的相似度，利用CGED的批处理能力，并行化TSED和CodeBLEU计算
    使用进程池替代线程池以获得更好的性能，特别是在CPU密集型任务上
    
    参数:
        code_pairs (list): 代码对列表，每个元素是(code1, code2)元组
        language (str): 代码语言
        num_processes (int): 并行处理进程数
        cged_params (dict): CGED参数配置
    
    返回:
        list: 包含每个代码对相似度字典的列表
    """
    results = []
    
    # 计算BLEU和Jaccard相似度（这些需要逐个计算）
    for code1, code2 in code_pairs:
        bleu_score = bleu(code1, code2)
        jaccard_score = jaccard(code1, code2)
        results.append({
            'bleu': bleu_score,
            'jaccard': jaccard_score,
            'tsed': 0.0,  # 将在后面填充
            'cged': 0.0,  # 将在后面填充
            'codebleu': 0.0  # 将在后面填充
        })
    
    # 默认CGED参数
    default_cged_params = {
        'src_update_long_term_cache': True,
        'dst_update_long_term_cache': False,
        'pdg_parallelism': 10,
        'nx_parallelism': 60,
        'nx_budget': 20,
        'verbose_level': 0
    }
    
    # 如果提供了自定义参数，则合并默认参数和自定义参数
    if cged_params:
        default_cged_params.update(cged_params)
    
    # 批量计算CGED相似度
    try:
        cged_scores = CGED(
            code_src_dst_list=code_pairs,  # 直接传递所有代码对
            language_src_dst_list=[(language, language)] * len(code_pairs),
            src_update_long_term_cache=default_cged_params['src_update_long_term_cache'],
            dst_update_long_term_cache=default_cged_params['dst_update_long_term_cache'],
            pdg_parallelism=default_cged_params['pdg_parallelism'],
            nx_parallelism=default_cged_params['nx_parallelism'],
            nx_budget=default_cged_params['nx_budget'],
            verbose_level=default_cged_params['verbose_level'],
        )
        
        # 填充CGED分数
        for i, score in enumerate(cged_scores):
            if i < len(results):
                results[i]['cged'] = score
    except Exception as e:
        print(f"批量CGED计算出错: {e}")
    
    # 并行计算TSED相似度
    tsed_args = [(i, language, code1, code2) for i, (code1, code2) in enumerate(code_pairs)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        tsed_futures = [executor.submit(_calculate_tsed_single, args) for args in tsed_args]
        for future in concurrent.futures.as_completed(tsed_futures):
            index, tsed_score = future.result()
            if index < len(results):
                results[index]['tsed'] = tsed_score
    
    # 并行计算CodeBLEU相似度
    codebleu_args = [(i, language, code1, code2) for i, (code1, code2) in enumerate(code_pairs)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        codebleu_futures = [executor.submit(_calculate_codebleu_single, args) for args in codebleu_args]
        for future in concurrent.futures.as_completed(codebleu_futures):
            index, codebleu_score = future.result()
            if index < len(results):
                results[index]['codebleu'] = codebleu_score
    
    return results

def get_pos_pairs_similarity(json_file_path, num_pairs_per_case, seed=42, output_json=None, num_processes=40):
    """
    处理数据集，计算每个case中correct_submission之间的相似度均值，并支持生成JSON输出
    
    参数:
        json_file_path (str): JSON文件路径
        num_pairs_per_case (int): 每个case要抽取的样本对数量
        seed (int): 随机种子，用于确保结果可复现
        output_json (str, optional): 输出JSON文件路径
    
    返回:
        dict: 包含整体均值结果和生成的JSON数据
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取语言信息
        language = data.get('language', 'python')
        # 如果是py3则映射为python
        if language == 'py3':
            language = 'python'
        # 如果是cpp则映射为c++
        elif language == 'cpp':
            language = 'cpp'
        
        cases = data.get('cases', [])
        print(f"共找到 {len(cases)} 个case")
        
        # 初始化总相似度累加器
        total_similarities = {
            'bleu': 0.0,
            'jaccard': 0.0,
            'tsed': 0.0,
            'cged': 0.0,
            'codebleu': 0.0
        }
        total_pairs = 0
        
        # 用于JSON输出的结果列表
        json_results = []
        
        # 初始化JSON输出文件，写入头部信息
        output_file_initialized = False
        if output_json:
            try:
                # 创建初始JSON结构
                initial_json = {
                    "task": "get_pos_pairs_similarity",
                    "num_pairs_per_case": num_pairs_per_case,
                    "seed": seed,
                    "results": []
                }
                # 写入初始文件
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(initial_json, f, ensure_ascii=False, indent=2)
                output_file_initialized = True
                print(f"已初始化输出文件: {output_json}")
            except Exception as e:
                print(f"初始化JSON输出文件时出错: {e}")
        
        # 处理每个case
        for idx, case in enumerate(tqdm(cases, desc="处理所有case")):
            case_id = case.get('case', f'case_{idx}')
            correct_submissions = case.get('correct_submission', [])
            
            # 提取代码列表
            code_list = [sub.get('code', '') for sub in correct_submissions if 'code' in sub]
            code_list = [code for code in code_list if code.strip()]
            
            print(f"处理case {case_id}: 有 {len(code_list)} 个正确提交")
            
            # 用于当前case的结果列表
            current_case_results = []
            
            # 计算两两组合的相似度
            if len(code_list) >= 2:
                # 使用代码对选择函数，传入num_pairs_per_case参数
                pairs = select_code_pairs(code_list, max_pairs=num_pairs_per_case, seed=seed)
                print(f"  生成/选择 {len(pairs)} 个代码对进行计算")
                
                # 检查并去重代码对
                unique_pairs = []
                for pair in pairs:
                    # 检查是否与已有对重复
                    is_duplicate = False
                    for unique_pair in unique_pairs:
                        if are_pairs_duplicate(pair, unique_pair):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_pairs.append(pair)
                
                if len(unique_pairs) < len(pairs):
                    print(f"  发现并移除 {len(pairs) - len(unique_pairs)} 个重复代码对")
                    pairs = unique_pairs
                
                # 计算该case中所有代码对的相似度均值
                case_similarities = {
                    'bleu': 0.0,
                    'jaccard': 0.0,
                    'tsed': 0.0,
                    'cged': 0.0,
                    'codebleu': 0.0
                }
                
                # 使用批处理计算相似度
                print(f"  开始批量计算代码对相似度...")
                try:
                    # 获取CGED参数
                    cged_params = getattr(batch_calculate_similarities, 'cged_params', None)
                    all_similarities = batch_calculate_similarities(pairs, language, num_processes=num_processes, cged_params=cged_params)
                    
                    # 累加相似度并构建JSON结果
                    for i, (pair, similarities) in enumerate(zip(pairs, all_similarities)):
                        code1, code2 = pair
                        
                        # 累加相似度
                        for key in case_similarities:
                            case_similarities[key] += similarities[key]
                        
                        # 创建当前代码对的结果项，使用src和dst键名
                        json_result_item = {
                            "case": case_id,
                            "src": {
                                "code": code1,
                                "language": language
                            },
                            "dst": {
                                "code": code2,
                                "language": language
                            },
                            "similarity": similarities
                        }
                        json_results.append(json_result_item)
                        current_case_results.append(json_result_item)
                    
                    # 计算该case的均值
                    if pairs:
                        for key in case_similarities:
                            case_similarities[key] /= len(pairs)
                        
                        print(f"  case {case_id} 相似度均值:")
                        print(f"    BLEU: {case_similarities['bleu']:.4f}")
                        print(f"    Jaccard: {case_similarities['jaccard']:.4f}")
                        print(f"    TSED: {case_similarities['tsed']:.4f}")
                        print(f"    CGED: {case_similarities['cged']:.4f}")
                        print(f"    CodeBLEU: {case_similarities['codebleu']:.4f}")
                        
                        # 累加到总相似度
                        for key in total_similarities:
                            total_similarities[key] += case_similarities[key]
                        total_pairs += len(pairs)
                except Exception as e:
                    print(f"  计算case {case_id} 的相似度时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 添加错误记录到结果中
                    error_item = {
                        "case": case_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    json_results.append(error_item)
                    current_case_results.append(error_item)
                
                # 每个case完成后立即追加写入JSON
                if output_json and output_file_initialized and current_case_results:
                    try:
                        # 读取现有文件
                        with open(output_json, 'r', encoding='utf-8') as f:
                            output_data = json.load(f)
                        
                        # 追加当前case的结果
                        output_data['results'].extend(current_case_results)
                        
                        # 写回文件
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)
                        print(f"  已将case {case_id} 的结果写入输出文件")
                    except Exception as e:
                        print(f"  写入case {case_id} 的结果到输出文件时出错: {e}")
            else:
                print(f"  case {case_id}: 提交数量不足，无法计算两两相似度")
                # 记录空结果
                empty_item = {
                    "case": case_id,
                    "status": "insufficient_submissions",
                    "submission_count": len(code_list)
                }
                json_results.append(empty_item)
                
                # 写入空结果信息
                if output_json and output_file_initialized:
                    try:
                        with open(output_json, 'r', encoding='utf-8') as f:
                            output_data = json.load(f)
                        output_data['results'].append(empty_item)
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)
                        print(f"  已将case {case_id} 的状态信息写入输出文件")
                    except Exception as e:
                        print(f"  写入case {case_id} 的状态信息到输出文件时出错: {e}")
        
        # 构建完整的JSON输出结构
        json_output = {
            "task": "get_pos_pairs_similarity",
            "num_pairs_per_case": num_pairs_per_case,
            "seed": seed,
            "results": json_results,
            "total_processed_cases": len(cases),
            "total_processed_pairs": total_pairs
        }
        
        # 最后更新一次文件，确保所有信息完整
        if output_json and output_file_initialized:
            try:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                print(f"\n相似度结果已完整保存到: {output_json}")
            except Exception as e:
                print(f"保存最终JSON输出时出错: {e}")
        
        # 计算整体均值
        overall_similarities = None
        if len(cases) > 0:
            overall_similarities = {}
            for key in total_similarities:
                overall_similarities[key] = total_similarities[key] / len(cases)
            
            print("\n所有case的整体相似度均值:")
            print(f"BLEU均值: {overall_similarities['bleu']:.4f}")
            print(f"Jaccard均值: {overall_similarities['jaccard']:.4f}")
            print(f"TSED均值: {overall_similarities['tsed']:.4f}")
            print(f"CGED均值: {overall_similarities['cged']:.4f}")
            print(f"CodeBLEU均值: {overall_similarities['codebleu']:.4f}")
        else:
            print("没有找到有效的case")
        
        # 返回结果
        return {
            'overall_similarities': overall_similarities,
            'json_output': json_output,
            'total_processed_cases': len(cases),
            'total_processed_pairs': total_pairs
        }
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_pos_neg_pairs_similarity(json_file_path, num_pairs_per_case=5, seed=42, output_json=None):
    """
    处理数据集，每个case抽取指定数量的"一正一负"样本对计算相似度
    每个样本对由一个正样本（同一case内的代码）和一个负样本（同一case的incorrect_submission）组成
    如果正样本或负样本不足，可以重复采样
    
    参数:
        json_file_path (str): JSON文件路径
        num_pairs_per_case (int): 每个case要抽取的"一正一负"样本对数量
        seed (int): 随机种子，用于确保结果可复现
        output_json (str, optional): 输出JSON文件路径，若提供则会在每个case完成后写入结果
    
    返回:
        dict: 包含相似度统计结果和生成的JSON数据
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取语言信息
        language = data.get('language', 'python')
        # 如果是py3则映射为python
        if language == 'py3':
            language = 'python'
        # 如果是cpp则映射为c++
        elif language == 'cpp':
            language = 'c++'
        
        cases = data.get('cases', [])
        print(f"共找到 {len(cases)} 个case")
        
        # 初始化结果统计
        total_similarities = {
            'bleu': 0.0,
            'jaccard': 0.0,
            'tsed': 0.0,
            'cged': 0.0,
            'codebleu': 0.0
        }
        total_processed_cases = 0
        total_processed_pairs = 0
        
        # 用于JSON输出的结果列表
        json_results = []
        
        # 初始化JSON输出文件，写入头部信息
        output_file_initialized = False
        if output_json:
            try:
                # 创建初始JSON结构
                initial_json = {
                    "task": "get_pos_neg_pairs_similarity",
                    "num_pairs_per_case": num_pairs_per_case,
                    "seed": seed,
                    "results": []
                }
                # 写入初始文件
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(initial_json, f, ensure_ascii=False, indent=2)
                output_file_initialized = True
                print(f"已初始化输出文件: {output_json}")
            except Exception as e:
                print(f"初始化JSON输出文件时出错: {e}")
        
        # 收集所有case的代码提交
        all_case_codes = []
        for case in cases:
            correct_submissions = case.get('correct_submission', [])
            code_list = [sub.get('code', '') for sub in correct_submissions if 'code' in sub]
            code_list = [code for code in code_list if code.strip()]
            if code_list:
                all_case_codes.append((case.get('case', ''), code_list))
        
        # 处理每个case
        for case_idx, (case_id, code_list) in enumerate(tqdm(all_case_codes, desc="处理所有case")):
            print(f"处理case {case_id}: 有 {len(code_list)} 个正确提交")
            
            # 用于当前case的结果列表
            current_case_results = []
            
            try:
                # 正样本：从同一case中随机选择代码（可以重复）
                positive_samples = []
                random.seed(seed + case_idx)  # 为每个case设置不同但固定的种子
                
                # 生成正样本（从同一case中随机选择，可重复）
                for _ in range(num_pairs_per_case):
                    positive_samples.append(random.choice(code_list))
                
                # 从当前case的incorrect_submission中获取负样本池
                # 直接获取当前case对象，而不是从all_case_codes中获取
                current_case = cases[case_idx]
                incorrect_submissions = current_case.get('incorrect_submission', [])
                negative_pool = [sub.get('code', '') for sub in incorrect_submissions if 'code' in sub]
                negative_pool = [code for code in negative_pool if code.strip()]
                
                # 负样本：从当前case的incorrect_submission中随机选择代码（可以重复）
                negative_samples = []
                if negative_pool:  # 检查negative_pool是否为空
                    for _ in range(num_pairs_per_case):
                        negative_samples.append(random.choice(negative_pool))
                else:
                    print(f"  警告：当前case {case_id} 没有有效的负样本，跳过负样本生成")
                    
                    # 记录空结果
                    empty_item = {
                        "case": case_id,
                        "status": "insufficient_negative_samples",
                        "positive_sample_count": len(positive_samples),
                        "negative_sample_count": 0
                    }
                    json_results.append(empty_item)
                    current_case_results.append(empty_item)
                    
                    # 写入空结果信息
                    if output_json and output_file_initialized:
                        try:
                            with open(output_json, 'r', encoding='utf-8') as f:
                                output_data = json.load(f)
                            output_data['results'].append(empty_item)
                            with open(output_json, 'w', encoding='utf-8') as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                            print(f"  已将case {case_id} 的状态信息写入输出文件")
                        except Exception as e:
                            print(f"  写入case {case_id} 的状态信息到输出文件时出错: {e}")
                    
                    continue
                
                print(f"  生成 {len(positive_samples)} 个正样本和 {len(negative_samples)} 个负样本")
                
                # 创建"一正一负"的样本对
                pos_neg_pairs = []
                for i in range(min(len(positive_samples), len(negative_samples))):
                    pos_neg_pairs.append((positive_samples[i], negative_samples[i]))
                
                print(f"  生成 {len(pos_neg_pairs)} 个'一正一负'样本对")
                
                # 如果没有有效的样本对，跳过此case
                if not pos_neg_pairs:
                    print(f"  警告：没有找到有效的'一正一负'样本对，跳过此case")
                    
                    # 记录空结果
                    empty_item = {
                        "case": case_id,
                        "status": "no_valid_pairs",
                        "positive_sample_count": len(positive_samples),
                        "negative_sample_count": len(negative_samples)
                    }
                    json_results.append(empty_item)
                    current_case_results.append(empty_item)
                    
                    # 写入空结果信息
                    if output_json and output_file_initialized:
                        try:
                            with open(output_json, 'r', encoding='utf-8') as f:
                                output_data = json.load(f)
                            output_data['results'].append(empty_item)
                            with open(output_json, 'w', encoding='utf-8') as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                            print(f"  已将case {case_id} 的状态信息写入输出文件")
                        except Exception as e:
                            print(f"  写入case {case_id} 的状态信息到输出文件时出错: {e}")
                    
                    continue
                
                # 计算样本对相似度
                # 获取CGED参数
                cged_params = getattr(batch_calculate_similarities, 'cged_params', None)
                similarity_results = batch_calculate_similarities(pos_neg_pairs, language, cged_params=cged_params)
                
                # 构建当前case的JSON结果
                for i, ((positive_code, negative_code), similarities) in enumerate(zip(pos_neg_pairs, similarity_results)):
                    json_result_item = {
                        "case": case_id,
                        "src": {
                            "code": positive_code,  # src是正样本（正确提交）
                            "language": language
                        },
                        "dst": {
                            "code": negative_code,  # dst是负样本（错误提交）
                            "language": language
                        },
                        "similarity": similarities,
                        "sample_type": "positive-negative",
                        "pair_index": i
                    }
                    json_results.append(json_result_item)
                    current_case_results.append(json_result_item)
                
                # 计算该case的平均相似度
                case_avg_similarity = {
                    'bleu': 0.0,
                    'jaccard': 0.0,
                    'tsed': 0.0,
                    'cged': 0.0,
                    'codebleu': 0.0
                }
                
                for result in similarity_results:
                    for key in case_avg_similarity:
                        case_avg_similarity[key] += result[key]
                
                for key in case_avg_similarity:
                    case_avg_similarity[key] /= len(similarity_results)
                
                print(f"  case {case_id} '一正一负'样本对相似度均值:")
                print(f"    BLEU: {case_avg_similarity['bleu']:.4f}")
                print(f"    Jaccard: {case_avg_similarity['jaccard']:.4f}")
                print(f"    TSED: {case_avg_similarity['tsed']:.4f}")
                print(f"    CGED: {case_avg_similarity['cged']:.4f}")
                print(f"    CodeBLEU: {case_avg_similarity['codebleu']:.4f}")
                
                # 累加到总统计
                for key in total_similarities:
                    total_similarities[key] += case_avg_similarity[key]
                total_processed_cases += 1
                total_processed_pairs += len(pos_neg_pairs)
                
                # 每个case完成后立即追加写入JSON
                if output_json and output_file_initialized and current_case_results:
                    try:
                        # 读取现有文件
                        with open(output_json, 'r', encoding='utf-8') as f:
                            output_data = json.load(f)
                        
                        # 追加当前case的结果
                        output_data['results'].extend(current_case_results)
                        
                        # 写回文件
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)
                        print(f"  已将case {case_id} 的结果写入输出文件")
                    except Exception as e:
                        print(f"  写入case {case_id} 的结果到输出文件时出错: {e}")
            except Exception as e:
                print(f"  处理case {case_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                # 添加错误记录到结果中
                error_item = {
                    "case": case_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                json_results.append(error_item)
                current_case_results.append(error_item)
                
                # 写入错误信息
                if output_json and output_file_initialized and current_case_results:
                    try:
                        # 读取现有文件
                        with open(output_json, 'r', encoding='utf-8') as f:
                            output_data = json.load(f)
                        
                        # 追加错误信息
                        output_data['results'].extend(current_case_results)
                        
                        # 写回文件
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)
                        print(f"  已将case {case_id} 的错误信息写入输出文件")
                    except Exception as e:
                        print(f"  写入case {case_id} 的错误信息到输出文件时出错: {e}")
        
        # 构建完整的JSON输出结构
        json_output = {
            "task": "get_pos_neg_pairs_similarity",
            "num_pairs_per_case": num_pairs_per_case,
            "seed": seed,
            "results": json_results,
            "total_processed_cases": total_processed_cases,
            "total_cases": len(all_case_codes),
            "total_processed_pairs": total_processed_pairs
        }
        
        # 最后更新一次文件，确保所有信息完整
        if output_json and output_file_initialized:
            try:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                print(f"\n相似度结果已完整保存到: {output_json}")
            except Exception as e:
                print(f"保存最终JSON输出时出错: {e}")
        
        # 计算整体均值
        overall_results = None
        if total_processed_cases > 0:
            overall_avg = {}
            for key in total_similarities:
                overall_avg[key] = total_similarities[key] / total_processed_cases
            
            print(f"\n所有case的整体相似度均值 ({total_processed_cases}/{len(all_case_codes)} 个case处理成功):")
            print(f"BLEU均值: {overall_avg['bleu']:.4f}")
            print(f"Jaccard均值: {overall_avg['jaccard']:.4f}")
            print(f"TSED均值: {overall_avg['tsed']:.4f}")
            print(f"CGED均值: {overall_avg['cged']:.4f}")
            print(f"CodeBLEU均值: {overall_avg['codebleu']:.4f}")
            print(f"总共处理 {total_processed_pairs} 个'一正一负'样本对")
            
            overall_results = {
                'overall': overall_avg,
                'processed_cases': total_processed_cases,
                'total_cases': len(all_case_codes),
                'processed_pairs': total_processed_pairs
            }
        
        # 返回结果
        return {
            'overall_results': overall_results,
            'json_output': json_output,
            'total_processed_cases': total_processed_cases,
            'total_processed_pairs': total_processed_pairs
        }
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    random_seed = 42
    random.seed(random_seed)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/data/lyy/code_similarity/similarity_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    json_file_path = "/data/lyy/code_similarity/datasets_old/dataset_test_py3_subset.json"
    print(f"开始处理文件: {json_file_path}")
    # 每个case要抽取的样本对数量
    num_pairs_per_case = 10
    # 最大进程数，默认使用80%的CPU核心数
    num_processes = int(max(multiprocessing.cpu_count() * 0.8, 40))
    # CGED参数
    cged_params = {
        'src_update_long_term_cache': True,
        'dst_update_long_term_cache': False,
        'pdg_parallelism': 10,
        'nx_parallelism': 60,
        'nx_budget': 20,
        'verbose_level': 0
    }
    
    batch_calculate_similarities.cged_params = cged_params
    
    # 指定输出JSON文件路径
    output_json_path = f"{output_dir}/pos_pairs_similarity_results.json"
    
    # 运行正样本对相似度计算
    results = get_pos_pairs_similarity(json_file_path, num_pairs_per_case, random_seed, output_json=output_json_path, num_processes=num_processes)

    output_json_path2 = f"{output_dir}/pos_neg_pairs_similarity_results.json"
    results = get_pos_neg_pairs_similarity(json_file_path, num_pairs_per_case, random_seed, output_json=output_json_path2, num_processes=num_processes)
    
    if results:
        print("\n相似度计算完成！")
        # 处理不同函数的返回值结构
        if 'total_processed_pairs' in results:
            print(f"总共处理 {results['total_processed_pairs']} 个代码对")
            if 'overall_results' in results and results['overall_results'] and 'overall' in results['overall_results']:
                overall = results['overall_results']['overall']
                print("\n整体相似度统计:")
                print(f"BLEU均值: {overall.get('bleu', 0):.4f}")
                print(f"Jaccard均值: {overall.get('jaccard', 0):.4f}")
                print(f"TSED均值: {overall.get('tsed', 0):.4f}")
                print(f"CGED均值: {overall.get('cged', 0):.4f}")
                print(f"CodeBLEU均值: {overall.get('codebleu', 0):.4f}")
        else:
            print(f"总共处理 {results.get('processed_pairs', 0)} 个代码对")
            if 'overall' in results:
                overall = results['overall']
                print("\n整体相似度统计:")
                print(f"BLEU均值: {overall.get('bleu', 0):.4f}")
                print(f"Jaccard均值: {overall.get('jaccard', 0):.4f}")
                print(f"TSED均值: {overall.get('tsed', 0):.4f}")
                print(f"CGED均值: {overall.get('cged', 0):.4f}")
                print(f"CodeBLEU均值: {overall.get('codebleu', 0):.4f}")
    else:
        print("\n相似度计算失败！")

if __name__ == "__main__":
    main()