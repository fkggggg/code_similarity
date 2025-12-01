import json
import os
from typing import Dict, List, Tuple
# 从CGED模块导入BatchCalculate函数
from CGED import BatchCalculate as CGED

def calculate_cged(src: str, dst: str, language: str = "python") -> float:
    """计算两个代码片段之间的CGED相似度
    
    Args:
        src: 源代码片段
        dst: 目标代码片段
        language: 代码语言，默认为python
        
    Returns:
        CGED相似度值
    """
    try:
        print(f"计算CGED: src长度={len(src)}, dst长度={len(dst)}")
        # 使用CGED模块计算相似度
        cged_score = CGED(
            code_src_dst_list=[(src, dst)],
            language_src_dst_list=[(language, language)],
            src_update_long_term_cache=True,
            dst_update_long_term_cache=False,
            pdg_parallelism = 10,
            nx_parallelism=60,
            nx_budget=30,
            verbose_level=2,   # set to 2 when debugging
        )
        
        # 获取返回的第一个相似度值
        if cged_score and isinstance(cged_score, list):
            return cged_score[0]
        else:
            print("CGED返回无效结果")
            return 0.0
            
    except Exception as e:
        print(f"计算CGED时出错: {str(e)}")
        return 0.0

def extract_bad_cases(results_file: str, output_file: str) -> int:
    """从结果文件中提取CGED为0的代码对，保存到bad_case.json
    
    Args:
        results_file: 相似度结果文件路径
        output_file: 输出的bad_case文件路径
        
    Returns:
        提取的bad case数量
    """
    try:
        # 读取结果文件
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bad_cases = []
        # 遍历所有结果
        for result in data.get('results', []):
            # 获取相似度结果
            similarity = result.get('similarity', {})
            # 检查cged是否为0
            if similarity.get('cged') == 0:
                # 提取src和dst代码内容
                bad_case = {
                    'src': result.get('src', {}).get('code', ''),
                    'dst': result.get('dst', {}).get('code', ''),
                    'language': result.get('src', {}).get('language', 'python')
                }
                bad_cases.append(bad_case)
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存bad cases到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        
        print(f"成功提取 {len(bad_cases)} 个CGED为0的bad case，保存到 {output_file}")
        return len(bad_cases)
        
    except Exception as e:
        print(f"提取bad cases时出错: {str(e)}")
        return 0

def recalculate_bad_cases(bad_case_file: str, output_file: str, default_language: str = "python") -> int:
    """从bad_case.json中批量读取代码对，重新计算CGED相似度
    
    Args:
        bad_case_file: bad_case文件路径
        output_file: 输出的修复结果文件路径
        default_language: 默认代码语言，默认为python
        
    Returns:
        处理的bad case数量
    """
    try:
        # 读取bad case文件
        with open(bad_case_file, 'r', encoding='utf-8') as f:
            bad_cases = json.load(f)
        
        # 重新计算CGED
        fixed_results = []
        for i, case in enumerate(bad_cases):
            print(f"处理bad case {i+1}/{len(bad_cases)}")
            src = case.get('src', '')
            dst = case.get('dst', '')
            # 获取语言信息，如果没有则使用默认值
            language = case.get('language', default_language)
            
            # 计算新的CGED
            try:
                new_cged = calculate_cged(src, dst, language)
                
                # 保存结果
                fixed_result = {
                    'src': src,
                    'dst': dst,
                    'language': language,
                    'original_cged': 0,  # 原始cged为0
                    'new_cged': new_cged
                }
                fixed_results.append(fixed_result)
                
                # 增量写入，避免大文件处理时内存问题
                if i % 10 == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(fixed_results, f, ensure_ascii=False, indent=2)
                        
            except Exception as e:
                print(f"计算CGED时出错: {str(e)}")
                # 出错时也保存结果，标记为错误
                fixed_result = {
                    'src': src,
                    'dst': dst,
                    'original_cged': 0,
                    'new_cged': None,
                    'error': str(e)
                }
                fixed_results.append(fixed_result)
        
        # 最终保存完整结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_results, f, ensure_ascii=False, indent=2)
        
        print(f"成功重新计算 {len(fixed_results)} 个bad case的CGED，保存到 {output_file}")
        return len(fixed_results)
        
    except Exception as e:
        print(f"处理bad cases时出错: {str(e)}")
        return 0

def main():
    # 定义文件路径
    base_url = '/data/lyy/code_similarity/similarity_results/20251128_181816'
    results_file = os.path.join(base_url, 'pos_neg_pairs_similarity_results.json')
    bad_case_file = os.path.join(base_url, 'bad_case.json')
    fixed_result_file = os.path.join(base_url, 'bad_case_fix.json')
    
    need_fix = False
    
    print("=== 开始检查CGED修复情况 ===")
    
    # 步骤1: 提取CGED为0的bad cases
    print("\n[步骤1] 提取CGED为0的代码对...")
    bad_case_count = extract_bad_cases(results_file, bad_case_file)
    
    if need_fix and bad_case_count > 0:
        # 步骤2: 重新计算这些代码对的CGED
        print("\n[步骤2] 重新计算CGED相似度...")
        recalculate_bad_cases(bad_case_file, fixed_result_file, default_language="python")
        
        # 分析修复效果
        print("\n[步骤3] 分析修复效果...")
        try:
            with open(fixed_result_file, 'r', encoding='utf-8') as f:
                fixed_results = json.load(f)
            
            if fixed_results:
                # 统计有变化的案例
                improved_cases = 0
                error_cases = 0
                cged_sum = 0
                valid_cases = 0
                
                # 按语言分组统计
                language_stats = {}
                
                for result in fixed_results:
                    language = result.get('language', 'unknown')
                    if language not in language_stats:
                        language_stats[language] = {'total': 0, 'improved': 0, 'cged_sum': 0}
                    language_stats[language]['total'] += 1
                    
                    new_cged = result.get('new_cged')
                    if result.get('error'):
                        error_cases += 1
                    elif new_cged is not None:
                        valid_cases += 1
                        cged_sum += new_cged
                        
                        if new_cged > 0:
                            improved_cases += 1
                            language_stats[language]['improved'] += 1
                            language_stats[language]['cged_sum'] += new_cged
                
                # 计算整体统计信息
                improvement_rate = (improved_cases / valid_cases) * 100 if valid_cases else 0
                avg_cged = cged_sum / valid_cases if valid_cases else 0
                
                print(f"修复效果统计:")
                print(f"- 总bad case数量: {len(fixed_results)}")
                print(f"- 有效计算数量: {valid_cases}")
                print(f"- 错误案例数量: {error_cases}")
                print(f"- 修复成功数量: {improved_cases}")
                print(f"- 修复成功率: {improvement_rate:.2f}%")
                print(f"- 平均CGED值: {avg_cged:.6f}")
                
                # 按语言输出统计
                print(f"\n按语言统计:")
                for lang, stats in language_stats.items():
                    lang_improvement_rate = (stats['improved'] / stats['total']) * 100 if stats['total'] else 0
                    lang_avg_cged = stats['cged_sum'] / stats['improved'] if stats['improved'] else 0
                    print(f"- {lang}: 总数={stats['total']}, 修复={stats['improved']}, 修复率={lang_improvement_rate:.2f}%, 平均CGED={lang_avg_cged:.6f}")
            else:
                print("未找到任何修复结果。")
                
        except Exception as e:
            print(f"分析修复效果时出错: {str(e)}")
    else:
        print("未发现CGED为0的bad case，无需修复。")
    
    print("\n=== 检查完成 ===")

if __name__ == "__main__":
    main()