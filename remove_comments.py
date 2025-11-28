#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去除JSON文件中代码的注释行
用于分析注释对代码相似度计算的影响
"""

import json
import re
import os
from tqdm import tqdm
from datetime import datetime

def remove_comments_from_code(code, language='python', log_file=None):
    """
    从代码中移除注释行
    
    参数:
        code (str): 原始代码字符串
        language (str): 代码语言，支持 'python', 'cpp', 'java' 等
        log_file (file object): 日志文件对象，用于记录注释行检测和处理过程
    
    返回:
        str: 去除注释后的代码
    """
    log_entries = []
    
    # 支持 'python' 和 'py3' 作为Python语言标识
    if language in ['python', 'py3']:
        # Python 注释处理：
        # 1. 去除块注释（''' 和 """）
        # 2. 去除行注释（# 开头）
        # 3. 保留字符串中的 #
        
        # 首先处理三引号块注释
        # 改进的状态机处理逻辑，确保正确处理所有三引号注释情况
        in_string = False
        in_triple_quote = False
        in_triple_string = False
        triple_quote_char = None
        string_char = None
        processed_code = []
        i = 0
        
        while i < len(code):
            # 检查是否在字符串或三引号注释中
            if not in_string and not in_triple_quote and not in_triple_string:
                # 检查是否开始三引号注释或字符串
                # 分别检查单引号和双引号三引号注释
                if i + 2 < len(code) and (code[i:i+3] == "'''" or code[i:i+3] == '"""'):
                    # 判断是否是注释（在缩进位置或行首）
                    # 检查这一行的前面是否只有空白字符或在行首
                    line_start = code.rfind('\n', 0, i) + 1
                    prefix = code[line_start:i]
                    
                    # 如果前缀为空或只有空白字符，认为是注释
                    is_comment = False
                    if prefix.lstrip() == '':
                        # 在行首或只有空白字符，很可能是注释
                        # 特别处理数据集中出现的情况：行首的三引号通常是文档字符串注释
                        if i + 3 < len(code):
                            # 获取三引号后的一部分内容来判断
                            after_quotes = code[i+3:i+50]  # 查看后面的50个字符
                            
                            # 特殊处理：如果紧跟着的是另一个引号，这可能是四重引号字符串
                            # 如 """"content""" 或 ''''content'''
                            if i + 3 < len(code) and code[i+3] in ['"', "'"] and code[i:i+4] != code[i]*4:
                                # 这种情况是四重引号字符串，不是注释
                                # 但要排除四个连续相同引号的情况，那可能是注释
                                is_comment = False
                            # 特殊处理：四重引号字符串，如 """"0 24 34 58 62 64 69 78"""
                            elif i + 4 < len(code) and code[i:i+4] == '"' * 4:
                                # 四个引号开头的字符串，这是字符串字面量，不是注释
                                is_comment = False
                            # 如果后面紧跟换行符，很可能是注释
                            elif code[i+3] == '\n':
                                is_comment = True
                            # 如果后面是空白字符后跟换行符，也很可能是注释
                            elif after_quotes.lstrip().startswith('\n'):
                                is_comment = True
                            # 如果后面的内容看起来像是文档字符串（包含字母）
                            elif after_quotes.strip():
                                # 检查是否看起来像文档字符串
                                stripped_after = after_quotes.lstrip()
                                # 如果不是在赋值语句中（如var = '''...'''），则可能是注释
                                # 检查前一行是否是赋值语句
                                prev_line_end = line_start - 1
                                if prev_line_end >= 0:
                                    prev_line_start = code.rfind('\n', 0, prev_line_end)
                                    prev_line = code[prev_line_start+1:prev_line_end].strip()
                                    # 如果前一行是赋值语句，则三引号是字符串字面量
                                    if '=' in prev_line and not prev_line.startswith('#'):
                                        # 进一步检查是否在同一行有闭合的三引号
                                        same_line_closure = False
                                        closing_pos = code.find(code[i:i+3], i+3)
                                        if closing_pos != -1:
                                            # 检查闭合三引号是否在同一行
                                            same_line = code.find('\n', i+3, closing_pos) == -1
                                            if same_line:
                                                same_line_closure = True
                                        
                                        # 只有在同一行有闭合三引号的情况下才认为是字符串字面量
                                        if same_line_closure:
                                            is_comment = False
                                        else:
                                            # 跨行的三引号，即使是赋值语句也可能是注释
                                            is_comment = True
                                    else:
                                        # 检查是否包含典型注释内容（字母、标点符号）
                                        if any(c.isalpha() or c in '.,!?;:' for c in stripped_after[:20]):
                                            # 进一步检查是否可能是字符串字面量
                                            # 如果三引号在同一行有闭合，则可能是字符串字面量
                                            closing_pos = code.find(code[i:i+3], i+3)
                                            if closing_pos != -1:
                                                # 检查闭合三引号是否在同一行
                                                same_line = code.find('\n', i+3, closing_pos) == -1
                                                if same_line:
                                                    is_comment = False
                                                else:
                                                    is_comment = True
                                            else:
                                                is_comment = True
                                        else:
                                            # 如果后面没有典型的注释内容，可能是字符串字面量
                                            is_comment = False
                                else:
                                    # 文件开头的三引号，很可能是注释
                                    is_comment = True
                            # 如果后面没有任何内容或者只有空白字符，可能是注释
                            elif not after_quotes.strip():
                                is_comment = True
                            # 其他情况下，默认认为是注释（因为我们已经在行首）
                            else:
                                is_comment = True
                        else:
                            # 如果三引号在文件末尾，认为是注释
                            is_comment = True
                    # 即使不在行首，但如果在赋值语句中且同一行有闭合，则不是注释
                    elif not in_string:
                        # 检查是否在同一行有闭合的三引号
                        same_line_closure = False
                        closing_pos = code.find(code[i:i+3], i+3)
                        if closing_pos != -1:
                            # 检查闭合三引号是否在同一行
                            same_line = code.find('\n', i+3, closing_pos) == -1
                            if same_line:
                                same_line_closure = True
                        
                        # 只有在同一行有闭合三引号的情况下才认为是字符串字面量
                        if same_line_closure:
                            is_comment = False
                        else:
                            # 跨行的三引号，即使不在行首也可能需要特殊处理
                            # 但我们暂时保守处理，只处理明确的行首情况
                            pass
                    
                    if is_comment:
                        in_triple_quote = True
                        triple_quote_char = code[i:i+3]
                        log_entries.append(f"检测到并移除三引号注释: {triple_quote_char}")
                    else:
                        in_triple_string = True
                        triple_quote_char = code[i:i+3]
                        processed_code.append(triple_quote_char)
                    i += 3
                    continue
                # 检查是否开始普通字符串
                elif code[i] in ['"', "'"] and (i == 0 or code[i-1] != '\\'):
                    in_string = True
                    string_char = code[i]
                    processed_code.append(code[i])
                    i += 1
                    continue
                else:
                    processed_code.append(code[i])
                    i += 1
            elif in_string:
                # 在普通字符串中，检查结束
                if code[i] == string_char and (i == 0 or code[i-1] != '\\'):
                    in_string = False
                processed_code.append(code[i])
                i += 1
            elif in_triple_string:
                # 在三引号字符串中，检查结束
                if i + 2 < len(code) and code[i:i+3] == triple_quote_char:
                    in_triple_string = False
                    processed_code.append(triple_quote_char)
                    i += 3
                else:
                    processed_code.append(code[i])
                    i += 1
            elif in_triple_quote:
                # 在三引号注释中，检查结束
                if i + 2 < len(code) and code[i:i+3] == triple_quote_char:
                    in_triple_quote = False
                    i += 3
                else:
                    i += 1  # 跳过注释内容
        
        # 将处理后的代码（已移除三引号注释）转换为字符串
        code_without_triple_quotes = ''.join(processed_code)
        
        # 然后处理行注释
        lines = code_without_triple_quotes.split('\n')
        cleaned_lines = []
        
        for line_no, line in enumerate(lines, 1):
            original_line = line.rstrip()
            # 找到第一个不在字符串中的 #
            in_string = False
            string_char = None
            comment_start = -1
            
            for i, char in enumerate(line):
                # 处理字符串
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                
                # 找到注释开始位置（不在字符串中）
                if char == '#' and not in_string:
                    comment_start = i
                    break
            
            # 移除注释部分
            if comment_start != -1:
                cleaned_line = line[:comment_start].rstrip()
                # 记录注释行信息
                comment_content = line[comment_start:].strip()
                log_entries.append(f"第{line_no}行 (行注释) | 原始: '{original_line}' | 注释: '{comment_content}' | 处理后: '{cleaned_line if cleaned_line else '(空)'}'")
                
                if cleaned_line:  # 只保留非空行
                    cleaned_lines.append(cleaned_line)
            else:
                # 没有注释的行，直接添加
                log_entries.append(f"第{line_no}行 (正常) | 原始: '{original_line}' | 处理后: '{original_line}'")
                cleaned_lines.append(line)
        
    elif language in ['cpp', 'c', 'java', 'javascript']:
        # 首先记录块注释的处理
        if '/*' in code:
            log_entries.append("检测到并移除块注释 /* ... */")
            # 移除块注释
            code = re.sub(r'(/\*.*?\*/)', '', code, flags=re.DOTALL)
        
        lines = code.split('\n')
        cleaned_lines = []
        
        for line_no, line in enumerate(lines, 1):
            original_line = line.rstrip()
            # 移除行注释，但避免匹配字符串中的 //
            in_string = False
            string_char = None
            comment_start = -1
            
            for i in range(len(line) - 1):
                if line[i] in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = line[i]
                    elif line[i] == string_char:
                        in_string = False
                
                if line[i:i+2] == '//' and not in_string:
                    comment_start = i
                    break
            
            if comment_start != -1:
                cleaned_line = line[:comment_start].rstrip()
                # 记录注释行信息
                comment_content = line[comment_start:].strip()
                log_entries.append(f"第{line_no}行 (注释) | 原始: '{original_line}' | 注释: '{comment_content}' | 处理后: '{cleaned_line if cleaned_line else '(空)'}'")
                
                if cleaned_line:  # 只保留非空行
                    cleaned_lines.append(cleaned_line)
            else:
                # 没有注释的行，直接添加
                log_entries.append(f"第{line_no}行 (正常) | 原始: '{original_line}' | 处理后: '{original_line}'")
                cleaned_lines.append(line)
    
    else:
        # 对于其他语言，暂时只做简单处理或原样返回
        warning_msg = f"警告：不支持的语言 {language}，原样返回代码"
        print(warning_msg)
        log_entries.append(warning_msg)
        log_entries.append("由于语言不支持，未进行注释处理")
        cleaned_lines = code.split('\n')
    
    # 写入日志
    if log_file:
        for entry in log_entries:
            log_file.write(entry + '\n')
        log_file.write('\n' + '-'*80 + '\n\n')
    
    return '\n'.join(cleaned_lines)

def process_dataset(json_file_path, output_file_path=None, log_file_path=None):
    """
    处理数据集JSON文件，去除所有代码中的注释，并记录处理过程
    
    参数:
        json_file_path (str): 输入JSON文件路径
        output_file_path (str): 输出JSON文件路径，如果为None则自动生成
        log_file_path (str): 日志文件路径，如果为None则自动生成
    
    返回:
        tuple: (output_file_path, log_file_path)
    """
    # 确定输出文件路径
    if output_file_path is None:
        file_dir, file_name = os.path.split(json_file_path)
        file_base, file_ext = os.path.splitext(file_name)
        output_file_path = os.path.join(file_dir, f"{file_base}_no_comments{file_ext}")
    
    # 确定日志文件路径
    if log_file_path is None:
        file_dir, file_name = os.path.split(json_file_path)
        file_base, file_ext = os.path.splitext(file_name)
        log_file_path = os.path.join(file_dir, f"{file_base}_comments_log.txt")
    
    print(f"读取数据集: {json_file_path}")
    print(f"将记录注释处理日志到: {log_file_path}")
    
    # 读取原始数据集
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取语言类型
    language = data.get('language', 'python')
    print(f"检测到语言: {language}")
    
    # 处理每个case
    cases = data.get('cases', [])
    print(f"总共有 {len(cases)} 个case需要处理")
    
    # 打开日志文件
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== 代码注释处理日志 ===\n")
        log_file.write(f"处理文件: {json_file_path}\n")
        log_file.write(f"语言类型: {language}\n")
        log_file.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总case数: {len(cases)}\n")
        log_file.write('\n' + '='*80 + '\n\n')
        
        # 使用tqdm显示进度
        comment_stats = {
            'total_cases': len(cases),
            'cases_with_comments': 0,
            'total_submissions': 0,
            'submissions_with_comments': 0
        }
        
        for case_id, case in enumerate(tqdm(cases, desc="处理case"), 1):
            case_has_comments = False
            
            log_file.write(f"处理 Case #{case_id}\n")
            
            # 处理正确提交的代码
            if 'correct_submission' in case:
                comment_stats['total_submissions'] += len(case['correct_submission'])
                
                for sub_id, sub in enumerate(case['correct_submission'], 1):
                    if 'code' in sub:
                        log_file.write(f"  处理正确提交 #{sub_id}\n")
                        
                        # 检查是否有注释
                        original_code = sub['code']
                        sub['code'] = remove_comments_from_code(original_code, language, log_file)
                        
                        # 判断是否有注释（通过比较处理前后的代码是否不同）
                        if original_code != sub['code']:
                            case_has_comments = True
                            comment_stats['submissions_with_comments'] += 1
            
            # 处理错误提交的代码
            if 'incorrect_submission' in case:
                comment_stats['total_submissions'] += len(case['incorrect_submission'])
                
                for sub_id, sub in enumerate(case['incorrect_submission'], 1):
                    if 'code' in sub:
                        log_file.write(f"  处理错误提交 #{sub_id}\n")
                        
                        # 检查是否有注释
                        original_code = sub['code']
                        sub['code'] = remove_comments_from_code(original_code, language, log_file)
                        
                        # 判断是否有注释
                        if original_code != sub['code']:
                            case_has_comments = True
                            comment_stats['submissions_with_comments'] += 1
            
            if case_has_comments:
                comment_stats['cases_with_comments'] += 1
        
        # 写入统计信息
        log_file.write('\n' + '='*80 + '\n')
        log_file.write("=== 注释处理统计信息 ===\n")
        log_file.write(f"总case数: {comment_stats['total_cases']}\n")
        log_file.write(f"含注释的case数: {comment_stats['cases_with_comments']}\n")
        log_file.write(f"总提交数: {comment_stats['total_submissions']}\n")
        log_file.write(f"含注释的提交数: {comment_stats['submissions_with_comments']}\n")
        
        if comment_stats['total_submissions'] > 0:
            percentage = (comment_stats['submissions_with_comments'] / comment_stats['total_submissions']) * 100
            log_file.write(f"含注释的提交百分比: {percentage:.2f}%\n")
    
    # 保存处理后的数据集
    print(f"保存处理后的数据集到: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！")
    print(f"输出文件: {output_file_path}")
    print(f"日志文件: {log_file_path}")
    return output_file_path, log_file_path

def main():
    # 指定输入文件路径
    input_file = "/data/lyy/code_similarity/datasets/dataset_test_py3.json"
    
    # 处理数据集
    output_file, log_file = process_dataset(input_file)
    
    print(f"原始文件: {input_file}")
    print(f"无注释文件: {output_file}")
    print(f"注释处理日志: {log_file}")

if __name__ == "__main__":
    main()