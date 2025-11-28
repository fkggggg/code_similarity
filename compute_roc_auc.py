#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROC-AUC计算和绘图工具
该脚本用于分析两个JSON文件中的相似度结果，计算各个指标的ROC-AUC分数，并绘制ROC曲线。
"""

import os
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算并绘制ROC-AUC曲线')
    parser.add_argument('--pos_file', 
                        type=str, 
                        default='/data/lyy/code_similarity/similarity_results/20251128_132006/pos_pairs_similarity_results.json',
                        help='正样本对结果JSON文件路径')
    parser.add_argument('--pos_neg_file', 
                        type=str, 
                        default='/data/lyy/code_similarity/similarity_results/20251128_132006/pos_neg_pairs_similarity_results.json',
                        help='负样本对结果JSON文件路径')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='/data/lyy/code_similarity/similarity_results/roc_results',
                        help='结果输出目录')
    parser.add_argument('--plot_file', 
                        type=str, 
                        default='roc_curves.png',
                        help='ROC曲线保存文件名')
    parser.add_argument('--score_file', 
                        type=str, 
                        default='auc_scores.json',
                        help='AUC分数保存文件名')
    return parser.parse_args()

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {e}")
        return None

def extract_metrics_from_results(results, label):
    """从结果中提取指标值并添加标签"""
    metrics_data = {
        'bleu': [],
        'jaccard': [],
        'tsed': [],
        'cged': [],
        'codebleu': [],
        'labels': []
    }
    
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            # 优先从 similarity 字段读取
            if 'similarity' in item and isinstance(item['similarity'], dict):
                sim = item['similarity']
                metrics_data['bleu'].append(float(sim.get('bleu', 0)))
                metrics_data['jaccard'].append(float(sim.get('jaccard', 0)))
                metrics_data['tsed'].append(float(sim.get('tsed', 0)))
                metrics_data['cged'].append(float(sim.get('cged', 0)))
                metrics_data['codebleu'].append(float(sim.get('codebleu', 0)))
            # 兼容 metrics 字段
            elif 'metrics' in item and isinstance(item['metrics'], dict):
                met = item['metrics']
                metrics_data['bleu'].append(float(met.get('bleu', 0)))
                metrics_data['jaccard'].append(float(met.get('jaccard', 0)))
                metrics_data['tsed'].append(float(met.get('tsed', 0)))
                metrics_data['cged'].append(float(met.get('cged', 0)))
                metrics_data['codebleu'].append(float(met.get('codebleu', 0)))
            # 直接顶层字段
            else:
                metrics_data['bleu'].append(float(item.get('bleu', 0)))
                metrics_data['jaccard'].append(float(item.get('jaccard', 0)))
                metrics_data['tsed'].append(float(item.get('tsed', 0)))
                metrics_data['cged'].append(float(item.get('cged', 0)))
                metrics_data['codebleu'].append(float(item.get('codebleu', 0)))
            metrics_data['labels'].append(label)
    
    return metrics_data

def prepare_data_for_roc(pos_results, neg_results):
    """准备ROC曲线所需的数据：合并正负样本"""
    pos_data = extract_metrics_from_results(pos_results, 1)
    neg_data = extract_metrics_from_results(neg_results, 0)
    
    print(f"从正样本结果中提取了 {len(pos_data['labels'])} 个正样本")
    print(f"从负样本结果中提取了 {len(neg_data['labels'])} 个负样本")
    
    combined_data = {
        'bleu': pos_data['bleu'] + neg_data['bleu'],
        'jaccard': pos_data['jaccard'] + neg_data['jaccard'],
        'tsed': pos_data['tsed'] + neg_data['tsed'],
        'cged': pos_data['cged'] + neg_data['cged'],
        'codebleu': pos_data['codebleu'] + neg_data['codebleu'],
        'labels': pos_data['labels'] + neg_data['labels']
    }
    
    return combined_data

def validate_roc_data(data):
    """验证ROC数据的有效性"""
    if not data or not data.get('labels'):
        return False, "标签数据为空"
    
    labels = data['labels']
    n = len(labels)
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return False, f"标签只有一种类别: {unique_labels}"
    
    for metric in ['bleu', 'jaccard', 'tsed', 'cged', 'codebleu']:
        if len(data[metric]) != n:
            return False, f"{metric} 长度与标签不一致"
    
    return True, "数据有效"

def compute_auc_scores(data):
    """计算各个指标的AUC分数（直接使用原始相似度得分）"""
    auc_scores = {}
    y_true = np.array(data['labels'])
    
    for metric in ['bleu', 'jaccard', 'tsed', 'cged', 'codebleu']:
        try:
            y_score = np.array(data[metric], dtype=np.float64)
            if len(y_score) == 0:
                raise ValueError("空数组")
            auc = roc_auc_score(y_true, y_score)
            auc_scores[metric] = float(auc)
            print(f"{metric.upper()}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"警告: 无法计算 {metric} 的 AUC: {e}")
            auc_scores[metric] = None
    
    return auc_scores

def plot_roc_curves(data, auc_scores, output_path):
    """绘制综合ROC曲线"""
    plt.figure(figsize=(12, 10), dpi=150)
    colors = {'bleu': '#1f77b4', 'jaccard': '#2ca02c', 'tsed': '#d62728', 'cged': '#9467bd', 'codebleu': '#ff7f0e'}
    linestyles = {'bleu': '-', 'jaccard': '--', 'tsed': '-.', 'cged': ':', 'codebleu': '-.'}
    display_names = {'bleu': 'BLEU', 'jaccard': 'Jaccard', 'tsed': 'TSED', 'cged': 'CGED', 'codebleu': 'CodeBLEU'}
    
    y_true = np.array(data['labels'])
    
    # 随机分类器基线
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random (AUC = 0.5)')
    
    for metric in ['bleu', 'jaccard', 'tsed', 'cged', 'codebleu']:
        if auc_scores[metric] is not None:
            y_score = np.array(data[metric])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, color=colors[metric], lw=2.5,
                     linestyle=linestyles[metric],
                     label=f'{display_names[metric]} (AUC = {auc_scores[metric]:.4f})')
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves for Code Similarity Metrics', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 样本信息
    total = len(y_true)
    pos = int(y_true.sum())
    neg = total - pos
    info = f'Total: {total}\nPositive: {pos}\nNegative: {neg}'
    plt.figtext(0.02, 0.02, info, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合ROC曲线已保存到: {output_path}")

def plot_individual_roc_curves(data, auc_scores, output_dir):
    """为每个指标绘制单独的ROC曲线"""
    display_names = {'bleu': 'BLEU', 'jaccard': 'Jaccard', 'tsed': 'TSED', 'cged': 'CGED', 'codebleu': 'CodeBLEU'}
    y_true = np.array(data['labels'])
    
    for metric in ['bleu', 'jaccard', 'tsed', 'cged', 'codebleu']:
        if auc_scores[metric] is None:
            continue
        
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random (AUC = 0.5)')
        
        y_score = np.array(data[metric])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'AUC = {auc_scores[metric]:.4f}')
        plt.fill_between(fpr, tpr, color='#1f77b4', alpha=0.2)
        
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.02])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve for {display_names[metric]}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        out_path = os.path.join(output_dir, f'roc_curve_{metric}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"单独ROC曲线已保存到: {out_path}")

def save_auc_scores(auc_scores, output_path):
    """保存AUC分数到JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(auc_scores, f, ensure_ascii=False, indent=2)
        print(f"AUC分数已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 无法保存AUC分数: {e}")

def main():
    args = parse_arguments()
    
    # 创建时间戳文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir
    timestamp_output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(timestamp_output_dir, exist_ok=True)
    print(f"创建时间戳文件夹: {timestamp_output_dir}")
    
    args.output_dir = timestamp_output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    pos_raw = load_json_file(args.pos_file)
    neg_raw = load_json_file(args.pos_neg_file)
    if pos_raw is None or neg_raw is None:
        return
    
    # 提取 results 字段（如果存在）
    pos_results = pos_raw.get('results', pos_raw) if isinstance(pos_raw, dict) else pos_raw
    neg_results = neg_raw.get('results', neg_raw) if isinstance(neg_raw, dict) else neg_raw
    
    # 准备数据
    roc_data = prepare_data_for_roc(pos_results, neg_results)
    is_valid, msg = validate_roc_data(roc_data)
    if not is_valid:
        print(f"数据验证失败: {msg}")
        return
    
    # 统计信息
    y_true = roc_data['labels']
    print(f"\n数据统计: 总样本={len(y_true)}, 正样本={sum(y_true)}, 负样本={len(y_true)-sum(y_true)}")
    
    # 计算 AUC
    print("\n[1/3] 计算 AUC 分数...")
    auc_scores = compute_auc_scores(roc_data)
    
    # 打印结果
    print("\nAUC 结果:")
    print("-" * 40)
    best_metric, best_auc = None, -1
    for m, s in auc_scores.items():
        if s is not None:
            print(f"{m.upper():<10} {s:.4f}")
            if s > best_auc:
                best_auc, best_metric = s, m
        else:
            print(f"{m.upper():<10} N/A")
    if best_metric:
        print(f"\n最佳指标: {best_metric.upper()} (AUC = {best_auc:.4f})")
    
    # 可视化
    print("\n[2/3] 绘制综合ROC曲线...")
    plot_roc_curves(roc_data, auc_scores, os.path.join(args.output_dir, args.plot_file))
    
    print("[3/3] 绘制单独ROC曲线...")
    plot_individual_roc_curves(roc_data, auc_scores, args.output_dir)
    
    # 保存分数
    save_auc_scores(auc_scores, os.path.join(args.output_dir, args.score_file))
    
    print(f"\n✅ 所有结果已保存至: {timestamp_output_dir}")

if __name__ == '__main__':
    main()