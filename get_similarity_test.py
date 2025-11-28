import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from TSED.TSED import Calculate as TSED
from CGED import BatchCalculate as CGED
from codebleu import calc_codebleu

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


def codebleu_similarity(code1, code2, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None):
    """
    计算两段代码的CodeBLEU相似度
    
    参数:
        code1 (str): 预测代码
        code2 (str): 参考代码
        lang (str): 代码语言，默认为"python"
        weights (tuple[float,float,float,float]): 分别为ngram_match、weighted_ngram_match、syntax_match和dataflow_match的权重
        tokenizer (callable): 将代码字符串拆分为标记，默认为None
    
    返回:
        dict[str, float]: 包含codebleu最终得分及各组成部分得分的字典
    """
    result = calc_codebleu([code2], [code1], lang=lang, weights=weights, tokenizer=tokenizer)
    return result


# 测试示例（如果直接运行此文件）
if __name__ == "__main__":
    # 测试代码示例
    code_example1 = """
    def add(a, b):
        return a + b

    result = add(5, 3)
    print(result)
    """
    
    code_example2 = """
    def add(x, y):
        return x + y


    outcome = add(5, 3)
    print(outcome)
    """
    
    # code_example1 = """
    # n, m = map(int, input().split())\nxyz = [tuple(map(int, input().split())) for _ in range(n)]\nans = 0\nfrom itertools import product\nfor p in product([-1, 1], repeat=3):\n    xyz_ = map(lambda l: l[0]*p[0] + l[1]*p[1] + l[2]*p[2], xyz)\n    xyz_ = sorted(xyz_, reverse=True)\n    ans = max(ans, sum(xyz_[:m]))\nprint(ans)\n
    # """

    # code_example2 = """
    # from itertools import product\nN, M = map(int, input().split())\ncake = [list(map(int, input().split())) for i in range(N)]\n\nans = 0\nfor x, y, z in product((1, -1), repeat=3):\n    S = []\n    for a, b, c in cake:\n    S.append(a*x + b*y + c*z)\n    S.sort(reverse=True)\n    ans = max(ans, sum(S[:M]))\n\nprint(ans)\n
    # """

    # 计算相似度
    bleu_score = bleu(code_example1, code_example2)
    jaccard_score = jaccard(code_example1, code_example2)
    tsed_score = TSED("python", code_example1, code_example2, 1.0, 0.8, 1.0)

    cged_score = CGED(
        code_src_dst_list=[(code_example1, code_example2)],
        language_src_dst_list=[("python", "python")],
        src_update_long_term_cache=True,
        dst_update_long_term_cache=False,
        pdg_parallelism = 10,
        nx_parallelism=60,
        nx_budget=20,
        verbose_level=2,   # set to 2 when debugging
    )
    # 获取返回的第一个相似度值（因为只有一对代码进行比较）
    if cged_score and isinstance(cged_score, list):
        cged_score = cged_score[0]
    
    # 计算CodeBLEU相似度
    codebleu_result = codebleu_similarity(code_example1, code_example2, lang="python")
    
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Jaccard Similarity: {jaccard_score:.4f}")
    print(f"TSED Score: {tsed_score:.4f}")
    print(f"CGED Score: {cged_score:.4f}")
    print(f"CodeBLEU Score: {codebleu_result['codebleu']:.4f}")
    print(f"  - ngram_match_score: {codebleu_result['ngram_match_score']:.4f}")
    print(f"  - weighted_ngram_match_score: {codebleu_result['weighted_ngram_match_score']:.4f}")
    print(f"  - syntax_match_score: {codebleu_result['syntax_match_score']:.4f}")
    print(f"  - dataflow_match_score: {codebleu_result['dataflow_match_score']:.4f}")
