import re
import string
from collections import defaultdict, Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))







def f1_score(prediction, ground_truth):
    """
    计算预测结果与真实标签之间的 F1 分数。

        F1公式：
            F1 = 2 * (precision * recall) / (precision + recall) = 

    :param prediction: 预测结果，通常是一个字符串。
    :param ground_truth: 真实标签，通常是一个字符串。
    :return: 预测结果与真实标签之间的 F1 分数。
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # 计算预测结果和真实标签中共同出现的单词及其数量
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values()) # TP
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens) # TP / TP + FP
    recall = 1.0 * num_same / len(ground_truth_tokens) # TP / TP + FN
    f1 = (2 * precision * recall) / (precision + recall) # F1 = 2 * (precision * recall) / (precision + recall)
    return f1
 


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)



def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    计算预测结果相对于所有真实标签（多标签）的最大指标得分。

    :param metric_fn: 用于计算指标得分的函数，该函数接受两个参数：预测结果和真实标签。
    :param prediction: 预测结果，通常是一个字符串或其他可比较的数据类型。
    :param ground_truths: 真实标签的列表，包含多个可能的真实结果。
    :return: 预测结果相对于所有真实标签的最大指标得分。
    """
    # 用于存储每个真实标签对应的指标得分
    scores_for_ground_truths = []
    # 遍历所有真实标签
    for ground_truth in ground_truths:
        # 调用传入的指标计算函数，计算当前预测结果与真实标签的得分
        score = metric_fn(prediction, ground_truth)
        # 将得分添加到得分列表中
        scores_for_ground_truths.append(score)
    # 返回得分列表中的最大值
    return max(scores_for_ground_truths)
