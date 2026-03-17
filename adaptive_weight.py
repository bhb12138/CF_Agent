# -*- adaptive weight scheduler -*-
from evaluator import is_responsive

def detect_use_D(response, rag_sentences):
    """
    极简版本：只要 response 中包含任意 RAG 知识片段关键词，就认为“使用了 D”
    rag_sentences 可以是 ["粮食安全", "森林破坏", ...]
    """
    return any(kw in response for kw in rag_sentences)


# def detect_responsiveness(response):
#     """
#     判断是否回应了上一位（这里简单粗暴地看是否出现“你/我/他/她”这些指代词）
#     也可以自定义对 agent 名称的判断
#     """
#     keywords = ["you", "think", "say", "agree"]  # 自己可扩展
#     return any(k in response for k in keywords)


def scheduler(round_num, last_response, responsive, rag_sentences,
              wT=1.0, wM=1.0, wD=1.5, alpha=0.1):
    """
    两阶段更新：1）time-based 趋势 2）behavior-based adaptive
    返回更新后的 wT, wM, wD
    """
    wT, wM, wD = wT, wM, wD

    max_w = 3.0
    min_w = 0.1

    # ----- (1) trend-based -----
    if round_num == 0:
        wT, wM, wD = wT, wM, wD  # 初始更依赖知识
    else:
        wT = 1.0
        wM = min(1 + 0.1 * round_num, max_w)
        wD = max(1.5 - 0.1 * round_num, min_w)

    # ----- (2) behavior-based -----
    if round_num > 0:
        # 未使用知识 -> 强化 wD
        if not detect_use_D(last_response, rag_sentences):
            wD = min(wD + alpha, max_w)
        else:
            wD = max(wD - alpha, min_w)
        # 未回应上轮 -> 强化 wM
        # if not detect_responsiveness(last_response):
        if not responsive:
            wM = min(wM + alpha, max_w)
        else:
            wM = max(wM - alpha, min_w)

    return wT, wM, wD
