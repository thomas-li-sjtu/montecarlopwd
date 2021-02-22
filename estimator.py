import bisect
import decimal
import math
import random

import numpy as np


class PosEstimator:

    def __init__(self, sample, realsize=None):
        # realsize is a hack to make plot_restrictions work.
        # don't use unless you know what you're doing!
        # 这一部分的计算可以参见论文《Monte Carlo Strength Evaluation：Fast and Reliable Password Checking》的方法
        self.logprobs = logprobs = np.fromiter((lp for lp, _ in sample), float)  # 从可迭代对象中建立ndarray对象，返回一维数组
        logprobs.sort()  # 由于取负，则密码的实际概率越小，在此有序数组的位置越靠后
        if realsize is None:
            realsize = len(logprobs)
        logn = math.log2(realsize)
        self.positions = (2 ** (logprobs - logn)).cumsum()  # 论文的公式；返回元素的梯形累计和

    def position(self, logprob):
        idx = bisect.bisect_right(self.logprobs, logprob)
        return self.positions[idx - 1] if idx > 0 else 0

    def logpos(self, logprob):
        return math.log2(self.position(logprob))

    def logprob(self, pos):
        return np.interp(math.log2(pos + 1), np.log2(self.positions + 1),
                         self.logprobs)

    def generate(self, model_generate, entropy):
        lp_threshold = self.logprob(2 ** entropy)
        for logprob, word in iter(model_generate, None):
            if logprob <= lp_threshold < logprob - math.log2(random.random()):
                return logprob, word

    def sample(self, model_generate, entropy, n):
        for _ in range(n):
            yield self.generate(model_generate, entropy)


class IPWEstimator:

    def __init__(self, sample, store=lambda lp, word: (lp, word)):
        sample = list(sample)
        self.logn = logn = math.log2(len(sample))
        self.ipw = [2 ** decimal.Decimal(lp - logn) for lp, _ in sample]
        self.stored = [store(lp, word) for lp, word in sample]

    def evaluate(self, fun):
        return sum(w * fun(v)
                   for w, v in zip(self.ipw, self.stored))