# standard library
import bisect
import collections
import itertools
import math
import operator
import os
import random
import shelve
import numpy

# internal imports
import model
import ngram_chain

TmpNode = collections.namedtuple('TmpNode',
                                 'transitions probabilities '
                                 'cumprobs logprobs')


class BackoffModel(ngram_chain.NGramModel):

    def __init__(self, words, threshold, start_symbol=True, with_counts=False, shelfname=None):
        # super().__init__(words, n, with_counts, shelfname)
        if not with_counts:
            words = [(1, w) for w in words]

        self.start = start = '\0' if start_symbol else ''
        self.end = end = '\0'
        lendelta = len(start) + len(end)
        words = [(len(w) + lendelta, (c, start + w + end))
                 for c, w in words]

        self.nodes = nodes = self.setup_nodes(shelfname)

        words.sort(key=operator.itemgetter(0))
        if not words:
            return

        lens, words = zip(*words)
        lens = numpy.array(lens)  # 每个密码的长度（计入开始和结束符号）
        nwords = len(words)

        def zerodict():
            return collections.defaultdict(itertools.repeat(0).__next__)

        charcounts = zerodict()  # 字符集字典，键为字符，值为出现次数
        for count, word in words:
            for c in word[start_symbol:]:
                charcounts[c] += count

        totchars = sum(charcounts.values())
        transitions, counts = zip(*sorted(charcounts.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True))  # 根据出现次数排序
        transitions = ''.join(transitions)  # 字符集组成的字符串，前面的字符在训练集中出现次数更多
        counts = numpy.array(counts)  # 总密码数
        totchars = counts.sum()

        probabilities = counts / totchars

        if transitions:
            nodes[''] = TmpNode(transitions,
                                probabilities,
                                probabilities.cumsum(),
                                -numpy.log2(probabilities))  # 第一个节点，也是空字符串对应的元组（下一个字符，概率，累计概率，对数概率）
        print(nodes)
        leftidx = 0
        skipwords = set()

        for n in range(2, lens[-threshold] + 1):  # 从到 2 到 len(第threshold长的密码)，即 2 到对应长的 ngram
            print(lens)
            leftidx = bisect.bisect_left(lens, n)  # 不同长度对应的起始下标

            ngram_counter = zerodict()
            for i in range(leftidx, nwords):
                if i in skipwords:  # 如果 i 是要跳过的密码下标
                    continue
                count, word = words[i]
                skip = True
                for j in range(lens[i] - n + 1):  # 对于第 i 个密码
                    ngram = word[j: j + n]
                    if ngram[:-2] in nodes:
                        ngram_counter[ngram] += count
                        skip = False
                if skip:
                    skipwords.add(i)

            tmp_dict = collections.defaultdict(list)
            for ngram, count in ngram_counter.items():  # 对于 ngram，前 n-1 为state，n 为 transition
                tmp_dict[ngram[:-1]].append((ngram[-1], count))  # state:(transition, count)

            for state, sscounts in tmp_dict.items():
                total = sum(count for _, count in sscounts)
                if total < threshold:  # 存储计数超过阈值的所有子字符串
                    continue
                trans_probs = {c: count / total
                               for c, count in sscounts
                               if count >= threshold}  # 某一个状态的转移概率计算
                missing = 1 - sum(trans_probs.values())  # 没有找到转移结果的概率
                if missing == 1:
                    continue

                if missing > 0:
                    parent_state = self.nodes[state[1:]]
                    for c, p in zip(parent_state.transitions,
                                    parent_state.probabilities):
                        trans_probs[c] = trans_probs.get(c, 0) + p * missing  # 归一化

                trans_probs = sorted(trans_probs.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)  # 转移概率字典排序
                transitions, probabilities = zip(*trans_probs)
                transitions = ''.join(transitions)  # 可能的转移字符串
                probabilities = numpy.array(probabilities)
                # probabilities must sum to 1
#                assert abs(probabilities.sum() - 1) < 0.001

                nodes[state] = TmpNode(transitions, probabilities,
                                       probabilities.cumsum(),
                                       -numpy.log2(probabilities))

        Node = ngram_chain.Node  # collections.namedtuple('Node', 'transitions cumprobs logprobs')
        for state, node in self.nodes.items():
            nodes[state] = Node(node.transitions, node.cumprobs,
                                node.logprobs)  # nodes字典的值列表中丢弃转移概率数组，

    def update_state(self, state, transition):
        nodes = self.nodes
        state += transition
        while state not in nodes:
            state = state[1:]
        return state


class LazyBackoff(model.Model):

    def __init__(self, path, threshold, start=True, end=True):
        self.threshold = threshold
        self.start = '\0' if start else ''
        self.end = '\0' if end else ''
        self.shelves = {
            int(fname): shelve.open(os.path.join(path, fname), 'r')
            for fname in os.listdir(path)
        }

    def hasnode(self, state):
        shelves = self.shelves
        return len(state) in shelves and state in shelves[len(state)]

    def getnode(self, state):
        return self.shelves[len(state)][state]

    def getcount(self, ngram):
        if ngram != '':
            return self.shelves[len(ngram) - 1][ngram[:-1]][ngram[-1]]
        else:
            return sum(self.shelves[0][''].values())

    def begin(self):
        start = self.start
        return start, self.getnode(start)

    def update_state(self, state, transition):
        state += transition
        while not self.hasnode(state):
            state = state[1:]
        while self.getcount(state) < self.threshold:
            state = state[1:]
        node = self.getnode(state)
        return state, node

    def backoff(self, state):
        state = state[1:]
        node = self.getnode(state)
        return state, node

    def logprob(self, word, leaveout=False):

        res = 0
        state, node = self.begin()
        for c in word + self.end:
            while True:  # break when we should stop backing off
                count = node.get(c, 0) - leaveout
                total = self.getcount(state) - leaveout
                if state == self.start == self.end:
                    total /= 2
                if count >= self.threshold or state == '':
                    break
                passing = sum(ct for ct in node.values()
                              if ct >= self.threshold)
                if leaveout and count == self.threshold - 1:
                    passing -= self.threshold
                try:
                    res -= math.log2(1 - (passing / total))
                except ValueError:
                    return float('inf')
                state, node = self.backoff(state)
            try:
                res -= math.log2(count / total)
            except ValueError:
                return float('inf')
            state, node = self.update_state(state, c)
        return res

    def generate(self, maxlen=100):

        logprob = 0
        state, node = self.begin()
        word = ''
        while True:  # return when we find self.end
            while True:  # break when we should stop backing off
                values = list(node.values())
                cumsum = numpy.cumsum(values)
                total = self.getcount(state)
                if state == self.start == self.end:
                    total /= 2
                idx = bisect.bisect_right(cumsum, random.randrange(total))
                if idx < len(values):
                    count = values[idx]
                    if state == '' or count >= self.threshold:
                        break
                state, node = self.backoff(state)
                passing = sum(ct for ct in values if ct >= self.threshold)
                logprob -= math.log2(1 - (passing / total))
            logprob -= math.log2(count / total)
            c = next(itertools.islice(node.keys(), idx, None))  # n-th elem
            if c == self.end:
                return logprob, word
            word += c
            if len(word) >= maxlen:
                return logprob, word
            state, node = self.update_state(state, c)
