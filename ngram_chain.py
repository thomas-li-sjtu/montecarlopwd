import bisect  # 针对有序数组的插入和排序操作的一个模块(二分查找算法模块，可以在已排序的列表等序列容器中查找与插入值)
import bz2
import collections
import heapq
import itertools
import math
import operator
import pickle
import random
import shelve

import numpy as np

import model

__default = object()  # 返回一个新的无特征对象


def default_start(n):
    return '\0' * (n - 1)


def ngrams(word, n, start=__default, end='\0'):
    if start is __default:
        start = default_start(n)
    word = start + word + end
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def ngrams_counter(words, n, start=__default, end='\0', with_counts=False):
    if start is __default:
        start = default_start(n)
    if not with_counts:
        words = ((1, w) for w in words)
    res = collections.defaultdict(itertools.repeat(0).__next__)  # n元组字典，默认值为0
    for count, word in words:
        word = start + word + end  # 密码转换为：起始符号+密码+终止符号
        for i in range(len(word) - n + 1):
            res[word[i:i + n]] += count  # 建立字典，字典值为n元组出现次数
    return res


def parse_textfile(fname='/usr/share/dict/words'):
    try:
        with open(fname) as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        with bz2.open('{}.bz2'.format(fname)) as f:
            return [line.decode('latin9').strip() for line in f]


def parse_rockyou(fname='datasets/rockyou-withcount.txt.bz2'):
    res = []
    #
    with bz2.open(fname) as f:
        lines = (line.rstrip() for line in f)
        for l in lines:
            if len(l) < 8 or l[7] != 32:
                continue
            try:
                res.append((int(l[:7]), l[8:].decode('utf-8')))
            except UnicodeDecodeError:
                continue
    return res


Node = collections.namedtuple('Node', 'transitions cumprobs logprobs')


class NGramModel(model.Model):
    def setup_nodes(self, shelfname, flags='c'):
        self.shelfname = shelfname
        if shelfname is None:
            return {}
        else:
            return shelve.open(shelfname, flags,
                               protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def get_from_shelf(cls, shelfname, *args, **kwargs):
        # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，
        # 但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等
        return cls([], *args, shelfname=shelfname, **kwargs)

    def __init__(self, words, n, with_counts=False, shelfname=None):
        self.start = start = default_start(n)  # 对密码第一个字符，需要添加特殊符号，形成n元组  这里特殊符号为'\0'
        self.end = end = '\0'  # 特殊结束符号，与特殊起始符号相同
        transitions = collections.defaultdict(list)  # 转换字典，键为状态（即前n-1个字符），值为元组（频次，第n个字符）
        for ngram, count in ngrams_counter(words, n, start, end,
                                           with_counts).items():
            state, transition = ngram[:-1], ngram[-1]  # ngram 的前 n-1 位，以及 ngram 的最后一位
            transitions[state].append((count, transition))  # 转换过程计入字典

        flags = 'c' if words else 'r'
        self.nodes = nodes = self.setup_nodes(shelfname, flags)
        for state, ctlist in transitions.items():
            # itemgetter函数用于获取对象的某些维的数据，这里相当于根据元组的count，对一个state排序
            ctlist.sort(reverse=True, key=operator.itemgetter(0))
            total = sum(c for c, _ in ctlist)  # 某一个state的频次
            transitions, cumprobs, logprobs = [], [], []
            cum_counts = 0
            for count, transition in ctlist:  # 对一个state下元组(count, transition)列表的每个元素
                cum_counts += count
                transitions.append(transition)
                cumprobs.append(cum_counts / total)  # 用于之后根据概率区间采样，即[0.5,0.8,1.0] 第一个元素采样概率0.5，第二个0.3，第三个0.2
                logprobs.append(-math.log2(count / total))  # 对应概率取对数
            nodes[state] = Node(''.join(transitions),  # 一个state对应的所有可能的第n个字符
                                np.array(cumprobs),
                                np.array(logprobs))  # 所有可能的第n个字符概率对数

    def __del__(self):
        if self.shelfname is not None:
            self.nodes.close()

    def update_state(self, state, transition):
        # 当前的state根据第n个元素transition转换到下一个state：abcd + e 转换到 bcde
        return (state + transition)[1:]

    def __iter__(self, threshold=float('inf')):

        nodes = self.nodes
        startnode = nodes[self.start]
        # queue items: logprob, word, state, node, node_logprob, index
        queue = [(startnode.logprobs[0], '', self.start, startnode, 0, 0)]

        while queue:
            logprob, word, state, node, node_lp, idx = heapq.heappop(queue)
            transition = node.transitions[idx]
            if transition == self.end:
                yield logprob, word
            else:
                # push new node
                new_state = self.update_state(state, transition)
                new_node = nodes[new_state]
                new_logprob = logprob + new_node.logprobs[0]
                if new_logprob <= threshold:
                    new_item = (new_logprob, word + transition, new_state,
                                new_node, logprob, 0)
                    heapq.heappush(queue, new_item)
            try:
                next_lp = node_lp + node.logprobs[idx + 1]
            except IndexError:
                # we're done exploring this node
                continue
            if next_lp <= threshold:
                # push the next transition in the current node
                next_item = (next_lp, word, state, node, node_lp, idx + 1)
                heapq.heappush(queue, next_item)

    def generate(self, maxlen=100):
        word = []
        state = self.start
        logprob = 0
        for _ in range(maxlen):
            node = self.nodes[state]
            # 实现按概率采样
            # bisect_left(a,x)：a是一个有序列表，查看x应当插入a的位置，返回位置下标（如[0.6, 1]，0.7 则返回2）
            idx = bisect.bisect_left(node.cumprobs, random.random())
            transition = node.transitions[idx]
            logprob += node.logprobs[idx]
            if transition == self.end:
                break
            state = self.update_state(state, transition)  # 更新state，准备生成下一个字符
            word.append(transition)  # 加入字符
        return logprob, ''.join(word)

    def logprob(self, word, leaveout=False):
        if leaveout:
            raise NotImplementedError
        state = self.start
        res = 0
        for c in word + self.end:
            node = self.nodes[state]
            try:
                idx = node.transitions.index(c)
            except ValueError:
                return float('inf')
            res += node.logprobs[idx]
            state = self.update_state(state, c)
        return res

    def generate_by_threshold(self, threshold, lower_threshold=0, maxlen=100):
        # Efficient generation of passwords -- Ma et al., S&P 2014
        nodes = self.nodes
        start = self.start

        # stack items: node, word, state, logprob, index
        stack = [[nodes[start], '', start, 0, 0]]
        while stack:
            node, word, state, logprob, idx = top = stack[-1]
            try:
                newprob = logprob + node.logprobs[idx]
            except IndexError:
                stack.pop()
                continue
            if newprob > threshold:
                stack.pop()
                continue
            transition = node.transitions[idx]
            if transition == self.end:
                if newprob >= lower_threshold:
                    yield newprob, word
            elif len(stack) == maxlen:
                stack.pop()
                continue
            else:
                newstate = self.update_state(state, transition)
                stack.append([nodes[newstate], word + transition, newstate,
                              newprob, 0])
            # set the new index
            top[4] += 1


class TextGenerator(NGramModel):
    def __init__(self, phrases, n, words, with_counts=False, shelfname=None):
        super().__init__(phrases, n, with_counts, shelfname)
        self.start = start = ('',) * (n - 1)
        self.end = end = ('',)
        transitions = collections.defaultdict(list)
        for ngram, count in ngrams_counter(phrases, n, start, end,
                                           with_counts).items():
            state, transition = ngram[:-1], ngram[-1]
            transitions[state].append((count, transition))

        flags = 'c' if phrases else 'r'
        self.nodes = nodes = self.setup_nodes(shelfname, flags)
        for state, ctlist in transitions.items():
            ctlist.sort(reverse=True, key=operator.itemgetter(0))
            total = sum(c for c, _ in ctlist)
            transitions, cumprobs, logprobs = [], [], []
            cum_counts = 0
            for count, transition in ctlist:
                cum_counts += count
                transitions.append(transition)
                cumprobs.append(cum_counts / total)
                logprobs.append(-math.log2(count / total))
            transitions = [(t,) for t in transitions]
            nodes[state] = Node(transitions,
                                np.array(cumprobs),
                                np.array(logprobs))

    def generate(self, maxlen=100):
        phrase = ''
        state = self.start
        logprob = 0
        for _ in range(maxlen):
            node = self.nodes[state]
            idx = bisect.bisect_left(node.cumprobs, random.random())
            transition = node.transitions[idx]
            logprob += node.logprobs[idx]
            if transition == self.end:
                break
            state = self.update_state(state, transition)
            if phrase and transition[0][0].isalpha():
                phrase += ' '
            phrase += transition[0]
        return logprob, phrase
