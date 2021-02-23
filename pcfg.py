# standard library
import bisect
import collections
import itertools
import math
import random
import re
import numpy

# internal imports
import model


L, D, S = range(3)
pattern_re = re.compile(r'(?:([a-zA-Z]+)|([0-9]+)|[^a-zA-Z0-9]+)')


def patterns(w):
    # 以 ab12ab123 为例，返回 ((0, 2), (1, 2), (0, 2), (1, 3)) 和 ['ab', '12', 'cd', '123']
    structure, groups = [], []
    for match in pattern_re.finditer(w):
        L_pat, D_pat = match.groups()
        pat_t = L if L_pat else D if D_pat else S
        group = match.group()
        structure.append((pat_t, len(group)))
        groups.append(group)
    return tuple(structure), groups


def zerodict():
    return collections.defaultdict(itertools.repeat(0).__next__)


class PCFG(model.Model):

    def __init__(self, training, dictionary=None, with_counts=False):

        LDS = collections.defaultdict(zerodict)
        structures = collections.defaultdict(itertools.repeat(0).__next__)

        # 输入数据没有计数信息，是纯文本
        if not with_counts:
            training = ((1, w) for w in training)

        for count, w in training:
            structure, groups = patterns(w)
            structures[tuple(structure)] += count  # 结构字典：{((0, 2), (1, 2), (0, 2), (1, 3)): 4, ...}
            # 如果字典为空，则根据密码本身计算 L 字典，D 字典，S 字典
            for pat_pair, group in zip(structure, groups):  # {L2: {ab: 1}, ...}
                if dictionary is None or pat_pair[0] != L:
                    LDS[pat_pair][group] += count
        # 如果字典不为空，则根据字典计算 L 字典
        if dictionary is not None:  # 如果字典不为空，则根据字典
            L_re = re.compile(r'[a-zA-Z]+')
            for w, count in dictionary.items():
                for pat in L_re.findall(w):
                    LDS[L, len(pat)][pat] += count  # 此时的键为 (L，字母串长度)

        def process(counter):
            items = list(counter.keys())
            cumcounts = numpy.array(list(counter.values())).cumsum()
            return counter, items, cumcounts

        self.structures = process(structures)  # 三元组：第一个元素为structures，第二个元素为键列表，第三个值为值的梯形和
        self.LDS = {k: process(v) for k, v in LDS.items()}  # LDS 的每个值同样转为三元组
        self.using_dictionary = dictionary is not None

    def generate_by_threshold(self, threshold):

        LDS = self.LDS
        struct_counter, _, struct_cumcounts = self.structures
        tot_structures = struct_cumcounts[-1]

        for struct, count in struct_counter.items():
            left = threshold + math.log2(count / tot_structures)
            if left < 0:
                continue
            # stack: left logprob, prefix, structure
            stack = [(left, '', struct)]
            while stack:
                left, prefix, structure = stack.pop()
                try:
                    counter, _, cumcounts = LDS[structure[0]]
                except IndexError:
                    # empty password
                    assert prefix == ''
                    yield threshold - left, ''
                total = cumcounts[-1]
                for s, count in counter.items():
                    new_left = left + math.log2(count / total)
                    if new_left > 0:
                        if len(structure) == 1:
                            yield threshold - new_left, prefix + s
                        else:
                            stack.append((new_left, prefix + s, structure[1:]))

    def generate(self):

        def pick(processed):
            # 从列表中选择一个模板/单元，并返回对数概率
            counter, items, cum_counts = processed
            total = cum_counts[-1]
            idx = bisect.bisect_right(cum_counts, random.randrange(total))
            item = items[idx]
            return -math.log2(counter[item] / total), item

        lp, structure = pick(self.structures)
        res = ''
        for pat_pair in structure:
            try:
                lpnew, group = pick(self.LDS[pat_pair])
            except KeyError:
                # 此时字母串来自字典
                # we're using a dictionary with no long enough words. We just
                # decrease the length until we find something suitable.
                pat_t, l = pat_pair
                while True:
                    l -= 1
                    assert l >= 0
                    try:
                        if (pat_t, l) in self.LDS:
                            break
                    except KeyError:
                        pass
                lpnew, group = pick(self.LDS[pat_t, l])
            lp += lpnew
            res += group
        return lp, res

    def logprob(self, word, leaveout=False):

        structure, groups = patterns(word)
        counter, _, cumsum = self.structures
        try:
            res = -math.log2((counter[structure] - leaveout) /
                             (cumsum[-1] - leaveout))
        except (ZeroDivisionError, ValueError):
            return float('inf')
        assert res > 0

        LDS = self.LDS
        for pat_pair, group in zip(structure, groups):
            try:
                counter, _, cumsum = LDS[pat_pair]
            except KeyError:
                return float('inf')
            lo = leaveout and (not self.using_dictionary or
                               pat_pair[0] != L)
            try:
                res -= math.log2((counter[group] - lo) /
                                 (cumsum[-1] - lo))
            except (ZeroDivisionError, ValueError):
                return float('inf')

        return res
