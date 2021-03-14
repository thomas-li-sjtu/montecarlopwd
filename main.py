#!/usr/bin/env python3
# standard library
import argparse
import csv
import time
import pickle

# internal imports
import backoff
import model
import ngram_chain
import pcfg
import estimator


def dump(filename, data, **kargs):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--password-file', "-i",
                        default="../Data/origin/Rockyou_clean/Rockyou_clean_10_train.txt",
                        dest="dataset",
                        help='password training set')
    parser.add_argument('--guess-file', "-g",
                        default="../Data/origin/CSDN/CSDN_10_test.txt",
                        dest="guess",
                        help='password set to estimate')
    parser.add_argument('--output-file', "-o",
                        default="output.txt",
                        dest="output",
                        help='estimate result')
    parser.add_argument('--min_ngram',
                        type=int,
                        default=4,
                        help='minimum n for n-grams')
    parser.add_argument('--max_ngram',
                        type=int,
                        default=4,
                        help='maximum n for n-grams')
    parser.add_argument('--backoff_threshold',
                        type=int,
                        default=10,
                        help='threshold for backoff')
    parser.add_argument('--samplesize',
                        type=int,
                        default=100000000,
                        help='total generation size for Monte Carlo model')
    parser.add_argument('--itersize',
                        type=int,
                        default=1000000,
                        help='each generation size for Monte Carlo model')
    parser.add_argument('--maxlen',
                        type=int,
                        default=100,
                        help='max length for generated passwords')
    parser.add_argument('--target-file', '-t',
                        default="../Data/origin/CSDN_clean/CSDN_clean.txt",
                        dest='target',
                        help='generate to crack (default: Rockyou train)')
    args = parser.parse_args()

    return args


def build_models(args, training):
    """
    模型建立
    """
    now = time.time()
    # models = {'{}-gram'.format(i): ngram_chain.NGramModel(training, i)
    #           for i in range(args.min_ngram, args.max_ngram + 1)}  # ngram没有考虑稀疏性的问题，这个在Backoff中被解决
    # models = {}
    # models['Backoff'] = backoff.BackoffModel(training, threshold=args.backoff_threshold)
    models = {}
    dictionary = pickle.load(open("../Data/dict.pickle", "rb"))
    models['PCFG'] = pcfg.PCFG(training, dictionary=dictionary)
    print("[ + ] models have been built in {}".format(time.time()-now))
    return models


def load_target(target: str):
    dict_target = {}
    total = 0
    with open(target, "r", encoding="utf-8", errors="ignore") as targets:
        for i in targets:
            total += 1
            i = i[:-1]
            if dict_target.get(i):
                dict_target[i] = [dict_target[i][0] + 1, 0]
            else:
                dict_target[i] = [1, 0]

    print("loaded {} data".format(total))
    return dict_target, total


if __name__ == '__main__':
    args = parser_args()

    with open(args.dataset, 'rt') as f:
        train_data = [w.strip('\r\n') for w in f]

    models = build_models(args, train_data)
    target_dict, total = load_target(args.target)

    # 蒙特卡洛模型建立
    now = time.time()

    iters = int(args.samplesize / args.itersize)
    coverage = {}
    for i in range(iters):
        montaCarlo_samples = {name: list(model.sample(args.itersize, args.maxlen))
                              for name, model in models.items()}

        samples = {}
        for name, sample in montaCarlo_samples.items():
            words = list(words for lp, words in sample)
            samples[name] = words

        for name, sample in samples.items():
            cracked = 0
            for line in sample:
                line = line[:-1]
                if target_dict.get(line) and target_dict[line][1] == 0:
                    target_dict[line] = [target_dict[line][0], 1]

            for value in target_dict.values():
                if value[1] == 1:
                    cracked += value[0]

            coverage[args.itersize*(i+1)] = round(cracked / total, 5)
            print("now the coverage of {} is {}".format(name, coverage[args.itersize*(i+1)]))
            # store coverage every 1000 batches
            dump("{}.pickle".format(name),
                          coverage)



    # montecarlo 估计
    # estimators = {name: estimator.PosEstimator(sample)
    #               for name, sample in montaCarlo_samples.items()}
    # print("[ + ] estimators have been built in {}".format(time.time()-now))
    # modelnames = sorted(models)
    #
    # # 对目标文本生成估计结果
    # with open(args.guess, "r", encoding="utf-8") as guesses:
    #     with open(args.output, "w", encoding="utf-8", newline='') as estimate_file:
    #         # newline=''，防止行之间留空行
    #         writer = csv.writer(estimate_file)
    #         writer.writerow(['password'] + modelnames)
    #
    #         # 蒙特卡洛估计
    #         for password in guesses:
    #             password = password.strip('\r\n')
    #
    #             estimations = [estimators[name].position(models[name].logprob(password))
    #                            for name in modelnames]
    #             writer.writerow([password] + estimations)
    # print("[ + ] done")
