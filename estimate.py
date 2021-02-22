#!/usr/bin/env python3
# standard library
import argparse
import csv
import time

# internal imports
import backoff
import model
import ngram_chain
import pcfg


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--password-file', "-i",
                        default="../Data/origin/CSDN/CSDN_10_test.txt",
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
                        default=2,
                        help='minimum n for n-grams')
    parser.add_argument('--max_ngram',
                        type=int,
                        default=5,
                        help='maximum n for n-grams')
    parser.add_argument('--backoff_threshold',
                        type=int,
                        default=10,
                        help='threshold for backoff')
    parser.add_argument('--samplesize',
                        type=int,
                        default=10000,
                        help='sample size for Monte Carlo model')
    parser.add_argument('--maxlen',
                        type=int,
                        default=100,
                        help='max length for generated passwords')
    args = parser.parse_args()

    return args


def build_models(args, training):
    """
    模型建立
    """
    now = time.time()
    models = {'{}-gram'.format(i): ngram_chain.NGramModel(training, i)
              for i in range(args.min_ngram, args.max_ngram + 1)}
    # models['Backoff'] = backoff.BackoffModel(training, threshold=args.backoff_threshold)
    # models['PCFG'] = pcfg.PCFG(training)
    print("[ + ] models have been built in {}".format(time.time()-now))
    return models


if __name__ == '__main__':
    args = parser_args()

    with open(args.dataset, 'rt') as f:
        train_data = [w.strip('\r\n') for w in f]

    models = build_models(args, train_data)

    # 蒙特卡洛模型建立
    now = time.time()
    montaCarlo_samples = {name: list(model.sample(args.samplesize, args.maxlen))
                          for name, model in models.items()}
    estimators = {name: model.PosEstimator(sample)
                  for name, sample in montaCarlo_samples.items()}
    print("[ + ] estimators have been built in {}".format(time.time()-now))
    modelnames = sorted(models)

    # 对目标文本生成估计结果
    with open(args.guess, "r", encoding="utf-8") as guesses:
        with open(args.output, "w", encoding="utf-8", newline='') as estimate_file:
            # newline=''，防止行之间留空行
            writer = csv.writer(estimate_file)
            writer.writerow(['password'] + modelnames)

            # 蒙特卡洛估计
            for password in guesses:
                password = password.strip('\r\n')

                estimations = [estimators[name].position(models[name].logprob(password))
                               for name in modelnames]
                writer.writerow([password] + estimations)
    print("[ + ] done")
