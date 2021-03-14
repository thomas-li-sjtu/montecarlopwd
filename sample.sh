# !bin/sh
# REM output-dir format
# REM out_[model]_[min_len]_[max_len]_[epoch]

python main.py --target-file ../Data/origin/Gmail_clean/Gmail_clean.txt --model PCFG

python main.py --target-file ../Data/origin/Gmail_clean/Gmail_clean.txt --model markov --min_ngram 4 --max_ngram 4

python main.py --target-file ../Data/origin/Gmail_clean/Gmail_clean.txt --model markov --min_ngram 3 --max_ngram 3

python main.py --target-file ../Data/origin/Gmail_clean/Gmail_clean.txt --model markov --min_ngram 2 --max_ngram 2
