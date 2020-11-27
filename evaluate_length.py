"""
Evaluate the performance for our model based on different values of K
"""
import sys
import string
import argparse
import tempfile
import os
import time
import shutil
from bs_pyrouge import Rouge155
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--generated", type=str, help="generated output file.")
parser.add_argument("--golden", type=str, help="Gold output file.")
parser.add_argument("--output", type=str, help="Path to save the output score")
parser.add_argument("--duplicate_rate", type=float, default=0.7,
                    help="If the duplicat rate (compared with history) is large, we can discard the current sentence.")
parser.add_argument("--trunc_len", type=int, default=0,
                    help="Truncate line by the maximum length.")
args = parser.parse_args()

fin = open(args.generated, 'r', encoding='utf-8')
fgolden = open(args.golden, 'r', encoding='utf-8')
dedup_rate = args.duplicate_rate
trunc_len = args.trunc_len

_tok_dict = {"(": "-LRB-", ")": "-RRB-",
             "[": "-LSB-", "]": "-RSB-",
             "{": "-LCB-", "}": "-RCB-"}

def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True

def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    text = ' '.join([x for x in output_tokens])
    fine_text = text.replace(' ##', '')
    # return " ".join(output_tokens)
    return fine_text

def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set)/len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list

def test_rouge(cand, ref):
    temp_dir = tempfile.mkdtemp()
    candidates = cand
    references = ref
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict

def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )

def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter

def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)

# create lists for storing the rouge score
rouge1, rouge2, rougeL = [], [], []

num_sentence = []

generated_list = []
for line in fin:
    buf = []
    modified_line = line.strip()
    num_sentence.append(len(modified_line.split('[X_SEP]')))
    # for sentence in modified_line.split('.'):
    modified_line = modified_line.replace('[ X_SE P', '[X_SEP] P').replace('[X_ SEP]', '[X_SEP]').replace('X_SEp]', '[X_SEP]').replace('[X _SEP', '[X_SEP]').replace('[ X_SEp]', '[X_SEP]')
    for sentence in modified_line.split('[X_SEP]'):
        sentence = fix_tokenization(sentence)
        if any(get_f1(sentence, s) > 1.0 for s in buf):
            continue
        s_len = len(sentence.split())
        if s_len <= 4:
            continue
        buf.append(sentence)
    if dedup_rate < 1:
        buf = remove_duplicate(buf, dedup_rate)
    if trunc_len:
        num_left = trunc_len
        trunc_list = []
        for bit in buf:
            tk_list = bit.split()
            n = min(len(tk_list), num_left)
            trunc_list.append(' '.join(tk_list[:n]))
            num_left -= n
            if num_left <= 0:
                break
    else:
        trunc_list = buf
    trunc_list = [item.replace('-LSB-','') for item in trunc_list]
    generated_list.append("\n".join(trunc_list))

golden_list = []
for line in fgolden:
    line = line.strip().replace(" <S_SEP> ", '\n')
    golden_list.append(line)

summary_length_char = []

for i in range(len(generated_list)):
    model_generate = [generated_list[i]]
    model_golden = [golden_list[i]]
    summary_length_char.append(len(model_generate[0]))
    # print(model_generate)
    # print(model_golden)
    scores = test_rouge(model_generate, model_golden)
    # record rouge 1 f
    rouge1.append(scores['rouge_1_f_score'])
    # record rouge 2 f
    rouge2.append(scores['rouge_2_f_score'])
    # record rouge l f
    rougeL.append(scores['rouge_l_f_score'])

fin = open(f"./data/cnndm/org_data/test.source", 'r', encoding='utf-8')
fout = open(f"./data/cnndm/org_data/test.target", 'r', encoding='utf-8')

# num_sentences: how many sentences in the article
# article_char_length: length of article based on char
# article_token_length: length of article based on token
num_sentences, article_char_length, article_token_length = [], [], []

article_list, source_list = [], []

for line in fin:
    article_list.append(line)

    sentence = line.strip().split('<S_SEP>')
    # calculate how many sentences in the dataset
    num_sentences.append(len(sentence))
    # calculate the length based on char
    content = "".join(sentence)
    article_char_length.append(len(content))
    token_list = content.split(" ")
    article_token_length.append(len(token_list))

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(num_sentences, rouge1, alpha=0.6)
ax.set_title("Number of Sentences (Article) vs ROUGE-1")
ax.set_xlabel("Number of Sentences (Article)")
ax.set_ylabel("ROUGE 1 Score")
# plt.show()
fig.savefig(f"{args.output}/article-sentence-rouge1.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(num_sentences, rouge2, alpha=0.6)
ax.set_title("Number of Sentences (Article) vs ROUGE-2")
ax.set_xlabel("Number of Sentences (Article)")
ax.set_ylabel("ROUGE 2 Score")
# plt.show()
fig.savefig(f"{args.output}/article-sentence-rouge2.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(num_sentences, rougeL, alpha=0.6)
ax.set_title("Number of Sentences (Article) vs ROUGE-L")
ax.set_xlabel("Number of Sentences (Article)")
ax.set_ylabel("ROUGE L Score")
# plt.show()
fig.savefig(f"{args.output}/article-sentence-rougeL.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(article_char_length, rouge1, alpha=0.6)
ax.set_title("Length of Article (Char) vs ROUGE-1")
ax.set_xlabel("Length of Article (Char)")
ax.set_ylabel("ROUGE 1 Score")
# plt.show()
fig.savefig(f"{args.output}/article-length-char-rouge1.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(article_char_length, rouge2, alpha=0.6)
ax.set_title("Length of Article (Char) vs ROUGE-2")
ax.set_xlabel("Length of Article (Char)")
ax.set_ylabel("ROUGE 2 Score")
# plt.show()
fig.savefig(f"{args.output}/article-length-char-rouge2.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(article_char_length, rougeL, alpha=0.6)
ax.set_title("Length of Article (Char) vs ROUGE-L")
ax.set_xlabel("Length of Article (Char)")
ax.set_ylabel("ROUGE L Score")
# plt.show()
fig.savefig(f"{args.output}/article-length-char-rougeL.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(article_token_length, rouge1, alpha=0.6)
ax.set_title("Length of Article (Token) vs ROUGE-1")
ax.set_xlabel("Length of Article (Token)")
ax.set_ylabel("ROUGE 1 Score")
fig.savefig(f"{args.output}/article-length-token-rouge1.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(article_token_length, rouge2, alpha=0.6)
ax.set_title("Length of Article (Token) vs ROUGE-2")
ax.set_xlabel("Length of Article (Token)")
ax.set_ylabel("ROUGE 2 Score")
fig.savefig(f"{args.output}/article-length-token-rouge2.png")
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.scatter(article_token_length, rougeL, alpha=0.6)
ax.set_title("Length of Article (Token) vs ROUGE-L")
ax.set_xlabel("Length of Article (Token)")
ax.set_ylabel("ROUGE L Score")
fig.savefig(f"{args.output}/article-length-token-rougeL.png")
plt.close(fig)

# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.scatter(num_sentence, rouge1, alpha=0.6)
# ax.set_title("Number of Sentences vs ROUGE-1")
# ax.set_xlabel("Number of Sentences")
# ax.set_ylabel("ROUGE 1 Score")
# # plt.show()
# fig.savefig(f"{args.output}/sentence-rouge1.png")
# plt.close(fig)
#
# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.scatter(num_sentence, rouge2, alpha=0.6)
# ax.set_title("Number of Sentences vs ROUGE-2")
# ax.set_xlabel("Number of Sentences")
# ax.set_ylabel("ROUGE 2 Score")
# # plt.show()
# fig.savefig(f"{args.output}/sentence-rouge2.png")
# plt.close(fig)
#
# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.scatter(num_sentence, rougeL, alpha=0.6)
# # plt.show()
# ax.set_title("Number of Sentences vs ROUGE-L")
# ax.set_xlabel("Number of Sentences")
# ax.set_ylabel("ROUGE L Score")
# fig.savefig(f"{args.output}/sentence-rougel.png")
# plt.close(fig)
#
# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.scatter(summary_length_char, rouge1, alpha=0.6)
# ax.set_title("Length of Articles vs ROUGE-1")
# ax.set_xlabel("Length of Articles")
# ax.set_ylabel("ROUGE 1 Score")
# # plt.show()
# fig.savefig(f"{args.output}/length-rouge1.png")
# plt.close(fig)
#
# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.scatter(summary_length_char, rouge2, alpha=0.6)
# ax.set_title("Length of Articles vs ROUGE-2")
# ax.set_xlabel("Length of Articles")
# ax.set_ylabel("ROUGE 2 Score")
# # plt.show()
# fig.savefig(f"{args.output}/length-rouge2.png")
# plt.close(fig)
#
# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.scatter(summary_length_char, rougeL, alpha=0.6)
# # plt.show()
# ax.set_title("Length of Articles vs ROUGE-L")
# ax.set_xlabel("Length of Articles")
# ax.set_ylabel("ROUGE L Score")
# fig.savefig(f"{args.output}/length-rougel.png")
# plt.close(fig)

# with open(f"{args.output}", "w") as f:
#     f.write(f"ROUGE 1: {np.mean(rouge1)}\n")
#     f.write(f"ROUGE 2: {np.mean(rouge2)}\n")
#     f.write(f"ROUGE L: {np.mean(rougeL)}\n")