import numpy as np
import pandas as pd

# import the testing file
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

for line in fout:
    source_list.append(line)

#                       Mean    25%     50%     75%
# ----------------------------------------------
# num_sentences         27      17      24      34
# article_char_length   4109    2515    3692    5255
# article_token_length  804     492     724     1030

# print(pd.DataFrame(num_sentences, columns=['num_sentences']).describe())
# print(pd.DataFrame(article_char_length, columns=['article_char_length']).describe())
# print(pd.DataFrame(article_token_length, columns=['article_token_length']).describe())

# divide datasets
num_sentences_short = np.quantile(num_sentences, 0.33)
num_sentences_med = np.quantile(num_sentences, 0.67)

short, med, long = [], [], []

for i in range(len(article_list)):
    sentence = article_list[i].split('<S_SEP>')
    if len(sentence) <= num_sentences_short:
        # belong to short
        short.append(i)
    elif len(sentence) > num_sentences_short and len(sentence) <= num_sentences_med:
        # belong to med
        med.append(i)
    else:
        # belong to long
        long.append(i)

# 4021 in short
# 3841 in med
# 3628 in long

# print(f"{len(short)} in short")
# print(f"{len(med)} in med")
# print(f"{len(long)} in long")
with open('./data/cnndm/length/test-short.source','w') as f:
    for num in short:
        f.write(f"{article_list[num]}")

with open('./data/cnndm/length/test-short.target','w') as f:
    for num in short:
        f.write(f"{source_list[num]}")

with open('./data/cnndm/length/test-med.source','w') as f:
    for num in med:
        f.write(f"{article_list[num]}")

with open('./data/cnndm/length/test-med.target','w') as f:
    for num in med:
        f.write(f"{source_list[num]}")

with open('./data/cnndm/length/test-long.source','w') as f:
    for num in long:
        f.write(f"{article_list[num]}")

with open('./data/cnndm/length/test-long.target','w') as f:
    for num in long:
        f.write(f"{source_list[num]}")

