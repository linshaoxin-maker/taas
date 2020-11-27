# Topic-Aware Abstractive Text Summarization
**This repo is working in progress. Currently the project is modified using ```transformer``` from HuggingFace and NTM from @YongfeiYan. We are planning to release the checkpoints in the sooner future, and we are still working on this repo. Thank you!**



This repository is the artifact associated with our paper **Topic-Aware Abstractive Text Summarization**. In this paper, we propose a topic-aware abstractive summarization (TAAS) framework by leveraging the underlying semantic structure of documents represented by their latent topics.

Note: our work is built on top of HuggingFace ```transformers```. There are two key components in our paper: topic modeling component and summarization component. 

- For summarization component, our code is built on top of the summarization pipeline built by HuggingFace. You can visit their example code in this link: https://github.com/huggingface/transformers/tree/master/examples/seq2seq.

- For topic modeling component, we modified the code from https://github.com/YongfeiYan/Neural-Document-Modeling.

## Dataset
We use CNN/Daily Mail dataset in our paper. You can download the dataset using the link provided by HuggingFace:
```bash
cd data
wget https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz
tar -xzvf cnn_dm_v2.tgz
mv cnn_cln cnndm
```

## Local Setup

Tested with Python 3.7 via virtual environment. Clone the repo, go to the repo folder, setup the virtual environment, and install the required packages:

```bash
$ python3.7 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Model Training
To train TAAS, you can execute the following command:

```bash
DATA_DIR=./data/cnndm/
SUFFIX=taas
OUTPUT_DIR=./log/$SUFFIX

./finetune.sh \
    --data_dir $DATA_DIR \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 100 \
    --model_name_or_path sshleifer/distilbart-cnn-12-6 \
    --save_top_k 5 \
    --early_stopping_patience 50 \
    --warmup_steps 10 \
```



## Generate Summary

You can use the trained checkpoint to generate the summary on the testing set.

```bash
export DATA_DIR=./data/cnndm
SUFFIX=taas
OUTPUT_FILE=./output/cnndm/$SUFFIX/output-$SUFFIX.txt
OUTPUT_DIR=./log/$SUFFIX
python run_eval.py sshleifer/distilbart-cnn-12-6 $DATA_DIR/test.source $OUTPUT_FILE \
    --reference_path $DATA_DIR/test.target \
    --task summarization \
    --device cuda \
    --fp16 \
    --bs 32 \
    --finetune_flag 1 \
    --checkpoint_path $OUTPUT_DIR \
```



## Performance Evaluation

We use ROUGE score as the evaluation metric in our paper.

```bash
SUFFIX=taas
OUTPUT_FILE=./output/cnndm/$SUFFIX/output-$SUFFIX.txt
SCORE_FILE=./output/cnndm/$SUFFIX/score-$SUFFIX.txt
python evaluate.py --generated $OUTPUT_FILE --golden ./data/cnndm/org_data/test.target > $SCORE_FILE
```

The best performance we achieved currently is:

```
1 ROUGE-1 Average_R: 0.48810 (95%-conf.int. 0.48545 - 0.49062)
1 ROUGE-1 Average_P: 0.42147 (95%-conf.int. 0.41874 - 0.42408)
1 ROUGE-1 Average_F: 0.44058 (95%-conf.int. 0.43839 - 0.44281)
---------------------------------------------
1 ROUGE-2 Average_R: 0.23262 (95%-conf.int. 0.22988 - 0.23522)
1 ROUGE-2 Average_P: 0.20166 (95%-conf.int. 0.19915 - 0.20415)
1 ROUGE-2 Average_F: 0.21017 (95%-conf.int. 0.20769 - 0.21257)
---------------------------------------------
1 ROUGE-L Average_R: 0.45306 (95%-conf.int. 0.45039 - 0.45556)
1 ROUGE-L Average_P: 0.39123 (95%-conf.int. 0.38858 - 0.39384)
1 ROUGE-L Average_F: 0.40899 (95%-conf.int. 0.40667 - 0.41120)
```



## Example

We present the summaries of the same article from CNN/DM dataset from TAAS and other baseline models in ```example``` folder.
