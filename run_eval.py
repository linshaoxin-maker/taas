import argparse
import json
import re
from pathlib import Path

import torch
from gensim.corpora import Dictionary
from tqdm import tqdm
from data_utils import DocDataset
from mlutils.exp import yaml_load
from mlutils.pt.training import GSMTrainer, extend_config_reference
from torch.utils.data import DataLoader
from gensim.test.utils import datapath
from torch.autograd import Variable

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from .utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
except ImportError:
    from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def generate_summaries(
        examples: list,
        out_file: str,
        model_name: str,
        batch_size: int = 8,
        device: str = DEFAULT_DEVICE,
        fp16=True,
        task="summarization",
        decoder_start_token_id=None,
        finetune_flag: int = 0,
        checkpoint_path: str = "",
        **gen_kwargs,
) -> None:
    fout = Path(out_file).open("w", encoding="utf-8")

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if our goal is to evaluate the original checkpoint
    if finetune_flag < 1:
        # initialize the model checkpoints
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    # if our goal is to evaluate our fine-tuned checkpoint
    else:
        # load the finetuned checkpoints
        model = AutoModelForSeq2SeqLM.from_pretrained(f"{checkpoint_path}/best_tfmr").to(device)

    if fp16:
        model = model.half()
    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)

    # update config with summarization specific params
    use_task_specific_params(model, task)

    for batch in tqdm(list(chunks(examples, batch_size))):
        batch = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)
        input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)

        # -----------------------------------------
        # Topic Modeling - GSM
        # -----------------------------------------
        docs = []
        # load dict
        dictionary = Dictionary.load(datapath('dict-www-cnndm-unigram'))
        # remove [SEP]
        sep_list = ['[SEP_0]', '[SEP_1]', '[SEP_2]', '[SEP_3]', '[SEP_4]', '[SEP_5]', '[SEP_6]', '[SEP_7]',
                    '[SEP_8]', '[SEP_9]']
        # vocab size for topic modeling
        vocab_size = len(dictionary)
        # load config for GSM
        config = yaml_load(f"data/config/gsm.yaml")
        # model
        config['hidden']['features'][0] = vocab_size

        # trainer batch
        config['trainer_batch']['test_sample'] = 1
        config = extend_config_reference(config)
        gsm_trainer = config['GSMtrainer']
        gsm_trainer['base_dir'] = f"log/bart-large-cnn-finetune"
        gsm_trainer = GSMTrainer.from_config(gsm_trainer)

        total_sample = len(batch['input_ids'])

        for batch_num in range(total_sample):
            # extract the batch_sentence
            batch_sentence = tokenizer.decode(batch['input_ids'][batch_num].tolist(), skip_special_tokens=True)
            # change to lowercase and split to list
            batch_sentence_list = batch_sentence.split(" ")
            # remove [SEP]
            batch_sentence_list_nosep = [item for item in batch_sentence_list if item not in sep_list]
            text = ' '.join([x for x in batch_sentence_list_nosep])
            fine_text = text.replace(' ##', '').lower()
            batch_sentence = re.sub(r'[^\w\s]', '', fine_text)
            # batch_sentence: change to the cleaned news for topic modeling
            # change to training data format in topic modeling
            gsm_data_bow = dictionary.doc2bow(batch_sentence.split(" "))
            docs.append(gsm_data_bow)

        # gsm_data: data for topic modeling
        gsm_data = DataLoader(DocDataset(docs, len(dictionary), device='cuda'),
                              batch_size=config['dataset']['batch_size'], drop_last=False, num_workers=0)

        gsm_trainer.__dict__['train_iterator'] = gsm_data

        gsm_loss, gsm_p = gsm_trainer.co_train(vocab_size=vocab_size, training=False)

        del gsm_data

        topic_p = gsm_p.cuda()

        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            topic_p=topic_p,
            **gen_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("save_path", type=str, help="where to save summaries")

    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task", type=str, default="summarization", help="typically translation or summarization")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--finetune_flag", type=int, required=False, default=0, help="0 uses the original checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="", required=False, help="customized checkpoint path")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="decoder_start_token_id (otherwise will look at config)",
    )
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    examples = [x.rstrip() for x in open(args.input_path, encoding='utf8').readlines()]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]

    generate_summaries(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        decoder_start_token_id=args.decoder_start_token_id,
        finetune_flag=args.finetune_flag,
        checkpoint_path=args.checkpoint_path,
    )
    if args.reference_path is None:
        return


if __name__ == "__main__":
    run_generate()