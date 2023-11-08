import os
import time
import random
import torch
import json
import argparse
from dataset import SQuADDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForMultipleChoice


def reformat_dataset(data):
    reformatted_data = []
    for i in range(len(data['data'])):
        paragraphs = data['data'][i]['paragraphs']
        for pid in range(len(paragraphs)):
            context = paragraphs[pid]['context']
            qas = paragraphs[pid]['qas']

            for qa in qas:
                question = qa['question']
                answer_text = qa['answers'][0]['text']
                answer_start = qa['answers'][0]['answer_start']

                reformatted_data.append({
                    'context': context,
                    'question': question,
                    'answers': {
                        'text': [answer_text],
                        'answer_start': [answer_start]
                    }
                })
    return reformatted_data


@torch.inference_mode()
def single_model_generate_predictions(tokenizers, models, batch):
    model_names = models.keys()
    predictions = {key: [] for key in model_names}
    for model_name in model_names:
        tokenizer = tokenizers[model_name]
        model = models[model_name]

        input_ids, attention_mask = batch[model]['input_ids'], batch[model]['attention_mask']
        # ans_starts, ans_ends = batch[model]['start_positions'], batch[model]['end_positions']

        outputs = model(input_ids, attention_mask)

        pred_starts = outputs.start_logits.argmax()
        pred_ends = outputs.end_logits.argmax()

        for i, (s, e) in enumerate(zip(pred_starts, pred_ends)):
            pred_answer = tokenizer.decode(input_ids[i][s: e + 1])
            predictions[model_name].append(pred_answer)

    return predictions


@torch.inference_mode()
def ensemble_model_generate_predictions(tokenizer, model, data_loader):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/path/to/train/data")
    parser.add_argument("--dev_data_path", type=str, default="/path/to/dev/data")
    parser.add_argument("--deberta_path", type=str, default="/path/to/bert")
    parser.add_argument("--albert_path", type=str, default="/path/to/albert")
    parser.add_argument("--electra_path", type=str, default="/path/to/electra")
    parser.add_argument("--mcq_model_path", type=str, default="/path/to/mcq/model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    tokenizers = {
        'deberta': AutoTokenizer.from_pretrained(args.deberta_path),
        'albert': AutoTokenizer.from_pretrained(args.albert_path),
        'electra': AutoTokenizer.from_pretrained(args.electra_path)
    }

    with open(args.train_data_path, "r") as f:
        train_data = json.load(f)

    with open(args.dev_data_path, "r") as f:
        dev_data = json.load(f)

    train_data = reformat_dataset(train_data)
    dev_data = reformat_dataset(dev_data)

    train_ds = SQuADDataset(train_data, tokenizers, args.max_length)
    dev_ds = SQuADDataset(dev_data, tokenizers, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    models = {
        'deberta': AutoModelForQuestionAnswering.from_pretrained(args.deberta_path),
        'albert': AutoModelForQuestionAnswering.from_pretrained(args.albert_path),
        'electra': AutoModelForQuestionAnswering.from_pretrained(args.electra_path)
    }


