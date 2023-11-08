import os
import time
import random
import torch
import json
import argparse
from dataset import SQuADDataset, MCQDataset
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

# data is a list of dicts
# predictions is a dict with model name as key and corresponding list of predictions as value
# assumes index correspondance
# outputs a list of dicts
def add_predictions_to_dataset(data, predictions):
    model_names = predictions.keys()
    qna_dataset = []
    for i, item in enumerate(data):
        for name in model_names:
            item[name] = predictions[name][i]
        qna_dataset.append(item)
    return qna_dataset

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


#need dataset for decoding answers
@torch.inference_mode()
def ensemble_model_generate_predictions(model, data_loader, dataset):
    prediction_labels = []
    for step, data in enumerate(data_loader, 0):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']

        outputs = model(input_ids, attention_mask)
        prediction_labels.extend(outputs.logits.argmax(axis=1).tolist())

    return dataset.decode_answer(prediction_labels)


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

    # Step 1: generate predictions for single models:
    # predictions = ...

    # Step 2: add the predictions to the corresponding dataset:
    # mcq_data = add_predictions_to_dataset(<corresponding_data>, predictions)

    # Step 3: create the mcq dataset
    # model_names = [k for k in models.keys()]
    # mcq_tokenizer = AutoTokenizer.from_pretrained(args.mcq_model_path)
    # mcq_ds = MCQDataset(mcq_data, model_names, mcq_tokenizer, args.max_length)
    # mcq_loader = DataLoader(mcq_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mcq_ds.collate)
    
    # Step 4: load mcq model
    # mcq_model = AutoModelForMultipleChoice.from_pretrained(args.mcq_model_path)

    #Step 4: generate predictions
    # mcq_predictions = ensemble_model_generate_predictions(mcq_model, mcq_loader, mcq_ds)




