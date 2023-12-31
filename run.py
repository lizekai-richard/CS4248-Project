import os
import time
import random
import torch
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from dataset import SQuADDataset, MCQDataset, DataCollatorForMultipleChoice, preprocess_dataset_for_training_qna
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForMultipleChoice, TrainingArguments, \
    Trainer


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
# assumes index correspondence
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
def single_model_generate_predictions(tokenizers, models, data_loader):
    model_names = models.keys()
    predictions = {key: [] for key in model_names}
    for batch in tqdm(data_loader):
        for model_name in model_names:
            tokenizer = tokenizers[model_name]
            model = models[model_name]

            input_ids, attention_mask = batch[model_name]['input_ids'], batch[model_name]['attention_mask']

            if torch.cuda.is_available():
                input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            # ans_starts, ans_ends = batch[model]['start_positions'], batch[model]['end_positions']

            outputs = model(input_ids, attention_mask)

            pred_starts = outputs.start_logits.argmax(dim=-1)
            pred_ends = outputs.end_logits.argmax(dim=-1)

            for i, (s, e) in enumerate(zip(pred_starts, pred_ends)):
                pred_answer = tokenizer.decode(input_ids[i][s: e + 1], skip_special_tokens=True)
                predictions[model_name].append(pred_answer)

    return predictions


# need dataset for decoding answers
@torch.inference_mode()
def ensemble_model_generate_predictions(model, data_loader, dataset):
    prediction_labels = []
    for step, data in enumerate(data_loader, 0):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()

        outputs = model(input_ids, attention_mask)
        prediction_labels.extend(outputs.logits.argmax(axis=1).tolist())

    return dataset.decode_answer(prediction_labels)

def collate_fn(batch):
    model_names = list(batch[0].keys())
    entry_names = list(batch[0][model_names[0]].keys())
    new_batch = {model_name: {} for model_name in model_names}
    for model_name in model_names:
        for example in batch:
            for key in entry_names:
                if key == 'token_type_ids':
                    continue
                if key not in new_batch[model_name]:
                    new_batch[model_name][key] = []
                new_batch[model_name][key].append(example[model_name][key])

    for model_name in model_names:
        new_batch[model_name]["start_positions"] = torch.tensor(new_batch[model_name]["start_positions"])
        new_batch[model_name]["end_positions"] = torch.tensor(new_batch[model_name]["end_positions"])
        new_batch[model_name]["input_ids"] = torch.stack(new_batch[model_name]["input_ids"])
        new_batch[model_name]["attention_mask"] = torch.stack(new_batch[model_name]["attention_mask"])
    return new_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_path", type=str, default="/path/to/save")
    parser.add_argument("--output_path", type=str, default="/path/to/output")
    parser.add_argument("--train_data_path", type=str, default="/path/to/train/data")
    parser.add_argument("--dev_data_path", type=str, default="/path/to/dev/data")
    parser.add_argument("--deberta_path", type=str, default="/path/to/bert")
    parser.add_argument("--albert_path", type=str, default="/path/to/albert")
    parser.add_argument("--electra_path", type=str, default="/path/to/electra")
    parser.add_argument("--mcq_model_path", type=str, default="/path/to/mcq/model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    models = {
        'deberta': AutoModelForQuestionAnswering.from_pretrained(args.deberta_path).to(device),
        'albert': AutoModelForQuestionAnswering.from_pretrained(args.albert_path).to(device),
        'electra': AutoModelForQuestionAnswering.from_pretrained(args.electra_path).to(device)
    }

    mcq_tokenizer = AutoTokenizer.from_pretrained(args.mcq_model_path)
    mcq_model = AutoModelForMultipleChoice.from_pretrained(args.mcq_model_path).to(device)

    if args.do_train:
        print("For convenience of format consistency, we use dataset on huggingface to do the training. "
              "But the evaluation is conducted on the given dataset")
        squad = load_dataset("squad")

        print("Generating single model predictions on the train dataset...")
        single_model_predictions_on_train_data = single_model_generate_predictions(tokenizers, models,
                                                                                   train_loader)
        train_mcq_ds = preprocess_dataset_for_training_qna(squad['train'], single_model_predictions_on_train_data,
                                                           mcq_tokenizer)

        print("Generating single model predictions on the dev dataset...")
        single_model_predictions_on_dev_data = single_model_generate_predictions(tokenizers, models,
                                                                                 dev_loader)
        dev_mcq_ds = preprocess_dataset_for_training_qna(squad['validation'], single_model_predictions_on_dev_data,
                                                         mcq_tokenizer)

        training_args = TrainingArguments(
            output_dir=args.output_path,
            evaluation_strategy="steps",
            eval_steps=5000,
            save_strategy="steps",
            save_steps=5000,
            load_best_model_at_end=True,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            weight_decay=0.01,
            push_to_hub=False,  # if you want to push to huggingface, please login first and change tis value to Trues
            prediction_loss_only=True
        )

        trainer = Trainer(
            model=mcq_model,
            args=training_args,
            train_dataset=train_mcq_ds,
            eval_dataset=dev_mcq_ds,
            tokenizer=mcq_tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=mcq_tokenizer),
        )

        trainer.train()
        trainer.save_model(args.output_path)
    else:
        print("Generating single model predictions on the dev dataset...")
        single_model_predictions_on_dev_data = single_model_generate_predictions(tokenizers, models,
                                                                                 dev_loader)
        eval_mcq_data = add_predictions_to_dataset(dev_data, single_model_predictions_on_dev_data)

        # Step 3: create the mcq dataset for prediction
        eval_mcq_ds = MCQDataset(eval_mcq_data, list(models.keys()), mcq_tokenizer, args.max_length)
        eval_mcq_loader = DataLoader(eval_mcq_ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=eval_mcq_ds.collate)
        # Step 4: generate predictions
        mcq_predictions = ensemble_model_generate_predictions(mcq_model, eval_mcq_loader, eval_mcq_ds)

        with open(args.save_path, "w") as f:
            json.dump(mcq_predictions, f)




