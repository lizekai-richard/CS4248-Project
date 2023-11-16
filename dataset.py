import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Union


class SQuADDataset(Dataset):
    def __init__(self, data, tokenizers, max_length):
        super().__init__()
        self.data = data
        self.tokenizers = tokenizers
        self.model_names = tokenizers.keys()
        self.max_length = max_length

    def __getitem__(self, item):
        example = self.data[item]
        context = example['context']
        question = example['question']
        answer = example['answers']

        input_by_model = {}
        for model_name in self.model_names:
            tokenizer = self.tokenizers[model_name]
            input_by_model[model_name] = self.process_examples(tokenizer, question, context, answer)

        return input_by_model

    def __len__(self):
        return len(self.data)

    def process_examples(self, tokenizer, question, context, answer):

        tokenized_input = tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors='pt'
        )
        input_ids = tokenized_input['input_ids'][0]
        attention_mask = tokenized_input['attention_mask'][0]

        offset_mapping = tokenized_input.pop("offset_mapping")
        sample_map = tokenized_input.pop("overflow_to_sample_mapping")
        start_position = -1
        end_position = -1

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = tokenized_input.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_position = 0
                end_position = 0
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_position = idx - 1

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_position = idx + 1

        tokenized_input['input_ids'] = input_ids
        tokenized_input['attention_mask'] = attention_mask
        tokenized_input["start_positions"] = start_position
        tokenized_input["end_positions"] = end_position
        tokenized_input["answer_text"] = answer['text']
        return tokenized_input


class MCQDataset(Dataset):
    def __init__(self, data, model_names, tokenizer, max_length):
        super().__init__()
        self.data = data
        self.model_names = model_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_count = len(model_names)

    def __getitem__(self, item):
        example = self.data[item]
        processed_example = self.preprocess_example(example)

        return processed_example

    def __len__(self):
        return len(self.data)

    def preprocess_example(self, example):
        context = [example["context"]] * self.model_count
        question = example["question"]
        qna = [f"{question} {example[ans]}" for ans in self.model_names]
        return self.tokenizer(context, qna, truncation=True, max_length=self.max_length)

    def collate(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        return batch

    def decode_answer(self, pred_labels):
        predictions = []
        for i, l in enumerate(pred_labels):
            m_name = self.model_names[l]
            predictions.append(self.data[i][m_name])
        return predictions


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def check_answer_mapping(tokenizer, input_ids, start_position, end_position):
    answer_ids = input_ids[start_position: end_position + 1]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)


# wrong_answers is a dict of lists. correct_answers is a list, seed for reproducibility of answer shuffling
def preprocess_dataset_for_training_qna(dataset, wrong_answers, tokenizer, seed=0):
    random.seed(seed)
    correct_answers = [x['text'][0] for x in dataset['answers']]
    for k,v in wrong_answers.items():
        dataset = dataset.add_column(k, v)
    dataset = dataset.add_column("correct_answer", correct_answers)

    labels = [random.randint(0, len(wrong_answers)) for _ in correct_answers] 
    dataset = dataset.add_column("label", labels)

    ans_names = ["correct_answer"]
    ans_names.extend(wrong_answers.keys())
    
    def pf(examples):
        context = [[c] * 4 for c in examples["context"]]
        question = examples["question"]
        labels = examples["label"]
        qna = [
            [f"{q} {examples[ans][i]}" for ans in ans_names] for i, q in enumerate(question)
        ]
        for i, q in enumerate(qna):
            label = labels[i]
            q[0], q[label] = q[label], q[0]
        context = sum(context, [])
        qna = sum(qna, [])

        tokenized_examples = tokenizer(context, qna, truncation="only_first")
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    
    return dataset.map(pf, batched=True)


if __name__ == '__main__':

    tokenizers = {
        'bert-base': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'roberta-base': AutoTokenizer.from_pretrained('roberta-base'),
        'albert-base': AutoTokenizer.from_pretrained('albert-base-v2')
    }

    data = load_dataset("squad")

    train_dataset = SQuADDataset(data['train'], tokenizers, 512)
    dev_dataset = SQuADDataset(data['validation'], tokenizers, 512)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False)

    sample_batch = next(iter(train_loader))
    for i in range(4):
        decoded_mapping = check_answer_mapping(tokenizers['bert-base'],
                                               sample_batch['bert-base']['input_ids'][i],
                                               sample_batch['bert-base']['start_positions'][i],
                                               sample_batch['bert-base']['end_positions'][i])
        print("Mapped answer: ", decoded_mapping)
        print("True answer: ", sample_batch['bert-base']['answer_text'][0][i])