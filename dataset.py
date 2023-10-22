import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


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


def check_answer_mapping(tokenizer, input_ids, start_position, end_position):
    answer_ids = input_ids[start_position: end_position + 1]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)


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




