import json
import torch

from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm

def extract_predicted_spans(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        return_tensors="pt",
        truncation="only_second",
    )

    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    predicted_span = tokenizer.decode(predict_answer_tokens)

    d = {"id": example["id"], "predicted_span": predicted_span}
    return d

model = AutoModelForQuestionAnswering.from_pretrained("valhalla/electra-base-discriminator-finetuned_squadv1")

tokenizer = AutoTokenizer.from_pretrained("valhalla/electra-base-discriminator-finetuned_squadv1")

validation_data = load_dataset("squad", split="validation")

predicted = {}
for i in tqdm(range(len(validation_data))):
# for example in dataset:
    example = validation_data[i]
    result = extract_predicted_spans(example)
    predicted[result["id"]] = result["predicted_span"]

with open("predicted.txt", "w") as f:
    json.dump(predicted, f)