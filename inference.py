import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer
from tqdm import tqdm
from transformers import pipeline

# Dev Accuracy: 87.36, F1: 93.73
MODEL_NAME = "Palak/microsoft_deberta-large_squad"
MAX_SEQ_LEN = 512

class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               paragraph_text,
               orig_answer_text=None,
               start_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (printable_text(self.qas_id))
    s += ", question_text: %s" % (
        printable_text(self.question_text))
    s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s

def read_squad_examples(input_data, is_training):
  """Read a SQuAD json file into a list of SquadExample."""


  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        orig_answer_text = None
        is_impossible = False

        if is_training:
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            start_position = answer["answer_start"]
          else:
            start_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            is_impossible=is_impossible)
        examples.append(example)

  return examples




class SentDataset(Dataset):
    def __init__(self, train_path, training=False):
        with open(train_path, "r",encoding="utf-8") as f:
            x = json.load(f)
        self.data = read_squad_examples(x['data'], training)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]
    
def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    qids = [b.qas_id for b in batch]
    qns = [b.question_text for b in batch]
    ct = [b.paragraph_text for b in batch]

    r = tokenizer(qns, ct, return_tensors="pt", padding="max_length", truncation="only_second", max_length=MAX_SEQ_LEN)
    
    starts = [b.start_position for b in batch]
    ends = [(b.start_position + len(b.orig_answer_text) - 1) for b in batch]
    
    return qids, r, torch.tensor(starts), torch.tensor(ends)

dtest = SentDataset("data/dev-v1.1.json")



if torch.cuda.is_available():
    device_str = 'cuda:{}'.format(0)
else:
    device_str = 'cpu'
device = torch.device(device_str)


ca = {}

qan = pipeline(
    "question-answering",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0
)
with torch.no_grad():
    for step, data in tqdm(enumerate(dtest)):
        qid, question, context = data.qas_id, data.question_text, data.paragraph_text
        # inputs = tokenizer(question, context, return_tensors="pt", padding="max_length", truncation="only_second", max_length=MAX_SEQ_LEN).to(device)
        # outputs = model(**inputs)
        # answer_start_index = outputs.start_logits.argmax()
        # answer_end_index = outputs.end_logits.argmax()
        # predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        ca[qid] = qan(question=question,context=context)['answer']
        # ca[qid] = tokenizer.decode(predict_answer_tokens)
        
    


with open("pred.json", "w", encoding="utf-8") as outfile:
    json.dump(ca,outfile)