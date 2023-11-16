# CS4248-Project

## Prerequisites

To run the project, perform the following steps to install all required dependencies

0. This project was tested on Python version 3.11.6
1. Run `pip install -r project-requirements.txt`
2. Download and place the files `dev-v1.1.json` and `train-v1.1.json` in a folder called `data/`. The folder structure should look like the diagram below.
3. Ensure you are in the main folder `CS4248-Project/` before running the shell scripts.

```
CS4248-Project/
├─ data/    # Place SQuAD dataset files here
│  ├─ dev-v1.1.json
│  ├─ train-v1.1.json
├─ notebooks/
├─ predictions/
├─ scripts/
│  ├─ eval.sh
│  ├─ train.sh
├─ requirements.txt
├─ project-requirements.txt
├─ run.py
├─ README.md
├─ ...
```

## Train

To train the MCQ model, run

```bash
bash scripts/train.sh
```

You can modify configurations in the train.sh

```bash
ALBERT_PATH="twmkn9/albert-base-v2-squad2"
DEBERTA_PATH="deepset/deberta-v3-base-squad2"
ELECTRA_PATH="deepset/electra-base-squad2"
# change to yours
TRAIN_DATA_PATH="YOUR DATA PATH"
DEV_DATA_PATH="YOUR DATA PATH"
MCQ_MODEL_PATH="YOUR MODEL PATH"
OUTPUT_PATH="YOUR MODEL CHECKPOINT PATH"

python3 run.py \
--do_train True \
--output_path $OUTPUT_PATH \
--train_data_path $TRAIN_DATA_PATH \
--dev_data_path $DEV_DATA_PATH \
--albert_path $ALBERT_PATH \
--deberta_path $DEBERTA_PATH \
--electra_path $ELECTRA_PATH \
--mcq_model_path $MCQ_MODEL_PATH \
# you can change the following training settings
--max_length 512 \
--batch_size 32 \
--eval_steps 5000 \
--save_steps 5000
```

## Evaluate

To do evaluation, you can run

```bash
bash scripts/eval.sh
```

Similarly, you should modify the script before you run

```bash
ALBERT_PATH="twmkn9/albert-base-v2-squad2"
DEBERTA_PATH="deepset/deberta-v3-base-squad2"
ELECTRA_PATH="deepset/electra-base-squad2"
# change to yours
TRAIN_DATA_PATH="YOUR DATA PATH"
DEV_DATA_PATH="YOUR DATA PATH"
MCQ_MODEL_PATH="YOUR TRAINED MODEL CHECKPOINT"
SAVE_PATH="YOUR PREDICTIONS SAVE PATH"

python3 run.py \
--do_train False \
--save_path $SAVE_PATH \
--train_data_path $TRAIN_DATA_PATH \
--dev_data_path $DEV_DATA_PATH \
--albert_path $ALBERT_PATH \
--deberta_path $DEBERTA_PATH \
--electra_path $ELECTRA_PATH \
--mcq_model_path $MCQ_MODEL_PATH
```
