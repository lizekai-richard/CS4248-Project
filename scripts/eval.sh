TRAIN_DATA_PATH="./data/train-v1.1.json"
DEV_DATA_PATH="./data/dev-v1.1.json"
ALBERT_PATH="twmkn9/albert-base-v2-squad2"
DEBERTA_PATH="deepset/deberta-v3-base-squad2"
ELECTRA_PATH="deepset/electra-base-squad2"
MCQ_MODEL_PATH="ensemble_mcq_model"
SAVE_PATH="ensemble_pred.json"

python3 run.py \
--do_train False \
--save_path $SAVE_PATH \
--train_data_path $TRAIN_DATA_PATH \
--dev_data_path $DEV_DATA_PATH \
--albert_path $ALBERT_PATH \
--deberta_path $DEBERTA_PATH \
--electra_path $ELECTRA_PATH \
--mcq_model_path $MCQ_MODEL_PATH \
--max_length 512 \
--batch_size 32

