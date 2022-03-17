#importing libraries
from dataclasses import dataclass
from datasets import load_dataset
import numpy as np
import math
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

#set model-configuration details
@dataclass
class Config:
    NUM_EPOCHS = 5
    LR = 1.5e-4
    BASE_PATH = "pretrained_models/"
    MODEL_PATH = BASE_PATH + "NegBioElectra/"
    OUT_PATH = BASE_PATH + "mnli_model"
    BATCH_SIZE = 32
    NUM_LABELS = 3
    MAPPER = {'entailment': 1, 'neutral': 0, 'contradiction': 2}

config = Config()

#loading the data
dataset = load_dataset('json', data_files={"train":"data/mli_train_v1.jsonl","test":"data/mli_test_v1.jsonl","validation":"data/mli_dev_v1.jsonl"})
print(dataset)

#loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

#pre-processing data
def preprocess_function(examples):
    l = {'label':[config.MAPPER[i] for i in examples['gold_label']]}
    l.update(tokenizer(examples['sentence1'], examples['sentence2'], truncation=True,max_length=256))
    return l

pre_tokenizer_columns = set(dataset["train"].features)
encoded_dataset = dataset.map(preprocess_function, batched=True)
tokenizer_columns = list(set(encoded_dataset["train"].features) - pre_tokenizer_columns)
tokenizer_columns.remove('label')
print("Columns added by tokenizer:", tokenizer_columns)

#train-test split
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = encoded_dataset["train"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols=["label"],
    shuffle=True,
    batch_size=config.BATCH_SIZE,
    collate_fn=data_collator,
)
tf_validation_dataset = encoded_dataset["validation"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols=["label"],
    shuffle=False,
    batch_size=config.BATCH_SIZE,
    collate_fn=data_collator,
)

#training model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = TFAutoModelForSequenceClassification.from_pretrained(
    config.MODEL_PATH, num_labels=config.NUM_LABELS,from_pt=True
)

batches_per_epoch = len(encoded_dataset["train"]) // config.BATCH_SIZE
total_train_steps = int(batches_per_epoch * config.NUM_EPOCHS)

optimizer, schedule = create_optimizer(
    init_lr=config.LR, num_warmup_steps=math.ceil(total_train_steps * 0.1), num_train_steps=total_train_steps
)
model.compile(optimizer=optimizer, loss=loss,metrics=['acc'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=config.OUT_PATH,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    save_weights_only=True)

tensorboard_callback = TensorBoard(log_dir=config.BASE_PATH+"logs")
callbacks = [tensorboard_callback,model_checkpoint_callback]
print(model.summary())

history = model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=config.NUM_EPOCHS,
    callbacks=callbacks,
)

#evaluate on validation dataset
model.load_weights(config.OUT_PATH)

tf_test_dataset = encoded_dataset["test"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols=["label"],
    shuffle=False,
    batch_size=config.BATCH_SIZE,
    collate_fn=data_collator,
)

print(model.evaluate(tf_test_dataset))






