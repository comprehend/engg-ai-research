#import libraries
import pandas as pd
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

#set model-configuration details
EPOCHS = 15
BATCH_SIZE = 16
BASE_MODEL_PATH = "kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt"
CKPT_PATH = "pretrained_models/NegBioElectra/"

#load the data
df = pd.read_excel("data/synthesized_data.xlsx",index_col=0)
df = df.astype({'given_label':float})
df = df.sample(frac=1).reset_index(drop=True)
print(df.head(), df.shape)

#train-test split
test_size = math.ceil(len(df) * 0.1)
ds = []
for indx, row in df.iterrows():
    ds.append(InputExample(texts=[row['sent1'], row['sent2']], label=row['given_label']))

train_ds = ds[:-test_size]
test_ds = ds[-test_size:]
print(len(train_ds), len(test_ds))

train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_ds, name='test')

#load the model
base_model = models.Transformer(BASE_MODEL_PATH)

pooling_model = models.Pooling(base_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[base_model, pooling_model], device='cuda')

#if resuming training from trained-checkpoint:
#model = SentenceTransformer(CKPT_PATH)

#model training
train_loss = losses.CosineSimilarityLoss(model=model)
warmup_steps = math.ceil(len(train_dataloader) * EPOCHS  * 0.1)

#loggers
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=EPOCHS,
          evaluation_steps=math.ceil(len(test_ds)/BATCH_SIZE),
          warmup_steps=warmup_steps,
          output_path=CKPT_PATH)