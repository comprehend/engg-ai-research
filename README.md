# No means 'No'; a non improper embedding model with understanding of uncertainty context
This repo's branch is for official implementation of the paper "No means 'No'; a non improper embedding model with understanding of uncertainty context".

# DESCRIPTION
The medical data are complex in nature as terms that appear in records usually appear in different contexts among which lies negated and uncertain texts which usually have great importance. There have been many models which detect cue and scopes in these texts and also have achieved SOTA in this task. 
We too propose a super-tuning approach which enhances any natural language model's capabilities of understanding the negation & uncertainity context.
Along with this we provide a synthesized dataset developed using T5 & Peagsus paraphraser.We super-tuned the BioELECTRA model on negation & speculation task on BioScope & Sherlock datasets and named the new model as NegBioELECTRA. The super-tuning of BioELECTRA helped us to achieve SOTA on benchmark in negation,speculation cue and scope detection on BioScope and Sherlock datasets. Our model detects not only cues but also states if the cue is of negation or speculation. For a detailed description and experimental results, please refer to our paper __________________.

# Try the approach yourself

Clone the branch using
`git clone -b uncertainity-super-tuning https://github.com/comprehend/engg-ai-research.git`

Create a vitual env and anctivate it, and navigate into `uncertainity_super_tuning` branch and install the required libraries using
`pip install -r requirements.txt`

## Super-tuning
Super-tuning approach helps models understand negated and uncertainity context better in a given sentence. We propose that all models should go through this process before finetuning the model on any domain specific task. In our research paper _______________ we have compared the results of super-tuned vs non-super-tuned models sentence embeddings on negation context and it is clear that the super-tuning helps the model embeddings to be more generalized in terms of uncertainity context. Below is the depiction of comparision of super-tuned vs non-supertuned biomodels

![Alt text](relative/path/to/img.jpg?raw=true "Title")

The supertuning approaches uses SBERT architecture, and we have showcased BioELECTRA in code but you can use any model that is available in hugging face by just changing the hugging face architecture url at model configuration in `super_tuning.py` inside `uncertainity_super_tuning` folder. Also feel free to tweak the hyperparameters epochs and batch_size. The dataset used is a synthesized dataset of Bioscope Abstracts data.

```python
#set model-configuration details
EPOCHS = 15
BATCH_SIZE = 16
BASE_MODEL_PATH = "kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt"
CKPT_PATH = "pretrained_models/NegBioElectra/"
```

## Fine Tuning
We have given ipynb files for Bioscope, Sherlock cue and scope detection, for which you can use your own supertuned models by description above or feel free to use our hosted models at Huggingface 
 * NegBioELECTRA (yet to be hosted)
 * NegBioBERT (yet to be hosted)
 * NegPubMedBERT (yet to be hosted)
To tweak the hyperparams and train your own model use the variables under MODEL CONFIGURATION DETAILS, the following is the example:

```python
BASE_PATH = "pretrained_models/"
MODEL_PATH = BASE_PATH + "NegBioElectra/"
LR_RATE = 3e-5
CKPT_PATH = BASE_PATH + "bioscope_models/NegBioElectra_bioscope_scope_model"
EPOCHS = 15
```
Additionaly we provide `Evaluation_Bioscope_Scope_leaderboard.ipynb` file for testing the model's performance on Bioscope and `finetune_mednli.py` which gives a basic structure of how the model can be fine tuned for any other fine tuning tasks.

## Citing Information
yet to be updated!!
