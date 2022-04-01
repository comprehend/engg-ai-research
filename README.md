# No means 'No'; a non improper embedding model with understanding of uncertainty context
This repo's branch is for official implementation of the paper "No means 'No'; a non improper embedding model with understanding of uncertainty context".

# DESCRIPTION
The medical data are complex in nature as terms that appear in records usually appear in different contexts among which lies negated and uncertain texts which usually have great importance. There have been many models which detect cue and scopes in these texts and also have achieved SOTA in this task. 
We too propose a super-tuning approach which enhances any natural language model's capabilities of understanding the negation & uncertainity context.
Along with this we provide a synthesized dataset developed using T5 & Peagsus paraphraser.We super-tuned the BioELECTRA model on negation & speculation task on BioScope & Sherlock datasets and named the new model as NegBioELECTRA. The super-tuning of BioELECTRA helped us to achieve SOTA on benchmark in negation,speculation cue and scope detection on BioScope and Sherlock datasets. Our model detects not only cues but also states if the cue is of negation or speculation. For a detailed description and experimental results, please refer to our paper __________________.

# Try the approach yourself

Clone the branch using
'git clone -b uncertainity-super-tuning https://github.com/comprehend/engg-ai-research.git'

Create a vitual env and anctivate it, and navigate into 'uncertainity_super_tuning' branch and install the required libraries using
'pip install -r requirements.txt'

