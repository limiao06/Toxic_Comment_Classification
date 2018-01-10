# Toxic_Comment_Classification

Codes for Kaggle Competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/

Data should in directory input/

Need DeepMoji installed: https://github.com/bfelbo/DeepMoji

## DeepMoji

Use DeepMoji to do the job.

Train script: scripts/train-chain-thaw-embedl2-acc-monitor.sh
Description: based on keras, chain-thaw mode, extend_embedding = 10000, maxlen=150, DeepMoji's tokenizer, early stop monitor = val acc
Results: 0.048

