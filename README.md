# CONFIT: Toward Faithful Dialogue Summarization with Linguistically-Informed Contrastive Fine-tuning

We propose a training strategy that improves the factual consistency and overall quality of summaries via a novel contrastive fine-tuning, called ConFiT. Based on our linguistically-informed typology of errors, we design different modular objectives that each target a specific type.

The code is based on huggaface's [transformers](https://github.com/huggingface/transformers). Thanks to them! 

## Test

```
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
model = PegasusForConditionalGeneration.from_pretrained('FROM_OUR_PRETRAINED_MODELS')
device = "cuda:0"
model = model.to(device)
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
ARTICLE_TO_SUMMARIZE = ('"#ENTER")
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
print (inputs['input_ids'].shape)
```


## Annotation

[Google Drive](https://drive.google.com/drive/folders/1-IkrMDvDoBaEq9Y_ghSyxaAQIQRNCUbu?usp=sharing) is the unblinded annotation files that I used to compute the distributions in Tables 3 and 4. 

And see analysis.ipynb: the script to calculate the score.

## Generate Summary

```
import torch
input_ids = torch.load('./input_ids.pt')
decoder_input_ids = torch.load('./decoder_input_ids.pt')
attention_mask = torch.load('./attention_mask.pt')
print (input_ids.shape)
print (decoder_input_ids.shape)
print (attention_mask.shape)
output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
```

## Released Model Checkpoints
We have released the following checkpoints for our pre-trained models:

* **T5-CONFIT**: [Google Drive](https://drive.google.com/drive/folders/1pbqfHxs3oa38lLJw7j1mWDtpaWYfL2UD?usp=sharing)
* **Pegasus-CONFIT**: [Google Drive](https://drive.google.com/drive/folders/1uikCqSNMEVK2vKOfShoC6FmU-fH5cGMh?usp=sharing)
* **BART-CONFIT**: [Google Drive](https://drive.google.com/drive/folders/1pbqfHxs3oa38lLJw7j1mWDtpaWYfL2UD?usp=sharing)

## Citation
```
@article{tang2021confit,
  title={CONFIT: Toward Faithful Dialogue Summarization with Linguistically-Informed Contrastive Fine-tuning},
  author={Tang, Xiangru and Nair, Arjun and Wang, Borui and Wang, Bingyao and Desai, Jai and Wade, Aaron and Li, Haoran and Celikyilmaz, Asli and Mehdad, Yashar and Radev, Dragomir},
  journal={arXiv preprint arXiv:2112.08713},
  year={2021}
}

```
