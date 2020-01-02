# -*- coding: utf-8 -*-




from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from pplm_interactive import generate_text_pplm
import torch
import numpy as np
import json
from typing import List, Optional, Tuple, Union
from pplm_classification_head import ClassificationHead
from transformers.file_utils import cached_path
import time
import os

#This is done to fix an issue on mac os with xgboost and matplotlib. Please comment this line if using in any other OS
os.environ['KMP_DUPLICATE_LIB_OK']='True'


model_size = "medium"
pretrained_model="DialoGPT-"+model_size
discrim_meta_file="generic_classifier_head_meta.json"
discrim_weights="generic_classifier_head.pt"
class_label='angry'
num_samples=5



PPLM_DISCRIM = 2
loss_type = PPLM_DISCRIM


def get_classifier(
        discrim_meta: Optional[dict],
        class_label: Union[str, int],
        device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if discrim_meta is None:
        return None, None

    params = discrim_meta
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]

    else:
        label_id = params["default_class"]

    return classifier, label_id

# set Random seed
seed=0
torch.manual_seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained model
model = GPT2LMHeadModel.from_pretrained(pretrained_model,output_hidden_states=True)

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

model.to(device)
model.eval()

# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False

with open(discrim_meta_file, 'r') as f:
    discrim_meta = json.load(f)
discrim_meta['path'] = discrim_weights

assert discrim_meta['pretrained_model']==pretrained_model


classifier, class_id = get_classifier(
    discrim_meta,
    class_label,
    device
)

pert_gen_tok_texts = []
discrim_losses = []
losses_in_time = []

output_so_far=None

while True:
    cond_text=input("user: ")
    if(cond_text)=="quit:":
        break
    context = tokenizer.encode(tokenizer.bos_token + cond_text +tokenizer.bos_token,add_special_tokens=False)
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = (
            context_t if output_so_far is None
            else torch.cat((output_so_far, context_t), dim=1)
        )
    min_discrim_loss=np.infty
    selected_pert_gen_tok_text=None
    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            output_so_far=output_so_far,
            device=device,
            perturb=True,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=200,
            stepsize=0.05,
            temperature=1.0,
            top_k=10,
            sample=True,
            num_iterations=2,
            grad_length=10000,
            horizon_length=1,
            window_length=0,
            decay=False,
            gamma=1,
            gm_scale=0.8,
            kl_scale=0.01
        )

        dl=discrim_loss.data.cpu().numpy().tolist()
        #print('resp: {} , loss: {}'.format(tokenizer.decode(pert_gen_tok_text.tolist()[0]),dl))
        if(dl<min_discrim_loss):
            min_discrim_loss=dl
            selected_pert_gen_tok_text=pert_gen_tok_text
    output_so_far = (
            selected_pert_gen_tok_text if output_so_far is None
            else torch.cat((output_so_far, selected_pert_gen_tok_text), dim=1)
    )
    if(output_so_far.shape[1]>50):
        index=(output_so_far.squeeze(dim=0)==50256).nonzero()[-3].cpu().numpy()[0]
        output_so_far=output_so_far[:,index:-1]
    #if device == 'cuda':
    #    torch.cuda.empty_cache()
    pert_gen_tok_texts.append(selected_pert_gen_tok_text)
    if classifier is not None:
        discrim_losses.append(min_discrim_loss)
    losses_in_time.append(loss_in_time)
    bot = tokenizer.decode(selected_pert_gen_tok_text.tolist()[0])
    print("bot: ",bot)
