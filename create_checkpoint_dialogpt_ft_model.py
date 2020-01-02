# -*- coding: utf-8 -*-

import os
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, WEIGHTS_NAME, CONFIG_NAME

# parameters
model_size = "medium"

# The fine-tuned DialoGPT models published by Microsoft on azure blob do not have a model config attached to them.
# Model config with vocab and merges is available in /configs folder at https://github.com/microsoft/DialoGPT.git.
# You can download the /configs folder from https://github.com/microsoft/DialoGPT.git and run the following code.
# gpt2_config= {'small': GPT2Config.from_json_file('DialoGPT/configs/117M/config.json'),
#              'medium': GPT2Config.from_json_file('DialoGPT/configs/345M/config.json'),
#              'large': GPT2Config.from_json_file('DialoGPT/configs/762M/config.json')}

# Alternatively the model config can also be manually set.
# These are the default model config for gpt-2 small, medium and large models.
gpt2_config = {'small': GPT2Config(), 'medium': GPT2Config(n_ctx=1024, n_embd=1024, n_layer=24, n_head=16),
               'large': GPT2Config(n_ctx=1024, n_embd=1280, n_layer=36, n_head=20)}

# load the gpt2 tokenizer. All three gpt2 models (small, medium, large) use the same vocabulary.
# A tokenizer is constructed from two files vocab.json and merges.txt.
# Both of these files are available in configs/117M/,  configs/345M/ and /configs/762M folders
# tokenizer = GPT2Tokenizer.from_pretrained('DialoGPT/configs/345M')

# Alternatively the following line of code will automatically download the vocab.json and merges.txt files
# and create a tokenizer from these files
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# create transformer model from the fine-tuned model weights
model = GPT2LMHeadModel(gpt2_config[model_size])
model.load_state_dict(torch.load(model_size + "_ft.pkl"), strict=False)

# The model.load_state_dict() function will show a warning as """_IncompatibleKeys(missing_keys=['lm_head.weight'], unexpected_keys=['lm_head.decoder.weight'])"""
# This is becasue the DialoGPT model was trained using huggingface's pytorch_pretrained_bert model, which is an older version of  huggingface transformers. The old pytorch_pretrained_bert used different attribute name for the model weights.
# Folloing line fixes this and assigns a correct attribute name
model.lm_head.weight.data = model.transformer.wte.weight.data

model_dir_name = 'DialoGPT-' + model_size


if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)

# This code is taken from huggingface transformers documentation at https://huggingface.co/transformers/serialization.html#serialization-best-practices
# This code saves the model and tokenizer into filesystem.
# The default name of model weights will be pytorch_model.bin, model config will be saved as config.json.
# The tokenizer will be saved as files vocab.json and merges.txt

output_dir = "./" + model_dir_name + "/"

# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)
