#### Setup Guide


Requirements


transformers
torchtext
seaborn
tqdm
flask



##### Step 1: Download DialoGPT fine-tuned model weights

Download the fine-tuned model weights from the URL https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl and save it in current(Emotion-Emphasis) directory. The filename of the downloaded file should be medium_ft.pkl





#####  Step 2: Save the model as PyTorch checkpoint

Run the create_checkpoint_dialogpt_ft_model.py file to save the model as PyTorch checkpoint. Please refer to https://huggingface.co/transformers/serialization.html#serialization-best-practices for more details.





##### Step 3: Train Discriminator head on SST dataset
