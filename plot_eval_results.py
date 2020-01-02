import os
import json
import math
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data as data
from tqdm import tqdm, trange
from transformers.file_utils import cached_path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pplm_classification_head import ClassificationHead
from typing import List, Optional, Tuple, Union
import numpy as np


#This is done to fix an issue on mac os with xgboost and matplotlib. Please comment this line if using in any other OS
os.environ['KMP_DUPLICATE_LIB_OK']='True'



model_size = "medium"
model_dir_name="DialoGPT-"+model_size
discrim_meta_file="generic_classifier_head_meta.json"
discrim_weights="generic_classifier_head.pt"
eval_dataset_type="preference"
eval_response_files={"preference": "response-preference-emotion.json", "opinion": "response-opinion-emotion.json"}

idx2class = ["anger", "joy","sadness"]
class2idx = {c: i for i, c in enumerate(idx2class)}

with open(eval_response_files[eval_dataset_type], 'r') as f:
  responses=json.load(f)


torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size,
            pretrained_model="gpt2-medium",
            cached_mode=False,
            device='cpu',
            classifier_head=None
    ):
        super(Discriminator, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = classifier_head
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        hidden, _ = self.encoder.transformer(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs



def get_classifier(
        discrim_meta: Optional[dict],
        device: str
) -> Optional[ClassificationHead]:
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

    return classifier





class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch

def train_epoch(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    train_loss = 0
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        # sum up batch loss
        train_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        #if batch_idx % log_interval == 0:
        #    print(
        #        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #            epoch + 1,
        #            samples_so_far, len(data_loader.dataset),
        #            100 * samples_so_far / len(data_loader.dataset), loss.item()
        #        )
        #    )
    train_loss /= (batch_idx+1)
    return train_loss


def evaluate_performance(data_loader, discriminator, device='cpu'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)
        )
    )
    return test_loss, 100. * correct / len(data_loader.dataset)


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader

#parameters
dataset='SST'
#dataset_fp=descriminator_dataset_file
pretrained_model=model_dir_name
batch_size=64
log_interval=10000
save_model=True
cached=True
no_cuda=False


device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"



with open(discrim_meta_file, 'r') as f:
    discrim_meta = json.load(f)
discrim_meta['path'] = discrim_weights

discriminator = Discriminator(
    class_size=len(idx2class),
    pretrained_model=pretrained_model,
    cached_mode=cached,
    device=device,
    classifier_head=get_classifier(discrim_meta,device)
).to(device)




# Evaluate all samples
response_x=[]
response_y=[]
for d in responses:
  for class_label in idx2class:
    for x in d[class_label]['utterances']:
      response_x.append(torch.tensor([50256] + discriminator.tokenizer.encode(x)  , device=device,dtype=torch.long))
      response_y.append(class2idx[class_label])


response_dataset = Dataset(response_x, response_y)
response_loader = get_cached_data_loader(
        response_dataset, batch_size, discriminator, device=device
    )


discriminator.eval()
response_loss = 0
correct = 0
target_ts=[]
pred_ts=[]

with torch.no_grad():
    for input_t, target_t in response_loader:
        input_t, target_t = input_t.to(device), target_t.to(device)
        output_t = discriminator(input_t)
        target_ts.extend(list(target_t.cpu().numpy()))
        # sum up batch loss
        response_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
        # get the index of the max log-probability
        pred_t = output_t.argmax(dim=1, keepdim=True)
        pred_ts.extend(list(pred_t.view_as(target_t).cpu().numpy()))
        correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

response_loss /= len(response_loader.dataset)

print(
    "Performance on response set: "
    "Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)".format(
        response_loss, correct, len(response_loader.dataset),
        100. * correct / len(response_loader.dataset)
    )
)



#Plotting the classifiaction matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns
conf_mat = confusion_matrix(target_ts, pred_ts)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=idx2class, yticklabels=idx2class, cmap=sns.color_palette("Blues"))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# Evaluating after greedy select

response_x=[]
response_y=[]
for d in responses:
  for class_label in idx2class:
    _, idx = min((val, idx) for (idx, val) in enumerate(d[class_label]['losses']))
    x=d[class_label]['utterances'][idx]
    response_x.append(torch.tensor([50256] + discriminator.tokenizer.encode(x)  , device=device,dtype=torch.long))
    response_y.append(class2idx[class_label])


response_dataset = Dataset(response_x, response_y)
response_loader = get_cached_data_loader(
        response_dataset, batch_size, discriminator, device=device
    )


discriminator.eval()
response_loss = 0
correct = 0
target_ts=[]
pred_ts=[]

with torch.no_grad():
    for input_t, target_t in response_loader:
        input_t, target_t = input_t.to(device), target_t.to(device)
        output_t = discriminator(input_t)
        target_ts.extend(list(target_t.cpu().numpy()))
        # sum up batch loss
        response_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
        # get the index of the max log-probability
        pred_t = output_t.argmax(dim=1, keepdim=True)
        pred_ts.extend(list(pred_t.view_as(target_t).cpu().numpy()))
        correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

response_loss /= len(response_loader.dataset)

print(
    "Performance on response set: "
    "Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)".format(
        response_loss, correct, len(response_loader.dataset),
        100. * correct / len(response_loader.dataset)
    )
)



#Plotting the classifiaction matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns
conf_mat = confusion_matrix(target_ts, pred_ts)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=idx2class, yticklabels=idx2class, cmap=sns.color_palette("Blues"))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
