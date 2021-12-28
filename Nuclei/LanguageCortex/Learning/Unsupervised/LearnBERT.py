import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler
from transformers import AutoModel, BertTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchinfo import summary
import matplotlib
import keras


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# specify GPU
device = torch.device('cuda')


def train_bert():
    templist =[]
    data = json.loads(open(r"C:\Users\jcsan\PycharmProjects\pythonProject\Nuclei\LanguageCortex\Learning\FunctionalIntents.json").read())
    for i in data["intents"]:
        for pattern in i["patterns"]:
            templist.append([pattern, i["intent"]])
    df = pd.DataFrame(templist, columns=['text', 'intent'])
    print(df['intent'].value_counts())

    # encode the data
    le = LabelEncoder()
    df['intent'] = le.fit_transform(df['intent'])
    train_text, train_labels = df['text'], df['intent']


    # tokenize and encode sequences in the training set
    inputs = tokenizer(
        train_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # for train set
    train_seq = torch.tensor(inputs['input_ids'])
    train_mask = torch.tensor(inputs['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # DataLoader for train set
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)

    # freeze all the parameters. This will prevent updating of model weights during fine-tuning.
    for param in bert.parameters():
        param.requires_grad = False
    model = BERT_Arch(bert, len(np.unique(train_labels)))  # BERT_Model
    # push the model to GPU
    model = model.to(device)
    summary(model)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # compute the class weights
    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    print(class_wts)

    # convert class weights to tensor
    weights = torch.tensor(class_wts, dtype=torch.float)
    weights = weights.to(device)
    # loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    # number of training epochs
    epochs = 100
    # We can also use learning rate scheduler to achieve better results
    lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train(model, train_dataloader, cross_entropy, optimizer)

        # append training and validation loss
        train_losses.append(train_loss)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f'\nTraining Loss: {train_loss:.3f}')

    torch.save(model,
               r"C:\Users\jcsan\PycharmProjects\pythonProject\Nuclei\LanguageCortex\Learning\Unsupervised\BertModel.pt")


class BERT_Arch(nn.Module):
    def __init__(self, bert,outputlayers):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()
        # dense layer
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, outputlayers)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        # define the forward pass

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)

        # apply softmax activation
        x = self.softmax(x)
        return x


# function to train the model
def train(model,train_dataloader,cross_entropy,optimizer):
    model.train()
    total_loss = 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # clear calculated gradients
        optimizer.zero_grad()

        # We are not using learning rate scheduler as of now
        # lr_sch.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions
    return avg_loss, total_preds


def load_bert():
    return torch.load(r"C:\Users\jcsan\PycharmProjects\pythonProject\Nuclei\LanguageCortex\Learning\Unsupervised\BertModel.pt")

#train_bert()


