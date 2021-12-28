import json
import logging
import re

import numpy as np
import torch
import random
from transformers import BertTokenizerFast

from Nuclei.LanguageCortex.Extensions.Extensions import get_extension, utilize_extension
from Nuclei.LanguageCortex.Learning.Unsupervised.LearnBERT import *
from tools.Helpers import Helpers
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F


class BertProcessor:
    def __init__(self, helper):
        self.helper = helper
        self.intents = json.loads(open(r"C:\Users\jcsan\PycharmProjects\Mac1\Nuclei\LanguageCortex\Learning\FunctionalIntents.json").read())
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda')

        templist = []
        for i in self.intents["intents"]:
            for pattern in i["patterns"]:
                templist.append([pattern, i["intent"]])
        self.df = pd.DataFrame(templist, columns=['text', 'intent'])

        # Converting the labels into encodings
        self.le = LabelEncoder()
        self.le.fit_transform(self.df['intent'])

        self.model = load_bert()
        self.model.eval()
        self.unsupervised_response("Hi", {})

    def predict_class(self, str):

        str = re.sub(r'[^a-zA-Z0-9 ]+', '', str)
        test_text = [str]

        tokens_test_data = tokenizer(
            test_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )


        test_seq = torch.tensor(tokens_test_data['input_ids'])
        test_mask = torch.tensor(tokens_test_data['attention_mask'])

        all_logits =[]
        preds = None
        with torch.no_grad():
            preds = self.model(test_seq.to(device), test_mask.to(device))
        all_logits.append(preds)
        all_logits = torch.cat(all_logits, dim=0)
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        intent = None
        if np.any(probs[0, :] > .85):
            preds = np.argmax(probs, axis=1)
            intent =self.le.inverse_transform(preds)[0]

        return intent

    def get_response(self, int, entities):
        result = ""
        for i in self.intents['intents']:
            if i['intent'] == int:
                extension_function, extension_responses = get_extension(i)
                if extension_function:
                    result = utilize_extension(extension_function, extension_responses, entities)
                else:
                    result = random.choice(i['responses'])
                break
        return result

    def unsupervised_response(self, msg, entities):
        res = ""
        int = self.predict_class(msg)
        if int:
            res = self.get_response(int, entities)
        else:
            for i in self.intents['intents']:
                if i['intent'] == "UnknownPhrase":
                    res = random.choice(i['responses'])

        return int, res


