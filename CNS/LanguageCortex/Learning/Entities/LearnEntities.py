import json
import re
import string
from pathlib import Path
from random import shuffle

import spacy
from spacy.gold import GoldParse
from spacy.util import compounding, minibatch


class EntityTrainer():
    def __init__(self):

        self.model_dir = Path(r"C:\Users\jcsan\PycharmProjects\Mac1\CNS\LanguageCortex\Learning\Entities\Model")

    def train(self, model=None):
        intents = json.loads(open(
            r"C:\Users\jcsan\PycharmProjects\Mac1\CNS\LanguageCortex\Learning\FunctionalIntents.json").read())
        training_data_set = self.createDataset(intents)
        if model is not None:
            nlp = spacy.load(model)  # load existing spacy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank('en')  # create blank Language class
            print("Created blank 'en' model")

        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe('ner')

        for i in self.get_entity_labels(intents):
            ner.add_label(i)

        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.entity.create_optimizer()

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(50):
                shuffle(training_data_set)
                losses = {}
                batches = minibatch(training_data_set, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                               losses=losses)
                print('Losses', losses)

            if not self.model_dir.exists():
                self.model_dir.mkdir()
            nlp.meta['name'] = "Mac1"
            nlp.to_disk(self.model_dir)
            print("Saved model to", self.model_dir)

    def createDataset(self, intents):
        dataset = []
        intentCounter = 0
        for intent in intents['intents']:
            sentenceCounter = 0
            for sentence in intent['patterns']:
                sentence_entities = []

                if intent['entities']:
                    for entity in intent['entities'][sentenceCounter]:
                        start, end = self.get_entity_index_ranges(sentence, entity['rangeStart'], entity['rangeEnd'])
                        sentence_entities.append(tuple((start, end, entity['entity'])))
                    dataset.append((sentence, {"entities": sentence_entities}))
                    sentenceCounter = sentenceCounter + 1
            intentCounter = intentCounter + 1
        return dataset

    def get_entity_labels(self, intents):
        labels = []
        for intent in intents["intents"]:
            for entity in intent["entities"]:
                for en in entity:
                    if en["entity"] not in labels:
                        labels.append(en["entity"])
        return labels

    def get_entity_index_ranges(self, sentence, start, end):
        word = ""
        index = 0
        while index <= (end - start):
            if (index == end - start):
                word += sentence.split(" ")[start + index]
                index += 1
            else:
                word += sentence.split(" ")[start + index] + " "
                index += 1

        a = sentence.find(word)
        return (a, a + len(word))

#e = EntityTrainer()
#e.train()