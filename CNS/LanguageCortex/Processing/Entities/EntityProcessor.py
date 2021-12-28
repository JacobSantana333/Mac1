import logging
from pathlib import Path
import spacy


class EntityProcessor():
    def __init__(self,helper):
        self.helper = helper

        self.model_dir = Path(r"C:\Users\jcsan\PycharmProjects\Mac1\CNS\LanguageCortex\Learning\Entities\Model")
        self.trained_nlp = spacy.load(self.model_dir)

    def predict_entities(self, input):

        doc = self.trained_nlp(input)
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = ent.text

        return entities


