import random
import shutil

from CNS.LanguageCortex.Learning.Entities.LearnEntities import EntityTrainer
from CNS.LanguageCortex.Learning.Supervised.LearnSupervised import learn_language
from CNS.LanguageCortex.Learning.Unsupervised.LearnBERT import train_bert


class gLanguage:

    def __init__(self):
        pass

    def relearn_language(self, responses, entities):

        ModifiedIntetns = r"C:\Users\jcsan\PycharmProjects\Mac1\CNS\LanguageCortex\Learning\ModifiableIntents.json"
        FunctionalIntents=r"C:\Users\jcsan\PycharmProjects\Mac1\CNS\LanguageCortex\Learning\FunctionalIntents.json"

        shutil.copyfile(ModifiedIntetns, FunctionalIntents)

        if entities["LANGUAGE_PROCESSOR"] == "supervised":
            learn_language()
        elif entities["LANGUAGE_PROCESSOR"] == "unsupervised":
            train_bert()
        return random.choice(responses)

    def relearn_Entities(self, responses, entities):
        e = EntityTrainer()
        e.train()
        return random.choice(responses)
