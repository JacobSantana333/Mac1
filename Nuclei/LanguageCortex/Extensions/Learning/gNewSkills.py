
import os, datetime, json, random


class gNewSkills:

    def __init__(self):
        pass

    def add_intent(self, responses,entities):
        # intent = intent
        # pattern = pattern
        # responses = responses
        # new_intent = {"intent": intent, "patterns": pattern, "responses": responses}
        # with open('Nuclei/LanguageCortex/Learning/Uintents.json', 'rw') as intent_file:
        #     data = json.loads(intent_file)
        #     data.update(new_intent)
        #     intent_file.seek(0)
        #     json.dump(data, intent_file)

        return random.choice(responses)
