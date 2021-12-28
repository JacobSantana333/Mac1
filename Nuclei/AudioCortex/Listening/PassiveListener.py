import logging

import speech_recognition as sr
from Nuclei.AudioCortex.Speaking.Respond import speak
from Nuclei.LanguageCortex.Processing.Entities.EntityProcessor import EntityProcessor
from Nuclei.LanguageCortex.Processing.Unsupervised.BertProcessor import BertProcessor


class PassiveListener:

    def __init__(self, helper):

        self.helper = helper
        self.language_processor = BertProcessor(helper)
        self.entity_processor = EntityProcessor(helper)
        self.r = sr.Recognizer()
        self.power_down = False

    def callback(self, recognizer, audio):

        try:
            statement = recognizer.recognize_google(audio, language='en-IN')
            self.helper.log_and_print(logging.INFO, "GSR: " + statement)

            entities = self.entity_processor.predict_entities(statement)
            intent, response = self.language_processor.unsupervised_response(statement, entities)

            self.helper.log_and_print(logging.INFO, "Response:" + response)
            speak(response)

            self.set_listener_variables(intent)

        except sr.UnknownValueError:
            pass
            #print("Unusual noise")

    def begin_passive_listener(self):

        self.helper.log_and_print(logging.INFO, "Begin Passive Listener")
        return self.r.listen_in_background(sr.Microphone(), self.callback)

    def set_listener_variables(self, intent):

        if intent == "PowerDown":
            self.power_down = True
        elif intent == "RebootLanguage":
            self.language_processor = BertProcessor(self.helper)
        elif intent == "RebootEntities":
            self.entity_processor = EntityProcessor(self.helper)
