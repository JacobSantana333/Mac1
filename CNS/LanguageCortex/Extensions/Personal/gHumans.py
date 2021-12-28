import os, time, json, hashlib, hmac, random

from datetime import datetime


class gHumans():

    def __init__(self):
        pass


    def get_current_human(self, responses,entities):

        #retrieve from database
        name = "Jacob"
        return random.choice(responses).replace("%%PERSON%%", name)


    def update_human(self, responses,entities):

        name =""
        if(entities):
            name = entities["PERSON"]
        return random.choice(responses).replace("%%PERSON%%", name)
