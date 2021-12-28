import os, time, json

from datetime import datetime
import logging


class Helpers:

    def __init__(self):
        self.path = os.path.join(r"Logs", datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S') + ".txt")

        logging.basicConfig(filename=self.path, format='%(asctime)s %(message)s', filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)


    def timerStart(self):

        return str(datetime.now()), time.time()

    def timerEnd(self, start):

        return time.time(), (time.time() - start), str(datetime.now())

    def load_json(self, path):
        with open(path) as jsonData:
            data = json.load(jsonData)
        return data

    def log(self, level, message):
        if level == logging.INFO:
            self.logger.log(logging.INFO, message)
        elif level == logging.ERROR:
            self.logger.log(logging.ERROR, message)
        elif level == logging.CRITICAL:
            self.logger.log((logging.ERROR, message))

    def log_and_print(self, level, message):
        if level == logging.INFO:
            self.logger.log(logging.INFO, message)
        elif level == logging.ERROR:
            self.logger.log(logging.ERROR, message)
        elif level == logging.CRITICAL:
            self.logger.log((logging.ERROR, message))

        print(message)



