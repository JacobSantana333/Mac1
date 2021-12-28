import logging
import time
from Nuclei.LanguageCortex.Learning.Unsupervised.LearnBERT import *
from Nuclei.AudioCortex.Listening.PassiveListener import PassiveListener
from tools.Helpers import Helpers

helper = Helpers()
helper.log_and_print(logging.INFO, "Starting up")

passive_listener = PassiveListener(helper)
stop_passive_lister = passive_listener.begin_passive_listener()


Active = True
while True:
    if passive_listener.power_down:
        stop_passive_lister()
        break
    time.sleep(0.1)
