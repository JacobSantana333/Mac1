
import pyttsx3


def speak(response):
    engine = pyttsx3.init()  # object creation

    """ RATE"""
    rate = engine.getProperty('rate')  # getting details of current speaking rate
    engine.setProperty('rate', 160)  # setting up new voice rate

    """VOLUME"""
    volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')  # getting details of current voice
    engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male

    engine.say(response)
    engine.runAndWait()