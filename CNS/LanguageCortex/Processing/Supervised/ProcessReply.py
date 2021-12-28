import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

from CNS.LanguageCortex.Extensions.Extensions import utilize_extension, get_extension

lemmatizer = WordNetLemmatizer()

model = load_model('CNS/LanguageCortex/Learning/Supervised/chatbot_model.h5')
import json
import random


intents = json.loads(open('CNS\LanguageCortex\Learning\FunctionalIntents.json').read())
words = pickle.load(open('CNS/LanguageCortex/Learning/Supervised/words.pkl', 'rb'))
classes = pickle.load(open('CNS/LanguageCortex/Learning/Supervised/classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.40
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints):
    result = ""
    intent = ints[0]['intent']
    for i in intents['intents']:
        if i['intent'] == intent:
            extension_function, extension_responses = get_extension(i)
            if extension_function:
                result = utilize_extension(extension_function, extension_responses)
            else:
                result = random.choice(i['responses'])
            break
    return result


def supervised_response(msg):
    res = ""
    ints = predict_class(msg, model)
    if ints:
        res = get_response(ints)
    else:
        for i in intents['intents']:
            if i['intent'] == "UnknownPhrase":
                res = random.choice(i['responses'])
    return res
