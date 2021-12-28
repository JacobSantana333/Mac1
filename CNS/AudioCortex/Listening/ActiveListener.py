import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

    with open('recording.wav', 'wb') as f:
        f.write(audio.get_wav_data())
try:
    print("GSR:" + r.recognize_google(audio))
except sr.UnknownValueError:
    print("GSR could not understand audio")
except sr.RequestError as e:
    print("GSR Services Down; {0}".format(e))


