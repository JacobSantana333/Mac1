import random

import geocoder
from bs4 import BeautifulSoup #webscraping
import requests

class gWeather:

    def __init__(self):
        pass

    def get_current_weather(self, responses, entities):

            g = geocoder.ip('me')
            loaction = g.current_result.json["city"] + "+" + g.current_result.json["state"]

            # creating url and requests instance
            url ="https://www.google.com/search?q=weather+"+loaction
            html = requests.get(url).content
            # getting raw data
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
            str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text

            # formatting data
            data = str.split('\n')
            time = data[0]
            sky = data[1]
            return random.choice(responses).replace("%%TEMPERATURE%%", temp).replace("%%SKY%%", sky)
    def get_current_weather_of_location(self, responses, entities):

            resp =""
            if entities["LOCATION"]:
                g = geocoder.ip('me')
                loaction = entities["LOCATION"].replace(" ", "+")

                # creating url and requests instance
                url ="https://www.google.com/search?q=weather+"+loaction
                html = requests.get(url).content
                # getting raw data
                soup = BeautifulSoup(html, 'html.parser')
                if soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}):
                    temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
                    str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text

                    # formatting data
                    data = str.split('\n')
                    time = data[0]
                    sky = data[1]
                    resp = random.choice(responses).replace("%%TEMPERATURE%%", temp).replace("%%SKY%%", sky).replace("%%LOCATION%%",entities["LOCATION"])
                else:
                    resp = "The temperature was not pulled in request"
            else:
                resp = "The location was not received correctly"
            return resp
