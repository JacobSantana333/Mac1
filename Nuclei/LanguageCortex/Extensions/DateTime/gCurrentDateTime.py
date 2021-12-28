
import os, datetime, json, random

import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder


class gCurrentDateTime:

    def __init__(self):
        self.geolocator = Nominatim(user_agent="Mac1_Geo_location")

    def get_time(self, responses,entities):
        return random.choice(responses).replace("%%TIME%%", datetime.datetime.now().strftime("%#I:%M %p"))

    def get_date(self, responses,entities):
        return random.choice(responses).replace("%%DATE%%", datetime.datetime.now().strftime("%B %#d"))

    def get_day_of_week(self, responses,entities):
        return random.choice(responses).replace("%%DAY%%", datetime.datetime.now().strftime("%A"))

    def get_time_in_location(self,responses,entities):
        if(entities):
            eloc = entities["LOCATION"]
            location = self.geolocator.geocode(eloc)
            obj = TimezoneFinder()
            result = obj.timezone_at(lng=location.longitude, lat=location.latitude)
            return datetime.datetime.now(pytz.timezone(result)).strftime("%#I:%M %p")

