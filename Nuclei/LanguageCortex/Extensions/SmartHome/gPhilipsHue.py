import random

import numba.cpython.setobj
from phue import Bridge


class gPhilipsHue:
    def __init__(self):

        self.b = Bridge('192.168.1.171')
        # If the app is not registered and the button is not pressed, press the button and call connect() (this only needs to be run a single time)
        # self.b.connect()

        # Get the bridge state (This returns the full dictionary that you can explore)
        self.b.get_api()

    def toggle_single_light(self, responses, entities):
        if entities["LEVEL"]:
            if entities["LIGHT"]:
                level = entities["LEVEL"].replace("%", "")
                light = ""
                for l in self.b.lights:
                    if l.name.lower() == entities["LIGHT"].lower():
                        light = l.name
            self.b.set_light(light, 'on', True)
            self.b.set_light(light, 'bri', int(254 * (int(level) / 100)))
        return random.choice(responses).replace("%%LIGHT%%", light).replace("%%LEVEL%%", level)

    def toggle_all_lights_level(self, responses, entities):
        if entities["LEVEL"]:
            level = entities["LEVEL"].replace("%", "")
            for light in self.b.get_light_objects(mode='list'):
                self.b.set_light(light.name, 'on', True)
                self.b.set_light(light.name, 'bri', int(254 * (int(level) / 100)))
        return random.choice(responses).replace("%%LEVEL%%", level)

    def toggle_all_lights_on(self, responses, entities):
        self.b.set_group('Lights', 'on', True)
        return random.choice(responses)

    def toggle_all_lights_off(self, responses, entities):
        self.b.set_group('Lights', 'on', False)
        return random.choice(responses)
