from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class VideoFeed:
    def build(self):
        cam = Camera()
        cam.allow_stretch = True
        cam.keep_ratio = True

        return cam