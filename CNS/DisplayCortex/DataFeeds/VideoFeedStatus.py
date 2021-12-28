from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class VideoFeedStatus:
    def build(self, width, height):
        # create camera instance
        status = Label(text="",  color=(.5, .7, 1, 1))
        # create grid layout
        layout = GridLayout(rows=1, cols=1, size_hint_x=width, size_hint_y=height)
        # add widgets in layout
        layout.add_widget(status)
        return layout
