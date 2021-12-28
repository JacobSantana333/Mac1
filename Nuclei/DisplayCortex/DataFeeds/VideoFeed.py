from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class VideoFeed:
    def build(self, width, height):
        # create camera instance
        cam = Camera()
        cam.allow_stretch = True
        cam.keep_ratio = True
        lbl = Button(text="Live feed", size_hint_y=.1)
        lbl.bind()

        # create grid layout
        layout = GridLayout(rows=2, cols=1, size_hint_x=width, size_hint_y=height)
        # add widgets in layout
        layout.add_widget(cam)
        layout.add_widget(lbl)
        return layout
