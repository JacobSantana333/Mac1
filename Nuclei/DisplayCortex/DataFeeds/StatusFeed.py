from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class StatusFeed:

    def build(self, width, height):
        # ceate button
        lbl = Button(text="status feed")
        lbl.font_size = 11
        lbl.bind()
        # create grid layout
        layout = GridLayout(rows=1, cols=1, size_hint_x=width, size_hint_y=height)
        # add widgets in layout
        layout.add_widget(lbl)
        return layout
