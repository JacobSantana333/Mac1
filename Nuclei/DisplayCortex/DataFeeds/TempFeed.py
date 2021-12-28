from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class TempFeed:

    def build(self, width, height):
        lbl = Button(text="Temporary", font_size=11)
        lbl.bind()
        # create grid layout
        layout = GridLayout(rows=1, cols=1, size_hint_x=width, size_hint_y=height)
        # add widgets in layout
        layout.add_widget(lbl)
        return layout
