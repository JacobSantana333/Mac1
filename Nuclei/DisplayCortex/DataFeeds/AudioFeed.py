from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class AudioFeed:

    def build(self, width, height):
        # ceate button
        lbl = Button(text="Audio feed", size_hint_y=.3)
        lbl.font_size = 11
        lbl.bind()
        lbl2 = Button(text="Audio Text", size_hint_y=.2)
        lbl2.font_size = 11
        lbl2.bind()
        lbl3 = Button(text="Audio feed", size_hint_y=.3)
        lbl3.font_size = 11
        lbl3.bind()
        lbl4 = Button(text="Audio Text", size_hint_y=.2)
        lbl4.font_size = 11
        lbl4.bind()

        # create grid layout
        layout = GridLayout(rows=4, cols=1, size_hint_x=width, size_hint_y=height)

        # add widgets in layout
        layout.add_widget(lbl)
        layout.add_widget(lbl2)
        layout.add_widget(lbl3)
        layout.add_widget(lbl4)
        return layout
