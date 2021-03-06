from kivy.app import App

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from CNS.DisplayCortex.DataFeeds.AudioFeed import AudioFeed
from CNS.DisplayCortex.DataFeeds.StatusFeed import StatusFeed
from CNS.DisplayCortex.DataFeeds.TempFeed import TempFeed
from CNS.DisplayCortex.DataFeeds.TopRowFeed import TopRowFeed
from CNS.DisplayCortex.DataFeeds.VideoFeed import VideoFeed
from CNS.DisplayCortex.DataFeeds.VideoFeedStatus import VideoFeedStatus

#Window.size = (1400, 1000)


class display(App):
    def build(self):
        main_layout = FloatLayout()
        main_layout.add_widget(VideoFeed().build())


        info_grid = GridLayout(rows=3, cols=3)

        top_row_height = .075
        middle_row_height = .75
        bottom_row_height = .175

        outside_width = .25
        inside_width = .75

        # Top Row feeds
        top_temp_1 = TopRowFeed().build(outside_width, top_row_height)
        top_temp_2 = TopRowFeed().build(inside_width, top_row_height)
        top_temp_3 = TopRowFeed().build(outside_width, top_row_height)

        info_grid.add_widget(top_temp_1)
        info_grid.add_widget(top_temp_2)
        info_grid.add_widget(top_temp_3)

        # Middle Row Feeds
        status_feed = StatusFeed().build(outside_width, middle_row_height)
        video_feed = VideoFeedStatus().build(inside_width, middle_row_height)
        Mid_temp_3 = StatusFeed().build(outside_width, middle_row_height)

        info_grid.add_widget(status_feed)
        info_grid.add_widget(video_feed)
        info_grid.add_widget(Mid_temp_3)

        # Bottom Row Feeds
        Bottom_temp_1 = TempFeed().build(outside_width, bottom_row_height)
        Audio_feed = AudioFeed().build(inside_width, bottom_row_height)
        Bottom_temp_3 = TempFeed().build(outside_width, bottom_row_height)

        info_grid.add_widget(Bottom_temp_1)
        info_grid.add_widget(Audio_feed)
        info_grid.add_widget(Bottom_temp_3)


        main_layout.add_widget(info_grid)
        return main_layout


if __name__ == '__main__':
    # run app
    display().run()
