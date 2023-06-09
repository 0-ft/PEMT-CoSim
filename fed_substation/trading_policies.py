from datetime import timedelta, datetime


class BoundedCrossoverTrader:

    def __init__(self, auction, short_window: timedelta, long_window: timedelta, buy_iqr_ratio, sell_iqr_ratio):
        self.auction = auction
        self.short_window = short_window
        self.long_window = long_window
        self.short_ma = 0.0
        self.long_ma = 0.0
        self.iqr = 0.0
        self.buy_iqr_ratio = buy_iqr_ratio
        self.sell_iqr_ratio = sell_iqr_ratio
        self.buy_threshold_price = float('inf')
        self.sell_threshold_price = float('-inf')

    def trade(self, current_time: datetime, buy_range):
        # update moving averages
        self.long_ma = self.auction.history["average_since"].last(self.long_window).iloc[0]
        self.short_ma = self.auction.history["average_since"].last(self.short_window).iloc[0]
        self.iqr = self.auction.history["iqr_since"].last(self.long_window).iloc[0]

        self.buy_threshold_price = self.long_ma - self.iqr * self.buy_iqr_ratio
        self.sell_threshold_price = self.long_ma + self.iqr * self.sell_iqr_ratio

        if buy_range[0] > 0:
            return [["buyer", float('inf'), buy_range[0]]]
        if buy_range[1] < 0:
            print("MUST SELL")
            return [["seller", 0, buy_range[1]]]
        return [
            ["buyer", self.buy_threshold_price, buy_range[1]],
            ["seller", self.sell_threshold_price, -buy_range[0]]
        ]
