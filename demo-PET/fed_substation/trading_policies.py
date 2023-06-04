from datetime import timedelta, datetime
from math import floor

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import iqr


def moving_average(x, w, pad=0):
    mr = np.convolve(x, np.ones(w), 'valid') / w
    return np.pad(mr, (len(x) - len(mr), 0), mode="constant", constant_values=pad)


class LimitedCrossoverTrader:

    def __init__(self, auction, short_window: timedelta, long_window: timedelta):
        self.auction = auction
        self.short_window = short_window
        self.long_window = long_window
        self.movement_window = timedelta(minutes=20)
        self.short_ma = 0.0
        self.long_ma = 0.0
        # self.amplitude = 0.0
        self.iqr = 0.0
        self.buy_threshold = 0.3
        self.sell_threshold = 0.3
        self.should_buy = False
        self.should_sell = False
        self.should_trade = 0.0
        self.predicted_clearing_price = 0.0

    def predict_clearing_price(self, current_time):
        movement_dir = self.auction.history["clearing_price"].diff()[
            self.auction.history.index >= current_time - self.movement_window] \
            .mean()
        print(f"moving {movement_dir}")
        self.predicted_clearing_price = self.auction.history["clearing_price"].asof(current_time) + movement_dir
        return self.predicted_clearing_price

    def price_for_probability(self, current_time, probability, buying=True):
        predicted_clearing_price = self.predict_clearing_price(current_time)
        extra = 0.1 * ((probability - 0.5) + np.random.uniform(-1, 1) / 1000) * [-1, 1][buying]
        return predicted_clearing_price * (1 + extra)

    def __repr__(self):
        return f"TradingPolicy({str(self.__dict__)})"

    def trade(self, current_time: datetime, buy_range):
        # current_time = pd.to_datetime(current_time)

        # print(self.auction.history.index, current_time - self.short_window, self.auction.history.index <= current_time)

        # update moving averages
        self.long_ma = self.auction.history["average_since"].asof(current_time - self.long_window)
        self.short_ma = self.auction.history["average_since"].asof(current_time - self.short_window)
        # long_ma_mask = self.auction.history.index >= (current_time - self.long_window)
        # long_ma_window = self.auction.history["clearing_price"][long_ma_mask]
        # if len(long_ma_window) < 2:
        #     return 0, 0
        # self.long_ma = long_ma_window.mean()
        #
        # short_ma_mask = self.auction.history.index >= (current_time - self.short_window)
        # short_ma_window = self.auction.history["clearing_price"][short_ma_mask]
        # self.short_ma = short_ma_window.mean()
        # self.amplitude = long_ma_window.max() - long_ma_window.min()
        self.iqr = self.auction.history["iqr_since"].asof(current_time - self.long_window)
        if np.any(np.isnan([self.long_ma, self.short_ma, self.iqr])):
            return []

        # predict price
        # self.predicted_price = short_ma_window.iloc[-1] + short_ma_window.diff().iloc[-1] * 1
        # self.predicted_clearing_price = self.predict_clearing_price(current_time)
        # diff = self.short_ma - self.long_ma
        buy_threshold_price = self.long_ma - self.iqr * self.buy_threshold
        # self.should_buy = diff < buy_threshold_price
        sell_threshold_price = self.long_ma + self.iqr * self.sell_threshold
        # self.should_sell = diff > sell_threshold_price
        # self.should_trade = self.should_buy * buy_range[1] + self.should_sell * buy_range[0]
        # price = sell_threshold_price if self.should_sell else buy_threshold_price
        if buy_range[0] > 0:
            return ["buyer", float('inf'), buy_range[0]]
        if buy_range[1] < 0:
            print("MUST SELL")
            return ["seller", 0, buy_range[1]]
        return [
            ["buyer", buy_threshold_price, buy_range[1]],
            ["seller", sell_threshold_price, -buy_range[0]]
        ]

# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


# def rolling_amplitude(a, window, axis=1):
#     qq = np.array([[max(a[max(0, i - window):i + 1]), min(a[max(0, i - window):i + 1])] for i in range(len(a))])
#     return qq[:, 0] - qq[:, 1]


# day = 3600 * 24
# days = 8
# samples = 3000
# sperd = floor(samples / days)
#
# seconds = np.sort(np.random.uniform(0, day * days, samples))
# price = 20 * np.sin(seconds / day * 2 * np.pi) + \
#         7 * np.sin(seconds / day / 3 * 2 * np.pi) + \
#         np.random.uniform(-1, 1, samples) * 3 + \
#         np.interp(seconds, np.sort(np.random.uniform(0, day * days, 100)), np.random.random(100)) * 10 + \
#         40
#
#
# def trade(price, long_window=sperd * 2):
#     amplitude = rolling_amplitude(price, long_window)
#     ma_short, ma_long = moving_average(price, 50), moving_average(price, long_window)
#
#     ramp = 0.1
#
#     diff = ma_long - ma_short
#     # buy = ((np.exp(diff - 3) - np.exp(10-diff)))
#     profit = [0.0]
#     stored = [1000]
#     buy = diff > (amplitude / 3)
#     sell = diff < -(amplitude / 3)
#     for i, p in enumerate(price):
#         if buy[i] and stored[-1] < 2000:
#             profit.append(profit[-1] - p)
#             stored.append(stored[-1] + 1)
#         elif sell[i] and stored[-1] > 0:
#             profit.append(profit[-1] + p)
#             stored.append(stored[-1] - 1)
#         else:
#             profit.append(profit[-1])
#             stored.append(stored[-1])
#
#     # bought = np.piecewise(diff, [buy, sell], [0.01, -0.01]) * (np.arange(samples) > sperd * 2)
#     # profit = -(bought * price).cumsum()
#     return ma_short, ma_long, buy, sell, profit[1:], amplitude, stored[1:]
#
#
# mau = [0.0]
# for p in price:
#     # mau.append(mau[-1] * 0.99 + p * 0.01)
#     mau.append(mau[-1] * (1 - 1 / (sperd * 2)) + p / (sperd * 2))
#
# ma_short, ma_long, buy, sell, profit, amplitude, stored = trade(price)
#
# df = pd.DataFrame(np.array([seconds, price, ma_short, ma_long, buy, sell, profit, amplitude, stored, mau[1:]]).T,
#                   columns=["time", "price", "ma_short", "ma_long", "buy", "sell", "profit", "amplitude", "stored",
#                            "mau"])
#
# fig = px.line(df, x="time", y=["price", "ma_short", "ma_long", "buy", "sell", "profit", "amplitude", "stored", "mau"],
#               render_mode="svg")
#
# fig.write_html("trading_test.html")
