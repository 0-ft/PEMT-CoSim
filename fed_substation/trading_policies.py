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
        self.ev_buy_threshold_price = float('inf')
        self.ev_sell_threshold_price = float('inf')

    def update_averages(self, current_time):
        if (current_time - self.auction.history.index.min()) < self.long_window:
            return False
        self.long_ma = self.auction.history["lmp_mean_since"].last(self.long_window).iloc[0]
        self.short_ma = self.auction.history["lmp_mean_since"].last(self.short_window).iloc[0]
        self.iqr = self.auction.history["iqr_since"].last(self.long_window).iloc[0]
        return True

    def formulate_ev_bids(self, current_time: datetime, ev_buy_range):
        if ev_buy_range[0] > 0:
            print("MUST BUY")
            return [["buyer", float('inf'), ev_buy_range[0]]]
        if ev_buy_range[1] < 0:
            print("MUST SELL")
            return [["seller", 0, ev_buy_range[1]]]

        if not self.update_averages(current_time):
            return []

        self.ev_buy_threshold_price = self.long_ma - self.iqr * self.buy_iqr_ratio
        self.ev_sell_threshold_price = max(self.ev_buy_threshold_price + 0.05 * self.iqr, self.short_ma + 0.2 * self.iqr)

        buy_bid = [["buyer", self.ev_buy_threshold_price, ev_buy_range[1]]] if ev_buy_range[1] > 0 else []
        sell_bid = [["seller", self.ev_sell_threshold_price, -ev_buy_range[0]]] if -ev_buy_range[0] > 0 else []
        return buy_bid + sell_bid

    def formulate_bids(self, house_name, current_time: datetime, ev_buy_range, pv_max_power):
        ev_bids = self.formulate_ev_bids(current_time, ev_buy_range) if ev_buy_range else []
        ev_sell_prices = [bid[1] for bid in ev_bids if bid[0] == "seller"]
        # ev_sell_price = min(ev_sell_prices) if ev_sell_prices else float('inf')
        pv_sell_price = min(self.auction.lmp * 0.9, self.ev_sell_threshold_price * 0.95)
        pv_bids = [[(house_name, "pv"), "seller", pv_sell_price, pv_max_power]] if pv_max_power > 0 else []
        return [[(house_name, "ev")] + bid for bid in ev_bids] + pv_bids
