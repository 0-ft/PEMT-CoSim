import pickle
from random import randint
from time import time as millis
from functools import reduce

import numpy as np
import pandas
from plotly.subplots import make_subplots


def key_collection(c, c_keys, default=None):
    vals = c.values() if isinstance(c, dict) else c
    nn = lambda x, d: d if x is None else x

    if c_keys[0] == "values":
        return [deep_get(x, c_keys[1:]) for x in vals]
    elif c_keys[0] == "sum":
        return sum(deep_get(x, c_keys[1:]) or 0 for x in vals)
    elif c_keys[0] == "mean":
        return np.mean([deep_get(x, c_keys[1:]) or 0 for x in vals])
    elif c_keys[0] == "max":
        return np.max([nn(deep_get(x, c_keys[1:]), float('inf')) for x in vals])
    elif c_keys[0] == "min":
        return np.min([nn(deep_get(x, c_keys[1:]), float('-inf')) or 0 for x in vals])
    else:
        return deep_get(c.get(c_keys[0], default) if isinstance(c, dict) else c[int(c_keys[0])], c_keys[1:])


def deep_get(target, keys, default=None):
    # print("invoke", target, keys)
    keys = keys.split(".") if isinstance(keys, str) else keys
    if target is None:
        return default
    if len(keys) == 0:
        return target

    if isinstance(target, dict) or isinstance(target, list):
        return key_collection(target, keys, default)

    return deep_get(getattr(target, keys[0], default), keys[1:])


class HistoryRecorder:
    def __init__(self, target, keys: list[str] | list[list[str]]):
        self.target = target
        self.keys = keys.split(".") if isinstance(keys, str) else keys
        self.times = []
        self.history = []

    def get_state(self):
        state = []
        for key in self.keys:
            try:
                state.append(deep_get(self.target, key, None))
            except Exception as e:
                raise Exception(f"couldn't get key {key}, {e}")
        return state

    def record(self, time):
        self.times.append(time)
        # self.history.append([deep_get(self.target, key, None) for key in self.keys])
        self.history.append(self.get_state())

    def df(self):
        return pandas.DataFrame(self.history, index=self.times, columns=self.keys)

    def save(self, path):
        self.df().to_pickle(path)


class SubstationRecorder:
    def __init__(self, vpp, houses, auction):
        self.vpp_recorder = HistoryRecorder(vpp, [
            "vpp_load_p",
            "vpp_load",
            "balance_signal",
            "weather_temp"
        ])
        self.house_recorder = HistoryRecorder(houses, list(set([x for t in [
            "mean.hvac.air_temp",
            "min.hvac.air_temp",
            "max.hvac.air_temp",
            "sum.hvac.hvac_load",
            "sum.hvac.hvac_on",
            "mean.hvac.set_point",
            "mean.hvac.base_point",
            # "sum.battery.power",
            # "sum.battery.soc",
            "values.ev.location",
            "sum.ev.stored_energy",
            "sum.ev.soc",
            "sum.ev.desired_charge_rate",
            "sum.ev.load",
            "sum.pv.solar_power",
            # "values.pv.solar_DC_V_out",
            # "values.pv.solar_DC_I_out",
            "sum.unresponsive_load",
            "sum.total_house_load"
        ] for x in [t, ".".join(["F0_house_A0"] + t.split(".")[1:])]])))

        self.bid_recorder = HistoryRecorder(houses, [
            "values.bid.values"
        ])

        self.auction_recorder = HistoryRecorder(auction, [
            "clearing_price",
            "clearing_type",
            "consumerSurplus",
            "averageConsumerSurplus",
            "supplierSurplus",
            "num_buyers",
            "num_sellers",
            "num_nontcp",
            "lmp"
        ])

    def record_houses(self, time):
        self.house_recorder.record(time)

    def record_bids(self, time):
        return
        self.bid_recorder.record(time)

    def record_auction(self, time):
        self.auction_recorder.record(time)

    def record_vpp(self, time):
        self.vpp_recorder.record(time)

    def history(self):
        return {
            "houses": self.house_recorder.df(),
            "bids": self.bid_recorder.df(),
            "auction": self.auction_recorder.df(),
            "vpp": self.vpp_recorder.df()
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history(), f)

    @staticmethod
    def make_figure(h: dict, path):
        houses = h["houses"]
        bids = h["bids"]
        auction = h["auction"]
        vpp = h["vpp"]
        s = millis()
        fig = make_subplots(rows=4, cols=1,
                            specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]])
        # specs=[[{}, {}], [{}, {}], [{"secondary_y": True}, {}], [{"secondary_y": True}, {}]])
        # fig.update_layout(width=2800, height=800)
        fig.add_traces([
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["mean.hvac.air_temp"],
                "name": "Mean House Air Temperature",
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["max.hvac.air_temp"],
                "name": "Max House Air Temperature",
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["min.hvac.air_temp"],
                "name": "Min House Air Temperature",
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["mean.hvac.set_point"],
                "name": "Mean HVAC Set Point",
            },
            {
                "type": "scatter",
                "x": vpp.index,
                "y": vpp["weather_temp"],
                "name": "Weather Temperature",
            },
        ], rows=1, cols=1)

        fig.update_yaxes(title_text="Temperature", row=1, col=1)

        fig.add_traces([
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.pv.solar_power"] * -1,
                "name": "Total Solar PV Power Generated",
                "stackgroup": "house_load"
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.ev.load"],
                "name": "Total EV Load",
                "stackgroup": "house_load"
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.unresponsive_load"],
                "name": "Total Unresponsive Load",
                "stackgroup": "house_load"
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.hvac.hvac_load"],
                "name": "Total HVAC Load",
                "stackgroup": "house_load"
            },
            {
                "type": "scatter",
                "x": vpp.index,
                "y": abs(vpp["vpp_load"]),
                "name": "Total VPP Load",
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.total_house_load"],
                "name": "Total House Load",
            },
        ], rows=2, cols=1)

        fig.update_yaxes(title_text="Load", row=2, col=1)

        fig.add_trace(
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.ev.stored_energy"],
                "name": "Total EV Energy Stored",
            }, row=3, col=1)
        fig.add_trace(
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.ev.desired_charge_rate"],
                "name": "Total Specified Charge Rate",
            }, row=3, col=1, secondary_y=True)

        fig.update_yaxes(title_text="Energy", row=3, col=1, range=[0, max(houses["sum.ev.stored_energy"])])
        fig.update_yaxes(title_text="Power", row=3, col=1, secondary_y=True,
                         range=[min(houses["sum.ev.desired_charge_rate"]), max(houses["sum.ev.desired_charge_rate"])])

        fig.add_trace(
            {
                "type": "scatter",
                "x": auction.index,
                "y": auction["clearing_price"],
                "name": "Cleared Price",
            }, row=4, col=1, secondary_y=True)
        fig.add_traces([
            {
                "type": "scatter",
                "x": auction.index,
                "y": auction[f"num_{key}"],
                "name": name,
            } for key, name in [("buyers", "Buyers"), ("sellers", "Sellers"), ("nontcp", "Non-participants")]
        ], rows=4, cols=1)
        fig.add_trace(
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["sum.hvac.hvac_on"],
                "name": "HVAC On",
            }, row=4, col=1)

        fig.update_yaxes(title_text="Count", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=4, col=1, secondary_y=True)

        fig.write_image(f"{path}.svg")
        fig.write_html(f"{path}.html")
        print(f"wrote figure in {(millis() - s) * 1000:3f}ms")

    @staticmethod
    def make_figure_solo(h: dict, path):
        houses = h["houses"]
        bids = h["bids"]
        auction = h["auction"]
        vpp = h["vpp"]
        s = millis()
        fig = make_subplots(rows=4, cols=1,
                            specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]])
        # specs=[[{}, {}], [{}, {}], [{"secondary_y": True}, {}], [{"secondary_y": True}, {}]])
        # fig.update_layout(width=2800, height=800)
        fig.add_traces([
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.hvac.air_temp"],
                "name": "House Air Temperature",
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.hvac.set_point"],
                "name": "HVAC Set Point",
            },
        ], rows=1, cols=1)

        fig.update_yaxes(title_text="Temperature", row=1, col=1)

        fig.add_traces([
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.pv.solar_power"] * -1,
                "name": "PV Load",
                "line": {"width": 0},
                "stackgroup": "house_load"
            },
            # {
            #     "type": "scatter",
            #     "x": houses.index,
            #     "y": houses["F0_house_A0.ev.load"].apply(lambda x: max(x, 0)),
            #     "name": "EV Power",
            #     "line": {"width": 0},
            #     "stackgroup": "house_generated"
            # },
            # {
            #     "type": "scatter",
            #     "x": houses.index,
            #     "y": houses["F0_house_A0.ev.load"].apply(lambda x: min(x, 0)),
            #     "name": "EV Load",
            #     "line": {"width": 0},
            #     "stackgroup": "house_load"
            # },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.ev.load"],
                "name": "EV Load",
                "line": {"width": 0},
                "stackgroup": "house_load",
                # "fillcolor": f"rgb({randint(0,255)}, {randint(0,255)}, {randint(0,255)})"
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.unresponsive_load"],
                "name": "Unresponsive Load",
                "line": {"width": 0},
                "stackgroup": "house_load",
                # "fillcolor": f"rgb({randint(0,255)}, {randint(0,255)}, {randint(0,255)})"
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.hvac.hvac_load"],
                "name": "HVAC Load",
                "line": {"width": 0},
                "stackgroup": "house_load",
                # "fillcolor": f"rgb({randint(0,255)}, {randint(0,255)}, {randint(0,255)})"
            },
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.total_house_load"],
                "name": "House Load",
            },
        ], rows=2, cols=1)

        fig.update_yaxes(title_text="Load", row=2, col=1)

        fig.add_trace(
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.ev.stored_energy"],
                "name": "EV Energy Stored",
            }, row=3, col=1)
        fig.add_trace(
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.ev.desired_charge_rate"],
                "name": "Specified Charge Rate",
            }, row=3, col=1, secondary_y=True)

        fig.update_yaxes(title_text="Energy", row=3, col=1, range=[0, max(houses["F0_house_A0.ev.stored_energy"])])
        fig.update_yaxes(title_text="Power", row=3, col=1, secondary_y=True,
                         range=[min(houses["F0_house_A0.ev.desired_charge_rate"]),
                                max(houses["F0_house_A0.ev.desired_charge_rate"])])

        fig.add_trace(
            {
                "type": "scatter",
                "x": auction.index,
                "y": auction["clearing_price"],
                "name": "Cleared Price",
            }, row=4, col=1, secondary_y=True)
        fig.add_trace(
            {
                "type": "scatter",
                "x": houses.index,
                "y": houses["F0_house_A0.hvac.hvac_on"].apply(lambda x: int(x)),
                "name": "HVAC On",
            }, row=4, col=1)

        fig.update_yaxes(title_text="Count", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=4, col=1, secondary_y=True)

        fig.write_image(f"{path}.svg")
        fig.write_html(f"{path}.html")
        print(f"wrote figure in {(millis() - s) * 1000:3f}ms")

    @staticmethod
    def make_figure_bids(h: dict, path):
        houses = h["houses"]
        bids = h["bids"]
        auction = h["auction"]
        vpp = h["vpp"]
        fig = make_subplots(rows=3, cols=1,
                            specs=[[{}], [{}], [{"secondary_y": True}]])
        prices = []
        for t, bids_round in bids["values.bid.values"].items():
            bids_round = pandas.DataFrame(bids_round,
                                          columns=["price", "quantity", "hvac_needed", "role", "unresp_load", "name",
                                                   "base_covered"])
            sellers = bids_round[bids_round["role"] == "seller"]
            buyers = bids_round[bids_round["role"] == "buyer"]
            prices.append([sellers["price"].min(), sellers["price"].mean(), sellers[
                "price"].max(), buyers["price"].min(), buyers["price"].mean(), buyers[
                               "price"].max()])

        prices = pandas.DataFrame(prices, columns=["min_seller_price", "avg_seller_price", "max_seller_price",
                                                   "min_buyer_price", "avg_buyer_price", "max_buyer_price"],
                                  index=bids.index)

        fig = make_subplots(rows=4, cols=1,
                            specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]])

        for name in ["min_seller_price", "avg_seller_price", "max_seller_price",
                     "min_buyer_price", "avg_buyer_price", "max_buyer_price"]:
            fig.add_trace(
                {
                    "type": "scatter",
                    "x": prices.index,
                    "y": prices[name],
                    "name": name,
                }, row=1, col=1)

        fig.add_trace(
            {
                "type": "scatter",
                "x": auction.index,
                "y": auction["clearing_price"],
                "name": "Clearing Price",
            }, row=2, col=1)


        fig.add_trace(
            {
                "type": "scatter",
                "x": auction.index,
                "y": auction["num_sellers"],
                "name": "Sellers",
            }, row=3, col=1)
        fig.add_trace(
            {
                "type": "scatter",
                "x": auction.index,
                "y": auction["num_buyers"],
                "name": "Buyers",
            }, row=3, col=1)

        fig.write_image(f"{path}.svg")
        fig.write_html(f"{path}.html")

    def figure(self):
        self.make_figure(self.history(), "progress")


if __name__ == "__main__":
    with open("metrics.pkl", "rb") as f:
        history = pickle.load(f)

    SubstationRecorder.make_figure_bids(history, "bids_metrics")
    # SubstationRecorder.make_figure(history, "final_metrics")
    # SubstationRecorder.make_figure_solo(history, "solo_metrics")

# class Tester:
#     def __init__(self):
#         self.test = 0
#         self.tester = None
#         self.dic = {"a": 2}
#         self.lis = None
#         self.lis2 = [{"a": x} for x in range(5)]
#
#     def add(self):
#         self.tester = Tester()
#
#
# a = Tester()
# a.add()
# a.lis = [Tester(), Tester(), Tester()]
# b = a.tester
# b.test = 1
#
# h = HistoryRecorder(a, ["test", "tester.test", "tester.dic.a", "lis.values.test", "lis.values.dic.a", "lis.1.dic.a",
#                         "lis.0.test"])
# # h = HistoryRecorder(a, ["lis2.sum.a"])
# h.record(1)
#
# a.test += 1
# b.test += 3
# b.dic = {"a": 10}
# a.lis[0].test = 501
# h.record(2)
#
# print(h.history)
