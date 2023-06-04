import pickle
from datetime import datetime
from random import choice

import helics
import numpy as np
import pandas
from dateutil.parser import parse
from helics import HelicsFederate
from pandas import DataFrame, DatetimeIndex


def approx_lte(x, y):
    return x <= y or np.isclose(x, y)


def approx_gte(x, y):
    return x >= y or np.isclose(x, y)


class NewAuction:
    def __init__(self, helics_federate: HelicsFederate, start_time: datetime):
        self.clearing_price = 0
        self.lmp = 0.0
        self.refload = 0.0
        self.refload_p = 0.0
        self.refload_q = 0.0
        self.helics_federate = helics_federate
        self.bids: DataFrame = DataFrame()
        self.response: DataFrame = DataFrame()
        self.period = 300

        self.num_bids = 0
        self.num_sellers = 0
        self.num_buyers = 0
        self.num_nontcp = 0

        self.history = DataFrame([[self.clearing_price, 0.0]], columns=["clearing_price", "cleared_quantity"],
                                 index=[start_time])
        # # substation always sells infinite at LMP
        # self.substation_seller_bid = [self.lmp, float('inf'), False, "seller", 0, "substation", False]

        # publications and subscriptions
        if helics_federate is not None:
            self.pubUnresp = helics_federate.publications["sub1/unresponsive_mw"]
            self.pubMax = helics_federate.publications["sub1/responsive_max_mw"]
            self.pubC1 = helics_federate.publications["sub1/responsive_c1"]
            self.pubC2 = helics_federate.publications["sub1/responsive_c2"]
            self.pubDeg = helics_federate.publications["sub1/responsive_deg"]
            self.pubAucPrice = helics_federate.publications["sub1/clear_price"]

            self.subFeeder = helics_federate.subscriptions["gld1/distribution_load"]
            self.sub_lmp = helics_federate.subscriptions["pypower/LMP_B7"]

    def collect_bids(self, bids):
        # self.substation_seller_bid = [self.lmp, float('inf'), False, "seller", 0, "substation", False]
        self.bids = DataFrame(bids, columns=["trader", "role", "price", "quantity"])
        print(self.bids)
        self.num_bids = len(self.bids)
        self.num_sellers = (self.bids["role"] == "seller").sum()
        self.num_buyers = (self.bids["role"] == "buyer").sum()
        self.num_nontcp = (self.bids["role"] == "none-participant").sum()

    def update_refload(self):
        c = self.subFeeder.complex
        self.refload_p = c.real * 0.001
        self.refload_q = c.imag * 0.001
        self.refload = self.refload_p  # math.sqrt(self.refload_p**2 + self.refload_q**2)

    def update_lmp(self):
        self.lmp = self.sub_lmp.double

    def clear_market(self, current_time: datetime):
        all_prices = sorted([p for p in set(self.bids["price"])])
        # multiunit auction
        buyers = self.bids[self.bids["role"] == "buyer"].sample(frac=1).sort_values("price",
                                                                                    ascending=False).reset_index()
        sellers = self.bids[self.bids["role"] == "seller"].sample(frac=1).sort_values("price",
                                                                                      ascending=True).reset_index()
        agg = np.array([
            [
                x,
                buyers[buyers["price"] >= x]["quantity"].sum(),  # total quantity buyers would buy at price x
                sellers[sellers["price"] <= x]["quantity"].sum()  # total quantity sellers would sell at price x
            ]
            for x in all_prices
        ])
        # print(agg)
        bought_lt_sold = agg[agg[:, 1] <= agg[:, 2]]
        print("AGG\n", agg)
        print("BLT\n", bought_lt_sold)
        initial_clearing_price, max_cleared_quantity = bought_lt_sold[0, [0, 2]] if len(bought_lt_sold) else (
            float('inf'), 0)
        # print(sold_gt_bought[-1])
        # cleared_quantity = agg[agg[:, 0] == clearing_price][0][2]
        # cleared_quantity = sold_gt_bought[0, 1] if len(sold_gt_bought) else 0
        cleared_buyers = buyers[np.logical_or(buyers["quantity"].cumsum() <= max_cleared_quantity,
                                              np.isclose(buyers["quantity"].cumsum(), max_cleared_quantity))]
        self.clearing_price = initial_clearing_price
        print(
            f"buy bids totaled {buyers['quantity'].sum()}, published clearing price {self.clearing_price}, max cleared quantity {max_cleared_quantity}")
        cleared_quantity = cleared_buyers["quantity"].sum()
        cleared_sellers = sellers[sellers["price"] <= self.clearing_price]
        marginal_quantity = cleared_sellers["quantity"].sum() - cleared_quantity
        if marginal_quantity > 0:
            cleared_sellers.loc[cleared_sellers.index[-1], "quantity"] = cleared_quantity - \
                                                                         cleared_sellers.loc[cleared_sellers.index[
                                                                                             :-1], "quantity"].sum()

        assert np.isclose(cleared_sellers["quantity"].sum(),
                          cleared_quantity), f"AUCTION ERROR: cleared sellers {cleared_sellers} vs buyers {cleared_buyers}"
        print("CLEARED BUYERS:\n", cleared_buyers)
        print("CLEARED SELLERS:\n", cleared_sellers)
        cleared_bids = pandas.concat([cleared_sellers, cleared_buyers])
        # cleared_bids = cleared_bids.assign(price=self.clearing_price)
        print("Cleared\n", cleared_bids)
        self.history = self.history.append(DataFrame(
            {"clearing_price": self.clearing_price, "cleared_quantity": cleared_quantity},
            index=[current_time]))
        print(self.history)
        return {
            trader: cleared_bids.loc[cleared_bids["trader"] == trader, ["quantity", "role"]].to_dict(
                'records')
            for trader in set(self.bids["trader"])
        }

    # print(cleared_sellers)

    # def clear_market_double_auction(self):
    #     # np.set_printoptions(precision=3, suppress=True)
    #     all_prices = sorted([p for p in set(self.bids["price"])])
    #
    #     # shuffle before sorting to randomise tiebreaking
    #     buyers = self.bids[self.bids["role"] == "buyer"].sample(frac=1).sort_values("price",
    #                                                                                 ascending=False).reset_index()
    #     sellers = self.bids[self.bids["role"] == "seller"].sample(frac=1).sort_values("price",
    #                                                                                   ascending=True).reset_index()
    #
    #     agg = np.array([
    #         [
    #             x,
    #             buyers[buyers["price"] >= x]["quantity"].sum(),  # total quantity buyers would buy at price x
    #             sellers[sellers["price"] <= x]["quantity"].sum()  # total quantity sellers would sell at price x
    #         ]
    #         for x in all_prices
    #     ])
    #     # print("bbuy", all_prices)
    #     # print(buyers[np.logical_or(buyers["price"] >= all_prices[0], np.isclose(buyers["price"], all_prices[0]))]["quantity"].sum())
    #     # print(buyers[np.logical_or(buyers["price"] >= all_prices[1], np.isclose(buyers["price"], all_prices[1]))]["quantity"].sum())
    #     # print(buyers[np.logical_or(buyers["price"] >= 0.0001, np.isclose(buyers["price"], 0.0001))]["quantity"].sum())
    #
    #     sold_gt_bought = agg[agg[:, 1] <= agg[:, 2]]
    #     # print(sold_gt_bought)
    #     cleared_quantity = sold_gt_bought[0, 1] if len(sold_gt_bought) else 0
    #
    #     cleared_sellers = sellers[:(sellers["quantity"].cumsum() > cleared_quantity).idxmax() + 1]
    #     self.clearing_price = cleared_sellers["price"].min()
    #     # if self.pubAucPrice:
    #     #     self.pubAucPrice.publish(self.clearing_price)
    #
    #     print(f"published clearing price {self.clearing_price}, cleared quantity {cleared_quantity}")
    #     cleared_buyers = buyers[np.logical_or(buyers["quantity"].cumsum() <= cleared_quantity,
    #                                           np.isclose(buyers["quantity"].cumsum(), cleared_quantity))]
    #     # print("bb")
    #     # print(buyers.cumsum())
    #     # print("cb")
    #     # print(cleared_buyers.cumsum())
    #     # print("cs")
    #     # print(cleared_sellers.cumsum())
    #
    #     if cleared_sellers["quantity"].sum() > cleared_quantity:
    #         marginal_quantity = cleared_quantity - cleared_sellers[:-1]['quantity'].sum()
    #         cleared_sellers.loc[cleared_sellers.index[-1], "quantity"] = marginal_quantity
    #
    #     # print("qrs", cleared_quantity, cleared_sellers["quantity"].sum(), cleared_buyers["quantity"].sum())
    #     assert np.isclose(cleared_sellers["quantity"].sum(), cleared_buyers[
    #         "quantity"].sum()), f"AUCTION ERROR: cleared sellers {cleared_sellers} vs buyers {cleared_buyers}"
    #     # print(cleared_sellers)
    #     # print(cleared_buyers)
    #     cleared_bids = pandas.concat([cleared_sellers, cleared_buyers])
    #     # cleared_bids = cleared_bids.assign(price=self.clearing_price)
    #     print("Cleared\n", cleared_bids)
    #     return {
    #         trader: cleared_bids.loc[cleared_bids["trader"] == trader, ["quantity", "role"]].to_dict(
    #             'records')
    #         for trader in set(self.bids["trader"])
    #     }
    #
    #     # response = DataFrame([
    #     #     cleared_self.bids[cleared_self.bids["trader"] == trader]["quantity", "role"] if
    #     #     for trader in self.bids["trader"]
    #     # ])
    #     # print(response)
    #     # self.response = DataFrame[[
    #     #     cleared_self.bids.loc["price"], cleared_self.bids["quantity"]
    #     # ]]
    #
    #     # self.response =
    #
    #     # sellers = self.bids[self.bids["role"] == "seller"]
    #     # # add bid number
    #     # buyers = np.c_[buyers.to_numpy(), buyers.index]
    #     # sellers = np.c_[sellers.to_numpy(), sellers.index]
    #
    #     # agg = np.array([
    #     #     [
    #     #         x,
    #     #         np.sum(buyers[buyers[:, 0] >= x][:, 1]),  # total quantity buyers would buy at price x
    #     #         np.sum(sellers[sellers[:, 0] <= x][:, 1])  # total quantity sellers would sell at price x
    #     #     ]
    #     #     for x in all_prices
    #     # ])
    #     # response = DataFrame([
    #     #     cleared_self.bids[cleared_self.bids["trader"] == trader]["quantity", "role"] if
    #     #     for trader in self.bids["trader"]
    #     # ])
    #     # print(response)
    #     # self.response = DataFrame[[
    #     #     cleared_self.bids.loc["price"], cleared_self.bids["quantity"]
    #     # ]]
    #
    #     # self.response =
    #
    #     # sellers = self.bids[self.bids["role"] == "seller"]
    #     # # add bid number
    #     # buyers = np.c_[buyers.to_numpy(), buyers.index]
    #     # sellers = np.c_[sellers.to_numpy(), sellers.index]
    #
    #     # agg = np.array([
    #     #     [
    #     #         x,
    #     #         np.sum(buyers[buyers[:, 0] >= x][:, 1]),  # total quantity buyers would buy at price x
    #     #         np.sum(sellers[sellers[:, 0] <= x][:, 1])  # total quantity sellers would sell at price x
    #     #     ]
    #     #     for x in all_prices
    #     # ])
    #     # sold_gt_bought = agg[agg[:, 1] <= agg[:, 2
    #     # sold_gt_bought = agg[agg[:, 1] <= agg[:, 2]]  # price points where quantity bought <= quantity sold
    #     # self.clearing_price, cleared_quantity = sold_gt_bought[0, :] if len(sold_gt_bought) else float('inf'), 0
    #     # print(agg, "\n", sold_gt_bought)
    #     # print(f"clearing price {self.clearing_price}, {cleared_quantity}, {cleared_quantity > 0.0}")
    #     #
    #     # cleared_buyers = (buyers[buyers[:, 0] >= self.clearing_price]) if cleared_quantity > 0.0 else np.array([[]])
    #     # # all sellers sell totally up to buying total
    #     # sellers = np.c_[
    #     #     sellers, [sellers[i, 1] * (np.sum(sellers[:i + 1, 1]) <= cleared_quantity) for i in range(len(sellers))]]
    #     #
    #     # # choose a marginal seller randomly to cover the remainder
    #     # if cleared_quantity > sellers[:, 3].sum():
    #     #     seller_marginal = choice(np.where(sellers[:, 3] == 0)[0])
    #     #     sellers[seller_marginal, 3] = cleared_quantity - sellers[:, 3].sum()
    #     #
    #     # cleared_sellers = sellers[sellers[:, 3] > 0][:, (2, 1)]
    #     #
    #     # accepted_self.bids = np.concatenate([cleared_buyers[:, (2, 1)], cleared_sellers]) if cleared_buyers else cleared_sellers
    #     #
    #     # self.response = DataFrame([
    #     #     [bid["role"], accepted_self.bids[accepted_self.bids[:, 0] == i][0][1] if i in accepted_self.bids[:, 0] else 0.0,
    #     #      bid["price"], bid["quantity"]]
    #     #     for i, bid in self.bids.iterrows()
    #     # ], columns=["role", "quantity", "bid_price", "bid_quantity"], index=self.bids["name"].values)
    #     # print(self.response)


if __name__ == "__main__":
    with open("metrics.pkl", "rb") as f:
        history = pickle.load(f)
        # tbids = [
        #     [1000, 200, False, "seller", 0, "ann", False],
        #     [10000, 200, False, "seller", 0, "bobby", False],
        #     [1500, 400, False, "seller", 0, "ekk", False],
        #     [3100, 800, False, "seller", 0, "okr", False],
        #     [3150, 800, False, "seller", 0, "asdasd", False],
        #     # [1210, 1000, False, "seller", 0, "5", False],
        #     [float('inf'), 1600, False, "buyer", 0, "a", False],
        #     [3150, 200, False, "buyer", 0, "a", False],
        #     [300, 200, False, "buyer", 0, "b", False],
        #     [10, 2000, False, "buyer", 0, "c", False]
        # ]
        bids = history["auction"]["bids"]
        # print(datetime.strptime("2013-07-01 18:30:00", '%Y-%m-%d %H:%M:%S').strftime("%z"))
        bid_round = bids.loc[parse("2013-07-01 13:09:58-08:00")]
        a = NewAuction(None)
        a.collect_bids(bid_round[["trader", "role", "price", "quantity"]])
        # a.collect_bids(bid_round[0])
        print(a.clear_market())
        # print(a.response[a.response.index == "F0_house_A0"])
        # print(a.response.loc["F0_house_A0"])
