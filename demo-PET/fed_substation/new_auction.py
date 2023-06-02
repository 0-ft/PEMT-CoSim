import pickle
from random import choice

import helics
import numpy as np
import pandas
from dateutil.parser import parse
from helics import HelicsFederate
from pandas import DataFrame


class NewAuction:
    def __init__(self, helics_federate: HelicsFederate):
        self.clearing_price = 0
        self.lmp = 0.0
        self.refload = 0.0
        self.refload_p = 0.0
        self.refload_q = 0.0
        self.helics_federate = helics_federate
        self.latest_bids: DataFrame = DataFrame()
        self.response: DataFrame = DataFrame()
        self.period = 300

        self.num_sellers = 0
        self.num_buyers = 0
        self.num_nontcp = 0

        # # substation always sells infinite at LMP
        # self.substation_seller_bid = [self.lmp, float('inf'), False, "seller", 0, "substation", False]

        # publications and subscriptions
        # self.pubUnresp = helics.helicsFederateGetPublication(helics_federate, "unresponsive_mw")
        # self.pubMax = helics.helicsFederateGetPublication(helics_federate, "responsive_max_mw")
        # self.pubC1 = helics.helicsFederateGetPublication(helics_federate, "responsive_c1")
        # self.pubC2 = helics.helicsFederateGetPublication(helics_federate, "responsive_c2")
        # self.pubDeg = helics.helicsFederateGetPublication(helics_federate, "responsive_deg")
        # self.pubAucPrice = helics.helicsFederateGetPublication(helics_federate, "clear_price")
        #
        # self.subFeeder = helics.helicsFederateGetSubscription(helics_federate, "gld1/distribution_load")
        # self.subLMP = helics.helicsFederateGetSubscription(helics_federate, "pypower/LMP_B7")

    def collect_bids(self, bids):
        # self.substation_seller_bid = [self.lmp, float('inf'), False, "seller", 0, "substation", False]
        self.latest_bids = DataFrame(bids, columns=["price", "quantity", "hvac_power_needed", "role",
                                             "unresponsive_power", "trader", "base_covered"])
        self.num_sellers = (self.latest_bids["role"] == "seller").sum()
        self.num_buyers = (self.latest_bids["role"] == "buyer").sum()
        self.num_nontcp = (self.latest_bids["role"] == "none-participant").sum()

    def update_refload(self):
        c = helics.helicsInputGetComplex(self.subFeeder)
        self.refload_p = c.real * 0.001
        self.refload_q = c.imag * 0.001
        self.refload = self.refload_p  # math.sqrt(self.refload_p**2 + self.refload_q**2)

    def update_lmp(self):
        self.lmp = helics.helicsInputGetDouble(self.subLMP)

    def publish_clearing_price(self):
        self.pubAucPrice.publish(self.clearing_price)

    def clear_market(self, bids):
        # np.set_printoptions(precision=3, suppress=True)
        all_prices = sorted([p for p in set(bids["price"])])

        # shuffle before sorting to randomise tiebreaking
        buyers = bids[bids["role"] == "buyer"].sample(frac=1).sort_values("price",
                                                                                    ascending=False).reset_index()
        sellers = bids[bids["role"] == "seller"].sample(frac=1).sort_values("price",
                                                                                      ascending=True).reset_index()

        agg = np.array([
            [
                x,
                buyers[buyers["price"] >= x]["quantity"].sum(),  # total quantity buyers would buy at price x
                sellers[sellers["price"] <= x]["quantity"].sum()  # total quantity sellers would sell at price x
            ]
            for x in all_prices
        ])
        print(agg)

        sold_gt_bought = agg[agg[:, 1] < agg[:, 2]]
        clearing_price, cleared_quantity = sold_gt_bought[0, :2] if len(sold_gt_bought) else (float('inf'), 0)
        print(f"clearing price {clearing_price}, cleared quantity {cleared_quantity}")
        cleared_buyers = buyers[buyers["quantity"].cumsum() <= cleared_quantity]
        cleared_sellers = sellers[:(sellers["quantity"].cumsum() >= cleared_quantity).idxmax() + 1]

        if cleared_sellers["quantity"].sum() > cleared_quantity:
            cleared_sellers.loc[cleared_sellers.index[-1], "quantity"] = cleared_quantity - sum(
                cleared_sellers[:-1]["quantity"])

        assert cleared_sellers["quantity"].sum() == cleared_buyers["quantity"].sum(), f"AUCTION ERROR: cleared sellers {cleared_sellers} vs buyers {cleared_buyers}"
        # print(cleared_sellers)
        # print(cleared_buyers)
        cleared_bids = pandas.concat([cleared_sellers, cleared_buyers])
        return {
            trader: cleared_bids.loc[cleared_bids["trader"] == trader, ["quantity", "role"]].to_dict('records')
            for trader in set(bids["trader"])
        }

        # response = DataFrame([
        #     cleared_bids[cleared_bids["trader"] == trader]["quantity", "role"] if
        #     for trader in bids["trader"]
        # ])
        # print(response)
        # self.response = DataFrame[[
        #     cleared_bids.loc["price"], cleared_bids["quantity"]
        # ]]

        # self.response =

        # sellers = bids[bids["role"] == "seller"]
        # # add bid number
        # buyers = np.c_[buyers.to_numpy(), buyers.index]
        # sellers = np.c_[sellers.to_numpy(), sellers.index]

        # agg = np.array([
        #     [
        #         x,
        #         np.sum(buyers[buyers[:, 0] >= x][:, 1]),  # total quantity buyers would buy at price x
        #         np.sum(sellers[sellers[:, 0] <= x][:, 1])  # total quantity sellers would sell at price x
        #     ]
        #     for x in all_prices
        # ])
        # sold_gt_bought = agg[agg[:, 1] <= agg[:, 2]]  # price points where quantity bought <= quantity sold
        # self.clearing_price, cleared_quantity = sold_gt_bought[0, :] if len(sold_gt_bought) else float('inf'), 0
        # print(agg, "\n", sold_gt_bought)
        # print(f"clearing price {self.clearing_price}, {cleared_quantity}, {cleared_quantity > 0.0}")
        #
        # cleared_buyers = (buyers[buyers[:, 0] >= self.clearing_price]) if cleared_quantity > 0.0 else np.array([[]])
        # # all sellers sell totally up to buying total
        # sellers = np.c_[
        #     sellers, [sellers[i, 1] * (np.sum(sellers[:i + 1, 1]) <= cleared_quantity) for i in range(len(sellers))]]
        #
        # # choose a marginal seller randomly to cover the remainder
        # if cleared_quantity > sellers[:, 3].sum():
        #     seller_marginal = choice(np.where(sellers[:, 3] == 0)[0])
        #     sellers[seller_marginal, 3] = cleared_quantity - sellers[:, 3].sum()
        #
        # cleared_sellers = sellers[sellers[:, 3] > 0][:, (2, 1)]
        #
        # accepted_bids = np.concatenate([cleared_buyers[:, (2, 1)], cleared_sellers]) if cleared_buyers else cleared_sellers
        #
        # self.response = DataFrame([
        #     [bid["role"], accepted_bids[accepted_bids[:, 0] == i][0][1] if i in accepted_bids[:, 0] else 0.0,
        #      bid["price"], bid["quantity"]]
        #     for i, bid in bids.iterrows()
        # ], columns=["role", "quantity", "bid_price", "bid_quantity"], index=bids["name"].values)
        # print(self.response)


if __name__ == "__main__":
    with open("metrics.pkl", "rb") as f:
        history = pickle.load(f)
        tbids = [
            [1000, 200, False, "seller", 0, "ann", False],
            [10000, 200, False, "seller", 0, "bobby", False],
            [1500, 400, False, "seller", 0, "ekk", False],
            [3100, 800, False, "seller", 0, "okr", False],
            [3150, 800, False, "seller", 0, "asdasd", False],
            # [1210, 1000, False, "seller", 0, "5", False],
            [float('inf'), 1600, False, "buyer", 0, "a", False],
            [3150, 200, False, "buyer", 0, "a", False],
            [300, 200, False, "buyer", 0, "b", False],
            [10, 2000, False, "buyer", 0, "c", False]
        ]
        # bids = history["bids"]
        # print(datetime.strptime("2013-07-01 18:30:00", '%Y-%m-%d %H:%M:%S').strftime("%z"))
        # bid_round = bids.asof(parse("2013-07-01 13:45:00-08:00"))
        a = NewAuction(None)
        a.collect_bids(tbids)
        # a.collect_bids(bid_round[0])
        a.clear_market(a.latest_bids)
        # print(a.response[a.response.index == "F0_house_A0"])
        # print(a.response.loc["F0_house_A0"])
