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
        self.lmp = None
        self.refload = None
        self.refload_p = None
        self.refload_q = None
        self.helics_federate = helics_federate
        self.bids: DataFrame = None
        self.response: DataFrame = None
        self.period = 300

        self.num_sellers = 0
        self.num_buyers = 0
        self.num_nontcp = 0

        self.substation_seller_bid = [self.lmp, 5000, False, "seller", 0, "substation", False]

        # publications and subscriptions
        self.pubUnresp = helics.helicsFederateGetPublication(helics_federate, "unresponsive_mw")
        self.pubMax = helics.helicsFederateGetPublication(helics_federate, "responsive_max_mw")
        self.pubC1 = helics.helicsFederateGetPublication(helics_federate, "responsive_c1")
        self.pubC2 = helics.helicsFederateGetPublication(helics_federate, "responsive_c2")
        self.pubDeg = helics.helicsFederateGetPublication(helics_federate, "responsive_deg")
        self.pubAucPrice = helics.helicsFederateGetPublication(helics_federate, "clear_price")

        self.subFeeder = helics.helicsFederateGetSubscription(helics_federate, "gld1/distribution_load")
        self.subLMP = helics.helicsFederateGetSubscription(helics_federate, "pypower/LMP_B7")

    def collect_bids(self, bids):
        self.substation_seller_bid = [self.lmp, 10000, False, "seller", 0, "substation", False]
        self.bids = DataFrame(bids + [self.substation_seller_bid], columns=["price", "quantity", "hvac_power_needed", "role",
                                             "unresponsive_power", "name", "base_covered"])
        self.num_sellers = (self.bids["role"] == "seller").sum()
        self.num_buyers = (self.bids["role"] == "buyer").sum()
        self.num_nontcp = (self.bids["role"] == "none-participant").sum()

    def update_refload(self):
        c = helics.helicsInputGetComplex(self.subFeeder)
        self.refload_p = c.real * 0.001
        self.refload_q = c.imag * 0.001
        self.refload = self.refload_p  # math.sqrt(self.refload_p**2 + self.refload_q**2)

    def update_lmp(self):
        self.lmp = helics.helicsInputGetDouble(self.subLMP)

    def publish_clearing_price(self):
        self.pubAucPrice.publish(self.clearing_price)

    def clear_market(self):
        # self.collect_bids([[0, 1000, 0, "buyer", 0, "b", 0]])
        np.set_printoptions(precision=3, suppress=True)
        all_prices = sorted(set(self.bids["price"]))
        # create [price, quantity] matrices
        buyers = self.bids[self.bids["role"] == "buyer"][["price", "quantity"]]
        sellers = self.bids[self.bids["role"] == "seller"][["price", "quantity"]]

        # add bid number
        buyers = np.c_[buyers.to_numpy(), buyers.index]
        sellers = np.c_[sellers.to_numpy(), sellers.index]
        # print(self.bids)
        # buyers = np.array([bid[["bid_price", "quantity"]] + [i] for i, bid in self.bids.iterrows() if bid["role"] == "buyer"])
        # sellers = np.array([bid[["bid_price", "quantity"]] for i, bid in enumerate(self.bids) if bid["role"] == "seller"])
        # print(buyers)
        # buyers = np.array([[0., 1000., 5.]])
        # sellers = np.array([[]])
        agg = np.array([
            [
                x,
                np.sum(buyers[buyers[:, 0] >= x][:, 1]),  # total quantity buyers would buy at price x
                np.sum(sellers[sellers[:, 0] <= x][:, 1])  # total quantity sellers would sell at price x
            ]
            for x in all_prices
        ])
        sold_gt_bought = agg[agg[:, 1] <= agg[:, 2]]  # lowest price where quantity bought < quantity sold
        matched = sold_gt_bought[0, :] if len(sold_gt_bought) else np.array([float('inf'), 0])
        self.clearing_price = float(matched[0])
        print(f"clearing price {self.clearing_price}")
        cleared_quantity = np.min(matched[1:])

        cleared_buyers = buyers[buyers[:, 0] >= self.clearing_price]
        # all sellers sell totally up to buying total
        sellers = np.c_[
            sellers, [sellers[i, 1] * (np.sum(sellers[:i + 1, 1]) < cleared_quantity) for i in range(len(sellers))]]
        # print("cbuy\n", cleared_buyers)
        # print("sel\n", sellers)
        # choose a marginal seller randomly to cover the remainder
        if cleared_quantity > sellers[:, 3].sum():
            seller_marginal = choice(np.where(sellers[:, 3] == 0)[0])
            sellers[seller_marginal, 3] = cleared_quantity - sellers[:, 3].sum()

        # print("buyers:\n", cleared_buyers)
        # print("cleared_quantity", cleared_quantity)
        # print("sellers:\n", sellers)
        accepted_bids = np.concatenate([cleared_buyers[:, (2, 1)], sellers[sellers[:, 3] > 0][:, (2, 1)]])
        # print(accepted_bids)
        self.response = DataFrame([
            [bid["role"], accepted_bids[accepted_bids[:, 0] == i][0][1] if i in accepted_bids[:, 0] else 0.0,
             bid["price"], bid["quantity"]]
            for i, bid in self.bids.iterrows()
        ], columns=["role", "quantity", "bid_price", "bid_quantity"], index=self.bids["name"].values)
        print(self.response)


if __name__ == "__main__":
    with open("metrics.pkl", "rb") as f:
        history = pickle.load(f)
    bids = history["bids"]
    # print(datetime.strptime("2013-07-01 18:30:00", '%Y-%m-%d %H:%M:%S').strftime("%z"))
    # bid_round = bids.asof(parse("2013-07-01 13:45:00-08:00"))
    a = NewAuction(None)
    # a.collect_bids(bid_round[0])
    a.clear_market()
    # print(a.response[a.response.index == "F0_house_A0"])
    # print(a.response.loc["F0_house_A0"])
