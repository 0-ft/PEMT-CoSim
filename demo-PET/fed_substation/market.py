import json
import pickle
from datetime import datetime

import numpy as np
import pandas
from dateutil.parser import parse
from helics import HelicsFederate
from pandas import DataFrame
from scipy.stats import iqr

# SELLER ONLY DIVISIBLE
def match_orders(bids):
    buyers = bids[bids["role"] == "buyer"].sample(frac=1).sort_values("price",
                                                                      ascending=False)[
        ["trader", "price", "quantity"]].values
    sellers = bids[bids["role"] == "seller"].sample(frac=1).sort_values("price",
                                                                        ascending=True)[
        ["trader", "price", "quantity"]].values
    buyers = buyers[buyers[:, 1] > 0]
    traders = set(bids["trader"])
    transactions = []
    while len(match := np.where([b[1] >= s[1] and s[2] >= b[2] for b, s in zip(buyers, sellers)])[0]):
        i, (buyer, seller) = match[0], (buyers[match[0]], sellers[match[0]])
        transaction_quantity = min(buyer[2], seller[2])
        transactions.append(
            {"seller": seller[0], "buyer": buyer[0], "quantity": transaction_quantity, "price": seller[1]})
        buyer[2] -= transaction_quantity
        seller[2] -= transaction_quantity
        if buyer[2] == 0.0:
            buyers = np.delete(buyers, i, axis=0)
        if seller[2] == 0.0:
            sellers = np.delete(sellers, i, axis=0)
    # print(json.dumps(transactions, indent=2))
    response = {
        trader: [
            {"price": t["price"], "quantity": t["quantity"], "role": "buyer" if t["buyer"] == trader else "seller"}
            for t in transactions
            if trader in [t["buyer"], t["seller"]]
        ] for trader in traders
    }
    # print(json.dumps(response, indent=2))
    return transactions, response


class ContinuousDoubleAuction:
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
        self.fraction_sellers_cleared = 0
        self.fraction_buyers_cleared = 0

        self.history = DataFrame([[self.clearing_price, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                 columns=["clearing_price", "cleared_quantity", "average_since", "iqr_since",
                                          "fraction_buyers_cleared", "fraction_sellers_cleared"],
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
            self.pub_clearing_price = helics_federate.publications["sub1/clear_price"]

            self.subFeeder = helics_federate.subscriptions["gld1/distribution_load"]
            self.sub_lmp = helics_federate.subscriptions["pypower/LMP_B7"]

    def update_stats(self):
        self.history["average_since"] = (self.history.loc[::-1, "clearing_price"]
                                         .cumsum() / range(1, len(self.history) + 1))[::-1]
        self.history["iqr_since"] = [
            iqr(self.history.loc[self.history.index >= i, "clearing_price"])
            for i in self.history.index
        ]

    def collect_bids(self, bids):
        # self.substation_seller_bid = [self.lmp, float('inf'), False, "seller", 0, "substation", False]
        self.bids = DataFrame(bids, columns=["trader", "role", "price", "quantity"])
        print("auction got bids:")
        print(self.bids.to_string())
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
        transactions, response = match_orders(self.bids)
        cleared_quantity = sum(t["quantity"] for t in transactions)
        average_price = sum(t["price"] * t["quantity"] for t in transactions) / cleared_quantity
        self.clearing_price = average_price
        self.pub_clearing_price.publish(self.clearing_price)
        self.history = pandas.concat([self.history, DataFrame(
            {"clearing_price": self.clearing_price, "cleared_quantity": cleared_quantity,
             "fraction_sellers_cleared": 0 / self.num_buyers,
             "fraction_buyers_cleared": 0 / self.num_sellers},
            index=[current_time])])
        self.update_stats()
        return response

    # def clear_market(self, current_time: datetime):
    #     all_prices = sorted([p for p in set(self.bids["price"])])
    #     # multiunit auction
    #     buyers = self.bids[self.bids["role"] == "buyer"].sample(frac=1).sort_values("price",
    #                                                                                 ascending=False).reset_index()
    #     sellers = self.bids[self.bids["role"] == "seller"].sample(frac=1).sort_values("price",
    #                                                                                   ascending=True).reset_index()
    #     print("CLEARING THESE BIDS:")
    #     print(self.bids)
    #     agg = np.array([
    #         [
    #             x,
    #             buyers[buyers["price"] >= x]["quantity"].sum(),  # total quantity buyers would buy at price x
    #             sellers[sellers["price"] <= x]["quantity"].sum()  # total quantity sellers would sell at price x
    #         ]
    #         for x in all_prices
    #     ])
    #     # print(agg)
    #     bought_lt_sold = agg[agg[:, 1] <= agg[:, 2]]
    #     # print("AGG\n", agg)
    #     # print("BLT\n", bought_lt_sold)
    #     initial_clearing_price, quantity_bought, quantity_sold = bought_lt_sold[0, [0, 1, 2]] if len(
    #         bought_lt_sold) else (
    #         float('inf'), 0, 0)
    #     # print(sold_gt_bought[-1])
    #     # cleared_quantity = agg[agg[:, 0] == clearing_price][0][2]
    #     # cleared_quantity = sold_gt_bought[0, 1] if len(sold_gt_bought) else 0
    #     # cleared_buyers = buyers[np.logical_or(buyers["quantity"].cumsum() <= max_cleared_quantity,
    #     #                                       np.isclose(buyers["quantity"].cumsum(), max_cleared_quantity))]
    #     cleared_buyers = buyers[buyers["price"] >= initial_clearing_price]
    #     # print((sellers["quantity"].cumsum() > quantity_sold))
    #     # print(sellers["quantity"].cumsum())
    #     cleared_sellers = sellers.iloc[:(sellers["quantity"].cumsum() > quantity_bought).idxmax() + 1]
    #     # print("CLEARED BUYERS:\n", cleared_buyers)
    #     # print("CLEARED SELLERS:\n", cleared_sellers, quantity_sold)
    #     print(
    #         f"buy bids totaled {buyers['quantity'].sum()}, initial clearing price {initial_clearing_price}, max sold {quantity_sold}, max bought {quantity_bought}")
    #     cleared_quantity = cleared_buyers["quantity"].sum()
    #     marginal_quantity = cleared_sellers["quantity"].sum() - cleared_quantity
    #     if marginal_quantity > 0:
    #         cleared_sellers.loc[cleared_sellers.index[-1], "quantity"] = cleared_quantity - \
    #                                                                      cleared_sellers.loc[cleared_sellers.index[
    #                                                                                          :-1], "quantity"].sum()
    #
    #     cleared_bids = pandas.concat([cleared_sellers, cleared_buyers])
    #     # cleared_bids = cleared_bids.assign(price=self.clearing_price)
    #     print("Cleared\n", cleared_bids)
    #     assert np.isclose(cleared_sellers["quantity"].sum(),
    #                       cleared_quantity), f"AUCTION ERROR: cleared sellers {cleared_sellers} vs buyers {cleared_buyers}"
    #
    #     # print("CLEARED BUYERS:\n", cleared_buyers)
    #     # print("CLEARED SELLERS:\n", cleared_sellers)
    #     self.clearing_price = cleared_bids[cleared_bids["quantity"] > 0.0]["price"].min()
    #     if self.pub_clearing_price:
    #         self.pub_clearing_price.publish(self.clearing_price)
    #     self.fraction_buyers_cleared = len(cleared_buyers) / self.num_buyers
    #     self.fraction_sellers_cleared = len(cleared_sellers) / self.num_sellers
    #     print("published CLEARING PRICE", self.clearing_price)
    #     self.history = pandas.concat([self.history, DataFrame(
    #         {"clearing_price": self.clearing_price, "cleared_quantity": cleared_quantity,
    #          "fraction_sellers_cleared": len(cleared_buyers) / self.num_buyers,
    #          "fraction_buyers_cleared": len(cleared_sellers) / self.num_sellers},
    #         index=[current_time])])
    #     self.update_stats()
    #     return {
    #         trader: cleared_bids.loc[cleared_bids["trader"] == trader, ["quantity", "role"]].to_dict(
    #             'records')
    #         for trader in set(self.bids["trader"])
    #     }


def test_auction(auction: ContinuousDoubleAuction):
    test_bids = [
        ["s1", "seller", 1.6, float('inf')],
        ["b1", "buyer", float('inf'), 1600],
    ]
    auction.lmp = 3.0
    auction.collect_bids(test_bids)
    response = auction.clear_market(datetime.now())
    # assert auction.clearing_price == 1.6
    assert response["s1"][0]["quantity"] == 1600.0
    assert response["s1"][0]["role"] == "seller"
    assert response["b1"][0]["quantity"] == 1600.0
    assert response["b1"][0]["role"] == "buyer"

    test_bids = [
        ["s1", "seller", 1.6, float('inf')],
        ["s2", "seller", 1.2, float('inf')],
        ["b1", "buyer", float('inf'), 1600],
    ]
    auction.lmp = 3.0
    auction.collect_bids(test_bids)
    response = auction.clear_market(datetime.now())
    # assert auction.clearing_price == 1.2
    assert len(response["s1"]) == 0
    assert response["s2"][0]["quantity"] == 1600.0
    assert response["s2"][0]["role"] == "seller"
    assert response["b1"][0]["quantity"] == 1600.0
    assert response["b1"][0]["role"] == "buyer"

    test_bids = [
        ["s1", "seller", 1.6, float('inf')],
        ["s2", "seller", 1.2, float('inf')],
        ["b1", "buyer", float('inf'), 1600],
        ["b2", "buyer", 0, 4000],
        ["b3", "buyer", 0, 0],
    ]
    auction.lmp = 3.0
    auction.collect_bids(test_bids)
    response = auction.clear_market(datetime.now())
    # assert auction.clearing_price == 1.2
    assert len(response["s1"]) == 0
    assert response["s2"][0]["quantity"] == 1600.0
    assert response["s2"][0]["role"] == "seller"
    assert response["b1"][0]["quantity"] == 1600.0
    assert response["b1"][0]["role"] == "buyer"
    assert len(response["b2"]) == 0
    assert len(response["b3"]) == 0

    test_bids = [
        ["s1", "seller", 1.6, float('inf')],
        ["s2", "seller", 1.2, float('inf')],
        ["b1", "buyer", float('inf'), 1600],
        ["b2", "buyer", 20, 4000],
        ["b3", "buyer", 0, 0],
    ]
    auction.lmp = 3.0
    auction.collect_bids(test_bids)
    response = auction.clear_market(datetime.now())
    print(response)
    # assert auction.clearing_price == 1.2
    assert len(response["s1"]) == 0
    assert response["s2"][0]["quantity"] == 5600.0
    assert response["s2"][0]["role"] == "seller"
    assert response["b1"][0]["quantity"] == 1600.0
    assert response["b1"][0]["role"] == "buyer"
    assert response["b2"][0]["quantity"] == 4000.0
    assert response["b2"][0]["role"] == "buyer"
    assert len(response["b3"]) == 0

    test_bids = [
        ["s1", "seller", 1.6, float('inf')],
        ["s2", "seller", 0.0, 0.0],
        ["s3", "seller", 0.0, 0.0],
        ["s4", "seller", 0.0, 0.0],
        ["s5", "seller", 1.2, float('inf')],
        ["b1", "buyer", float('inf'), 1600],
        ["b2", "buyer", 20, 4000],
        ["b3", "buyer", 0, 0],
    ]
    auction.lmp = 3.0
    auction.collect_bids(test_bids)
    response = auction.clear_market(datetime.now())
    # assert auction.clearing_price == 1.2
    assert len(response["s1"]) == 0

    assert response["s2"][0]["quantity"] == 0.0
    assert response["s2"][0]["role"] == "seller"

    assert response["s3"][0]["quantity"] == 0.0
    assert response["s3"][0]["role"] == "seller"

    assert response["s4"][0]["quantity"] == 0.0
    assert response["s4"][0]["role"] == "seller"

    assert response["s5"][0]["quantity"] == 5600.0
    assert response["s5"][0]["role"] == "seller"

    assert response["b1"][0]["quantity"] == 1600.0
    assert response["b1"][0]["role"] == "buyer"
    assert response["b2"][0]["quantity"] == 4000.0
    assert response["b2"][0]["role"] == "buyer"
    assert len(response["b3"]) == 0

    # more buy than sell

    test_bids = [
        ["s1", "seller", 1.6, 1000],
        ["b2", "buyer", 1.9, 1000],
        ["b3", "buyer", 1.8, 400],
    ]
    auction.lmp = 3.0
    auction.collect_bids(test_bids)
    response = auction.clear_market(datetime.now())
    print(response)
    # assert auction.clearing_price == 1.2
    # assert len(response["s1"]) == 0
    # assert response["s2"][0]["quantity"] == 1600.0
    # assert response["s2"][0]["role"] == "seller"
    # assert response["b1"][0]["quantity"] == 1600.0
    # assert response["b1"][0]["role"] == "buyer"
    # assert len(response["b2"]) == 0
    # assert len(response["b3"]) == 0


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
        # bids = history["auction"]["bids"]
        # print(datetime.strptime("2013-07-01 18:30:00", '%Y-%m-%d %H:%M:%S').strftime("%z"))
        # bid_round = bids.loc[parse("2013-07-01 13:09:58-08:00")]
        a = ContinuousDoubleAuction(None, datetime.now())
        test_auction(a)
        # a.collect_bids(bid_round[["trader", "role", "price", "quantity"]])
        # print(a.clear_market())
        # print(a.response[a.response.index == "F0_house_A0"])
        # print(a.response.loc["F0_house_A0"])
