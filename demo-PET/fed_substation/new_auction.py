import pickle
from datetime import datetime

import helics
from dateutil.parser import parse
from helics import HelicsFederate
import pymarket as pm
from matplotlib import pyplot as plt


class NewAuction:
    def __init__(self, helics_federate: HelicsFederate):
        self.lmp = None
        self.refload = None
        self.refload_p = None
        self.refload_q = None
        self.helics_federate = helics_federate
        self.bids = []

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
        self.bids = bids

    def update_refload(self):
        c = helics.helicsInputGetComplex(self.subFeeder)
        self.refload_p = c.real * 0.001
        self.refload_q = c.imag * 0.001
        self.refload = self.refload_p  # math.sqrt(self.refload_p**2 + self.refload_q**2)

    def update_lmp(self):
        self.lmp = helics.helicsInputGetDouble(self.subLMP)

    def clear_market(self):
        mar = pm.Market()
        # self.bid = [self.bid_price, quantity, self.hvac.power_needed, self.role, self.unresponsive_load, self.name,
        #             base_covered]
        for i, bid in enumerate([b for b in self.bids if b[3] != "none-participant"]):
            mar.accept_bid(bid[1], bid[0], i, bid[3] == "buyer", 0, False)
        print(mar.bm.get_df())
        transactions, extras = mar.run('muda')
        mar.plot()
        plt.savefig("market.png")
        # mar.plot_method('p2p')
        # plt.savefig("p2p.png")
        print(transactions.get_df())


if __name__ == "__main__":
    with open("metrics.pkl", "rb") as f:
        history = pickle.load(f)
    bids = history["bids"]
    # print(datetime.strptime("2013-07-01 18:30:00", '%Y-%m-%d %H:%M:%S').strftime("%z"))
    bid_round = bids.asof(parse("2013-07-01 13:45:00-08:00"))
    a = NewAuction(None)
    a.collect_bids(bid_round[0])
    a.clear_market()
