import json
from math import isnan

import helics
from helics import HelicsFederate

market_period = 300
period = 20

fed_json = {
    "name": "f1",
    # "uninterruptible": False,
    "publications": [
        {
            "key": f"charge_rate",
            "type": "double",
            "global": False
        },
        {
            "key": f"charge_max",
            "type": "double",
            "global": False
        },
        {
            "key": f"stored_energy",
            "type": "double",
            "global": False
        }
    ],
    "subscriptions": [
        {
            "key": f"f2/desired_charge_rate",
            "type": "double",
            "only_update_on_change": True
        }
    ]
}
with open("f1.json", "w") as config_file:
    json.dump(fed_json, config_file, indent=4)

helics_fed: HelicsFederate = helics.helicsCreateValueFederateFromConfig("f1.json")
fed_name = helics_fed.name

stored_energy = 100
charge_rate = 0
desired_charge_rate = 0
charge_max = 5

pub_charge_rate = helics_fed.publications["f1/charge_rate"]
pub_charge_max = helics_fed.publications["f1/charge_max"]
pub_stored_energy = helics_fed.publications["f1/stored_energy"]
sub_desired_charge_rate = helics_fed.subscriptions["f2/desired_charge_rate"]

helics_fed.enter_initializing_mode()
print("f1 entered initializing mode", flush=True)
pub_charge_rate.publish(charge_rate)
pub_charge_max.publish(charge_max)
pub_stored_energy.publish(stored_energy)

print("published initial states")

print("EV federate to enter execution mode")
helics_fed.enter_executing_mode()

prev_granted_s = 0
granted_s = 0
request_s = 0
iter_res = 5555

while granted_s < 100000:
    next_market = (granted_s // market_period + 1) * market_period
    next_p = (granted_s // period + 1) * period
    request_s = min(next_market, next_p)
    print(f"    next_market {next_market}, next_p {next_p}, request_s {request_s}")

    if granted_s % period == 0 and not isnan(charge_rate):
        stored_energy -= charge_rate * (granted_s - prev_granted_s)
        pub_stored_energy.publish(stored_energy)
        print(f"    SE: {stored_energy} (published, updated {prev_granted_s} -> {granted_s})")

    if granted_s % market_period == 0:
        print(f"    market!")

        # old_desired_charge_rate = desired_charge_rate
        desired_charge_rate = sub_desired_charge_rate.double
        print(f"    got desired charge rate {desired_charge_rate}")#, old is {old_desired_charge_rate}")
        charge_rate = desired_charge_rate
        pub_charge_rate.publish(charge_rate)
        print(f"    published charge rate {charge_rate}")

        charge_max = 5 if granted_s < 1000 else 3
        pub_charge_max.publish(charge_max)
        print(f"    published charge max {charge_max}")

    prev_granted_s = granted_s
    granted_s, iter_res = helics_fed.request_time_iterative(request_s, helics.HELICS_ITERATION_REQUEST_ITERATE_IF_NEEDED)

    granted_delta = granted_s - prev_granted_s
    print(f"REQUESTED {request_s}, granted {granted_s} | {iter_res}, delta {granted_delta}")

