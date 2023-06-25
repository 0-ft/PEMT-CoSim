import json

import helics
from helics import HelicsFederate

market_period = 300
period = 26

fed_json = {
    "name": "f2",
    # "uninterruptible": True,
    "publications": [
        {
            "key": f"desired_charge_rate",
            "type": "double",
            "global": False
        }
    ],
    "subscriptions": [
        {
            "key": f"f1/charge_max",
            "type": "double",
            # "only_update_on_change": True
        },
        {
            "key": f"f1/charge_rate",
            "type": "double"
        },
        {
            "key": f"f1/stored_energy",
            "type": "double"
        }
    ]
}
with open("f2.json", "w") as config_file:
    json.dump(fed_json, config_file, indent=4)

helics_fed: HelicsFederate = helics.helicsCreateValueFederateFromConfig("f2.json")
fed_name = helics_fed.name

stored_energy = 100
desired_charge_rate = 0
charge_max = 5

pub_desired_charge_rate = helics_fed.publications["f2/desired_charge_rate"]
sub_charge_max = helics_fed.subscriptions["f1/charge_max"]
sub_charge_rate = helics_fed.subscriptions["f1/charge_rate"]
sub_stored_energy = helics_fed.subscriptions["f1/stored_energy"]

helics_fed.enter_initializing_mode()
print("f1 entered initializing mode", flush=True)
pub_desired_charge_rate.publish(desired_charge_rate)

print("published initial states")

print("EV federate to enter execution mode")
helics_fed.enter_executing_mode()

prev_granted_s = 0
granted_s = 0
next_market = 0
next_p = 0
request_s = 0
iter_res = 5555

while granted_s < 100000:

    if granted_s % period == 0:
        stored_energy = sub_stored_energy.double
        print(f"    SE: {stored_energy}")
    if granted_s % market_period == 0:
        print(f"    market!")
        old_charge_max = charge_max
        old_desired_charge_rate = desired_charge_rate
        if sub_charge_max.get_last_update_time() < granted_s:
            charge_max = float('nan')
            desired_charge_rate = float('nan')
            print(f"    NaN both")
        else:
            charge_max = sub_charge_max.double
            print(f"    got charge max {charge_max}, old is {old_charge_max}")

            desired_charge_rate = charge_max * 0.9
            print(f"    dcr {desired_charge_rate}, old is {old_desired_charge_rate}")

        # if old_desired_charge_rate != desired_charge_rate:
        pub_desired_charge_rate.publish(desired_charge_rate)
        print(f"    published desired charge rate {desired_charge_rate}")

    prev_granted_s = granted_s
    granted_s, iter_res = helics_fed.request_time_iterative(request_s, helics.HELICS_ITERATION_REQUEST_ITERATE_IF_NEEDED)

    granted_delta = granted_s - prev_granted_s
    print(f"REQUESTED {request_s}, granted {granted_s} | {iter_res}, delta {granted_delta}")

    next_market = (granted_s // market_period + 1) * market_period
    next_p = (granted_s // period + 1) * period
    request_s = min(next_market, next_p)
    print(f"    next_market {next_market}, next_p {next_p}, request_s {request_s}")
