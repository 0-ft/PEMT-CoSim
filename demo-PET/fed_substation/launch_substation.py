# file: launch substation.py
"""
Function:
        start running substation federate (main federate) as well as other federates
last update time: 2022-6-15
modified by Yuanliang Li
"""

import sys

from recording import SubstationRecorder

sys.path.append('..')
# this is necessary since running this file is actually opening a new process
# where my_tesp_support_api package is not inside the path list
import os
import helics
from PET_Prosumer import House, VPP  # import user-defined my_hvac class for hvac controllers
from datetime import datetime
from datetime import timedelta
from my_auction import Auction  # import user-defined my_auction class for market
from federate_helper import FEDERATE_HELPER

"""================================Declare something====================================="""
data_path = './data/exp(test)/'
if not os.path.exists(data_path):
    os.makedirs(data_path)
configfile = 'TE_Challenge_agent_dict.json'
helicsConfig = 'TE_Challenge_HELICS_substation.json'
metrics_root = 'TE_ChallengeH'
hour_stop = 48  # simulation duration (default 48 hours)
drawFigure = True  # draw figures during the simulation
has_demand_response = False
fh = FEDERATE_HELPER(configfile, helicsConfig, metrics_root, hour_stop)  # initialize the federate helper

helics_federate = helics.helicsCreateValueFederateFromConfig(helicsConfig)

"""=============================Start The Co-simulation==================================="""
# fh.create_federate()  # launch the broker; launch other federates; the substation federate enters executing mode

# initialize a user-defined VPP coordinator object
vpp = VPP(helics_federate, True)

# initialize a user-defined auction object
auction = Auction(helics_federate, fh.market_row, fh.market_key)
auction.init_auction()

# initialize House objects
houses = {}
for house_id, (key, info) in enumerate(
        fh.housesInfo_dict.items()):  # key: house name, info: information of the house, including names of PV, battery ...
    houses[key] = House(helics_federate, house_id, fh.agents['hvacs'][info['HVAC']], info['PV'], True, False,
                        # info['battery'],
                        auction,
                        house_id + 1)  # initialize a house object
    last_house_name = key

# initialize DATA_TO_PLOT class to visualize data in the simulation
num_houses = len(houses)

# initialize time parameters
StopTime = int(hour_stop * 3600)  # co-simulation stop time in seconds
StartTime = '2013-07-01 00:00:00 -0800'  # co-simulation start time
current_time = datetime.strptime(StartTime, '%Y-%m-%d %H:%M:%S %z')

dt = fh.dt  # HELCIS period (1 seconds)
update_period = houses[last_house_name].hvac.update_period  # state update period (15 seconds)
control_period = houses[last_house_name].hvac.update_period
request_period = houses[last_house_name].hvac.request_period  # local controller samples energy packet request period
market_period = auction.period  # market period (300 seconds)
adjust_period = market_period  # market response period (300 seconds)
fig_update_period = 10000  # figure update time period
tnext_update = dt  # the next time to update the state
tnext_control = control_period
tnext_request = dt  # the next time to request
tnext_lmp = market_period - dt
tnext_bid = market_period - 2 * dt  # the next time controllers calculate their final bids
tnext_agg = market_period - 2 * dt  # the next time auction calculates and publishes aggregate bid
tnext_clear = market_period  # the next time clear the market
tnext_adjust = market_period  # the next time controllers adjust control parameters/setpoints based on their bid and clearing price
tnext_fig_update = market_period + dt  # the next time to update figures

time_granted = 0
time_last = 0

print("Substation federate to enter initializing mode")
helics_federate.enter_initializing_mode()
print("Substation federate entered initializing mode")

for house_id, (key, info) in enumerate(
        fh.housesInfo_dict.items()):  # key: house name, info: information of the house, including names of PV, battery ...
    houses[key].set_meter_mode()  # set meter mode
    houses[key].hvac.set_on(False)  # at the beginning of the simulation, turn off all HVACs
    houses[key].ev.set_desired_charge_rate(0)

print("Substation federate to enter executing mode")
helics_federate.enter_executing_mode()
print("Substation federate entered executing mode")

"""============================Substation Loop=================================="""

recorder = SubstationRecorder(vpp, houses, auction)

while time_granted < StopTime:

    """ 1. step the co-simulation time """
    print("times", [tnext_update, tnext_lmp, tnext_bid, tnext_agg, tnext_clear, tnext_adjust, StopTime])
    nextHELICSTime = int(
        min([tnext_update, tnext_lmp, tnext_bid, tnext_agg, tnext_clear, tnext_adjust, StopTime]))
    time_granted = int(helics.helicsFederateRequestTime(helics_federate, nextHELICSTime))
    time_delta = time_granted - time_last
    time_last = time_granted
    current_time = current_time + timedelta(seconds=time_delta)  # this is the actual time
    print(f"substation federate requested time {nextHELICSTime}, granted {time_granted} = {current_time}")

    """ 2. houses update state/measurements for all devices, 
           update schedule and determine the power needed for hvac,
           make power predictions for solar,
           make power predictions for house load"""
    if time_granted >= tnext_update or True:
        for key, house in houses.items():
            house.update_measurements()  # update measurements for all devices
            house.hvac.change_basepoint(current_time.hour, current_time.weekday())  # update schedule
            house.hvac.determine_power_needed()  # hvac determines if power is needed based on current state
            # house.predict_solar_power()  # predict the solar power generation
        vpp.update_load()  # get the VPP load
        recorder.record_houses(current_time)
        recorder.record_vpp(current_time)
        tnext_update += update_period

    # """ 3. houses launch basic real-time control actions (not post-market control)
    #     including the control for battery"""
    # if time_granted >= tnext_control:
    #     for key, house in houses.items():
    #         if house.battery:
    #             house.battery.auto_control(
    #                 house.unresponsive_kw)  # real-time basic control of battery to track the HVAC load
    #     tnext_control += control_period

    """ 4. market gets the local marginal price (LMP) from the bulk power grid,"""
    if time_granted >= tnext_lmp:
        auction.update_lmp()  # get local marginal price (LMP) from the bulk power grid
        auction.update_refload()  # get distribution load from gridlabd
        tnext_lmp += market_period

    """ 5. houses formulate and send their bids"""
    if time_granted >= tnext_bid:
        auction.clear_bids()  # auction remove all previous records, re-initialize
        for key, house in houses.items():
            bid = house.formulate_bid()  # bid is [bid_price, quantity, hvac.power_needed, role, unres_kw, name]
            auction.collect_bid(bid)
        recorder.record_bids(current_time)
        tnext_bid += market_period

    """ 6. market aggregates bids from prosumers"""
    if time_granted >= tnext_agg:
        auction.aggregate_bids()
        auction.publish_agg_bids_for_buyer()
        tnext_agg += market_period

    """ 7. market clears the market """
    if time_granted >= tnext_clear:
        auction.clear_market(tnext_clear, time_granted)
        auction.calculate_surplus(tnext_clear, time_granted)
        auction.publish_clearing_price()
        print("Published auction clearing price: ", auction.clearing_price)
        for key, house in houses.items():
            house.publish_meter_price()
            house.post_market_control(auction.market_condition,
                                      auction.marginal_quantity)  # post-market control is needed
        time_key = str(int(tnext_clear))
        recorder.record_auction(current_time)
        tnext_clear += market_period

    """ 8. prosumer demand response (adjust control parameters/setpoints) """
    if time_granted >= tnext_adjust:
        if has_demand_response:
            for key, house in houses.items():
                house.demand_response()
        tnext_adjust += market_period

    """ 9. visualize some results during the simulation"""
    if drawFigure and time_granted >= tnext_fig_update:
        recorder.figure()
        tnext_fig_update += fig_update_period
        recorder.save("metrics.pkl")

"""============================ Finalize the metrics output ============================"""
# curves.save_statistics(data_path)
recorder.save("metrics.pkl")
print('writing metrics', flush=True)
# auction_op = open(data_path + 'auction_' + metrics_root + '_metrics.json', 'w')
# house_op = open(data_path + 'house_' + metrics_root + '_metrics.json', 'w')
# print(json.dumps(fh.auction_metrics), file=auction_op)
# print(json.dumps(fh.prosumer_metrics), file=house_op)
# auction_op.close()
# house_op.close()
helics.helicsFederateDestroy(helics_federate)
print(f"federate {helics_federate.name} has been destroyed")
# fh.destroy_federate()  # destroy the federate
# fh.show_resource_consumption()  # after simulation, print the resource consumption
# plt.show()
# fh.kill_processes(True) # it is not suggested here because some other federates may not end their simulations, it will affect their output metrics
