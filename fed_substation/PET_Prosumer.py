# Copyright (C) 2017-2019 Battelle Memorial Institute
# file: hvac.py
"""Class that controls the responsive thermostat for one house.

Implements the ramp bidding method, with HVAC power as the
bid quantity, and thermostat setting changes as the response
mechanism.
"""
from datetime import timedelta

import helics
import numpy as np
from helics import HelicsFederate

from market import ContinuousDoubleAuction
from trading_policies import BoundedCrossoverTrader


# hvac_load_predictor = pickle.load(open("hvac_load_predictor.pkl", "rb"))

class HVAC:

    def __init__(self, helics_federate: HelicsFederate, house_id: int, json_object, auction: ContinuousDoubleAuction):
        self.house_id = house_id
        self.auction = auction
        self.wakeup_start = float(json_object['wakeup_start'])
        self.daylight_start = float(json_object['daylight_start'])
        self.evening_start = float(json_object['evening_start'])
        self.night_start = float(json_object['night_start'])
        self.wakeup_set = float(json_object['wakeup_set'])
        self.daylight_set = float(json_object['daylight_set'])
        self.evening_set = float(json_object['evening_set'])
        self.night_set = float(json_object['night_set'])
        self.weekend_day_start = float(json_object['weekend_day_start'])
        self.weekend_day_set = float(json_object['weekend_day_set'])
        self.weekend_night_start = float(json_object['weekend_night_start'])
        self.weekend_night_set = float(json_object['weekend_night_set'])
        self.deadband = float(json_object['deadband'])

        # state
        self.air_temp = 78.0
        self.hvac_load = 4000
        self.hvac_load_last = self.hvac_load  # save the last power measurement that > 0
        self.hvac_on = False
        self.power_needed = False
        self.predicted_load = 4000

        # setpoint related
        self.base_point = 80.6
        self.fix_basepoint = False  # if ture, the base point will not be changed all the time
        self.set_point = 0.0
        self.offset = 0

        self.probability = 0  # current request probability

        # publications and subscriptions
        self.sub_temp = helics_federate.subscriptions[f"gld1/H{house_id}#air_temperature"]
        self.sub_hvac_load = helics_federate.subscriptions[f"gld1/H{house_id}#hvac_load"]

        self.pub_thermostat_mode = helics_federate.publications[f"pet1/H{house_id}#thermostat_mode"]
        self.pub_setpoint = helics_federate.publications[f"pet1/H{house_id}#cooling_setpoint"]

    def update_state(self):
        self.air_temp = self.sub_temp.double
        # hvac load
        self.hvac_load = self.sub_hvac_load.double * 1000  # max(self.sub_hvac_load.double * 1000, 0)
        # print(f"house {self.house_id} hvac load: {self.hvac_load}, hvac_needed {self.power_needed}")

    def determine_power_needed(self, weather_temperature):
        self.set_point = self.base_point + self.offset

        up_bound = self.set_point + 1 / 2 * self.deadband
        lower_bound = self.set_point - 1 / 2 * self.deadband

        if self.air_temp > up_bound:
            self.power_needed = True
        if self.air_temp < lower_bound:
            self.power_needed = False
        self.predict_load(weather_temperature)

        # if self.air_temp < (lower_bound - 3) and self.hvac_on:
        #     print(
        #         f"WARN: HVAC {self.name}: Air temperature {self.air_temp} is lower than {lower_bound} - 3, while HVAC is on")

    # def formulate_bid_price(self):
    #     will_request = random.random() < self.probability
    #     return self.auction.lmp + 0.01 * (self.probability - 0.5)

    def predict_load(self, weather_temperature):
        self.predicted_load = self.power_needed * 4000
        # self.predicted_load = (4703
        #                        - 560 * self.set_point
        #                        + 152 * self.air_temp
        #                        + 372 * weather_temperature
        #                        ) * self.power_needed
        # self.predicted_load = hvac_load_predictor.predict(np.array([[
        #     self.set_point - self.air_temp, weather_temperature - self.air_temp, weather_temperature - self.set_point
        # ]]))[0] * self.power_needed
        # print(self.house_id, self.predicted_load)

    def set_on(self, on: bool):
        # on = self.power_needed
        # self.pub_setpoint.publish(self.set_point)
        self.pub_thermostat_mode.publish("COOL" if on else "OFF")
        self.hvac_on = on

    def change_basepoint(self, hod, dow):
        if False and dow > 4:  # a weekend
            val = self.weekend_night_set if self.weekend_day_start <= hod < self.weekend_night_start else self.weekend_day_set
        else:  # a weekday
            val = [self.night_set, self.wakeup_set, self.daylight_set, self.evening_set, self.night_set][
                sum([h < hod for h in
                     [self.wakeup_start, self.daylight_start, self.evening_start, self.night_start]])]
        if abs(self.base_point - val) > 0.1:
            self.base_point = val


class PV:
    def __init__(self, helics_federate: HelicsFederate, house_id: int):
        # measurements
        self.solar_power = 0.0
        self.solar_DC_V_out = 0.0
        self.solar_DC_I_out = 0.0
        self.desired_power = 0.0
        self.max_power = 0
        self.fixed_price = 0.014 + (np.random.uniform(-1, 1) / 1000)
        # control parameters
        self.subSolarPower = helics_federate.subscriptions[f"gld1/H{house_id}_solar_meter#measured_power"]
        # for i in range(helics.helicsFederateGetInputCount(helics_federate)):
        #     print(f"{i} INPUT: ", helics.helicsFederateGetInputByIndex(helics_federate, i))
        self.subSolarDCVOut = helics_federate.subscriptions[f"gld1/H{house_id}_solar#V_Out"]
        self.subSolarDCIOut = helics_federate.subscriptions[f"gld1/H{house_id}_solar#I_Out"]

        self.pubSolarPOut = helics_federate.publications[f"pet1/H{house_id}_solar_inv#P_Out"]
        self.pubSolarQOut = helics_federate.publications[f"pet1/H{house_id}_solar_inv#Q_Out"]

    def update_state(self):
        self.solar_power = abs(self.subSolarPower.complex.real)  # unit. kW
        self.solar_DC_V_out = self.subSolarDCVOut.double  # unit. V
        self.solar_DC_I_out = self.subSolarDCIOut.double  # unit. A
        self.max_power = self.solar_DC_V_out * self.solar_DC_I_out

    def power_range(self):
        return 0, self.max_power

    # def predicted_power(self):
    #     return self.solar_DC_V_out * self.solar_DC_I_out

    def set_power_out(self, p):
        self.desired_power = p
        self.pubSolarPOut.publish(p)
        self.pubSolarQOut.publish(0.0)


class EV:
    def __init__(self, helics_federate: HelicsFederate, house_id: int, auction: ContinuousDoubleAuction):
        self.house_id = house_id

        self.load_range = 0.0, 0.0

        self.location = "home"
        self.stored_energy = 0.0
        self.soc = 0.5
        self.desired_charge_rate = 0.0
        self.charging_load = 0.0
        self.driving_load = 0.0
        self.measured_load = 0.0

        self.sub_location = helics_federate.subscriptions[f"ev1/H{house_id}_ev#location"]
        self.sub_stored_energy = helics_federate.subscriptions[f"ev1/H{house_id}_ev#stored_energy"]
        self.sub_soc = helics_federate.subscriptions[f"ev1/H{house_id}_ev#soc"]
        self.sub_charging_load = helics_federate.subscriptions[f"ev1/H{house_id}_ev#charging_load"]
        self.sub_driving_load = helics_federate.subscriptions[f"ev1/H{house_id}_ev#driving_load"]
        self.sub_max_charging_load = helics_federate.subscriptions[f"ev1/H{house_id}_ev#max_charging_load"]
        self.sub_min_charging_load = helics_federate.subscriptions[f"ev1/H{house_id}_ev#min_charging_load"]
        self.sub_measured_load = helics_federate.subscriptions[f"gld1/H{house_id}_ev_meter#measured_power"]

        self.pub_desired_charge_rate = helics_federate.publications[f"pet1/H{house_id}_ev#charge_rate"]

    def update_state(self):
        self.location = self.sub_location.string
        self.stored_energy = self.sub_stored_energy.double
        self.soc = self.sub_soc.double
        self.charging_load = self.sub_charging_load.complex.real
        self.driving_load = self.sub_driving_load.double
        self.measured_load = abs(self.sub_measured_load.complex)
        self.load_range = self.sub_min_charging_load.double, self.sub_max_charging_load.double
        # print(f"EV {self.house_id} got subs {self.load:3f}, {self.soc:3f}, {self.location}, {self.stored_energy:3f}")

    def set_desired_charge_rate(self, desired_charge_rate):
        min_load, max_load = self.load_range
        if not (min_load <= desired_charge_rate <= max_load):
            print(
                f"WARNING: EV {self.house_id} setting desired_charge_rate {desired_charge_rate:3f}, outside range {min_load}-{max_load}")

        self.desired_charge_rate = desired_charge_rate
        self.pub_desired_charge_rate.publish(self.desired_charge_rate)
        # print(f"EV {self.house_id} published desired_charge_rate {desired_charge_rate:3f}, {self.desired_charge_rate}")

    def predicted_power(self):
        return self.desired_charge_rate

        # print(f"EV {self.house_id} at location {self.location} and stored energy {self.stored_energy}")


class House:
    def __init__(self, helics_federate: HelicsFederate, house_id: int, scenario, hvac_config, has_pv,
                 has_ev, auction: ContinuousDoubleAuction):
        self.current_time = scenario.start_time
        self.number = house_id
        self.name = f"H{house_id}"
        self.auction = auction
        self.hvac = HVAC(helics_federate, house_id, hvac_config, auction) if hvac_config else None
        print(f"house with {has_pv} PV, {has_ev} EV")
        self.pv = PV(helics_federate, house_id) if has_pv else None
        self.ev = EV(helics_federate, house_id, auction) if has_ev else None

        # measurements
        self.total_house_load = 0.0
        self.unresponsive_load = 0.0

        self.intended_load = 0.0

        self.trading_policy = BoundedCrossoverTrader(auction, timedelta(hours=0.5), timedelta(hours=24),
                                                     scenario.buy_iqr_threshold, scenario.sell_iqr_threshold)

        self.pub_meter_monthly_fee = helics_federate.publications[f"pet1/H{house_id}_meter_billing#monthly_fee"]
        self.pub_meter_mode = helics.helicsFederateGetPublication(helics_federate,
                                                                  f"H{house_id}_meter_billing#bill_mode")

        self.pub_meter_price = helics.helicsFederateGetPublication(helics_federate, f"H{house_id}_meter_billing#price")

        # self.sub_meter_voltage = helics_federate.subscriptions[f"gld1/H{house_id}_meter#measured_voltage_1"]
        self.sub_house_load = helics_federate.subscriptions[f"gld1/H{house_id}_meter_house#measured_power"]

        self.bids = []
    def set_meter_mode(self):
        self.pub_meter_mode.publish('HOURLY')
        self.pub_meter_monthly_fee.publish(0.0)

    def update_measurements(self, time):
        self.current_time = time
        # for billing meter measurements ==================
        # # billing meter voltage
        # self.mtr_voltage = self.sub_meter_voltage.complex

        # for house meter measurements ==================
        # house meter power, measured at house_F0_tpm_A(N) (excludes solar + ev)
        self.total_house_load = self.sub_house_load.double
        # print(f"{self.name} got house load {self.total_house_load}")

        # for HVAC measurements  ==================
        self.hvac.update_state()  # state update for control

        # for unresponsive load ==================
        self.unresponsive_load = self.total_house_load - self.hvac.hvac_load  # for unresponsive load

        if self.pv:
            self.pv.update_state()

        if self.ev:
            self.ev.update_state()

    def publish_meter_price(self):
        self.pub_meter_price.publish(self.auction.average_price)

    def formulate_bids(self):
        """ formulate the bid for prosumer

        Returns:
            [float, float, boolean, string, float, name]:
            bid price unit. $/kwh,
            bid quantity unit. kw,
            hvac needed,
            unresponsive load unit. kw,
            name of the house
        """
        min_pv_power, max_pv_power = self.pv.power_range() if self.pv else (0, 0)
        hvac_load = self.hvac.predicted_load

        unresponsive_bid = [(self.name, "unresponsive"), "buyer", float('inf'), self.unresponsive_load]
        bids = [unresponsive_bid]
        if hvac_load > 0:
            bids.append([(self.name, "hvac"), "buyer", 9999, hvac_load])

        if self.ev:
            ev_bids = self.trading_policy.trade(self.current_time, self.ev.load_range)
            print(f"{self.name}: {self.trading_policy.sell_threshold_price}, {self.trading_policy.long_ma} + {self.trading_policy.iqr} * {self.trading_policy.sell_iqr_ratio}")
            bids += [[(self.name, "ev")] + bid for bid in ev_bids]
        if self.pv:
            # self.pv.fixed_price = max(self.auction.lmp * 0.95, 0)
            # pv_bids = self.trading_policy.trade(self.current_time, (-max_pv_power, 0))
            pv_bid = [(self.name, "pv"), "seller", self.auction.lmp * 0.9, max_pv_power]
            if max_pv_power > 0:
                bids += [pv_bid]
        # print(f"house {self.name} bids: {bids}")
        self.bids = bids
        return bids

    def post_market_control(self, transactions):
        # print(self.name, transactions)
        buys = [bid for bid in transactions if bid["role"] == "buyer"]
        sells = [bid for bid in transactions if bid["role"] == "seller"]
        total_bought = sum(bid["quantity"] for bid in buys)
        total_sold = sum(bid["quantity"] for bid in sells)
        self.intended_load = total_bought - total_sold
        unresponsive_bought = sum([bid["quantity"] for bid in buys if bid["target"] == "unresponsive"])
        hvac_bought = sum([bid["quantity"] for bid in buys if bid["target"] == "hvac"])
        ev_bought = sum([bid["quantity"] for bid in buys if bid["target"] == "ev"])
        ev_sold = sum([bid["quantity"] for bid in sells if bid["target"] == "ev"])
        pv_sold = sum([bid["quantity"] for bid in sells if bid["target"] == "pv"])
        print(f"{self.name} ev bids {[b for b in self.bids if b[0][1] == 'ev']} bought {ev_bought} sold {ev_sold}")

        hvac_allowed = hvac_bought >= self.hvac.predicted_load
        self.hvac.set_on(self.hvac.power_needed and hvac_allowed)
        hvac_load = self.hvac.hvac_on * self.hvac.predicted_load

        if self.ev:
            self.ev.set_desired_charge_rate(ev_bought - ev_sold)
        if self.pv:
            self.pv.set_power_out(pv_sold)
        print(
            f"{self.name} bought {total_bought}, sold {total_sold}, unresponsive @ {self.unresponsive_load} HVAC on={self.hvac.hvac_on}, EV @ {ev_bought - ev_sold}, solar @ {pv_sold}")


class GridSupply:
    def __init__(self, helics_federate: HelicsFederate, auction: ContinuousDoubleAuction, power_cap: int):
        self.name = "grid"

        self.vpp_load_p = 0
        self.vpp_load_q = 0
        self.vpp_load = complex(0)
        self.power_cap = power_cap

        self.request_list = []
        self.response_list = []

        self.sub_vpp_power = helics_federate.subscriptions[f'gld1/F0_triplex_node_A#measured_power']
        self.sub_weather = helics_federate.subscriptions[f"localWeather/temperature"]
        self.sub_lmp = helics_federate.subscriptions["pypower/LMP_B7"]

        self.weather_temp = 0
        self.lmp = 0.0

        self.bid = None

    def formulate_bid(self):
        self.bid = [(self.name, "main"), "seller", self.lmp, self.power_cap]
        return self.bid

    def update_load(self):
        self.vpp_load = self.sub_vpp_power.complex
        self.vpp_load_p = self.vpp_load.real
        self.vpp_load_q = self.vpp_load.imag
        self.weather_temp = self.sub_weather.double
        self.lmp = self.sub_lmp.double

    def post_market_control(self, transactions):
        pass
