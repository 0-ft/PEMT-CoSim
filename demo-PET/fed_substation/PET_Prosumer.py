# Copyright (C) 2017-2019 Battelle Memorial Institute
# file: hvac.py
"""Class that controls the responsive thermostat for one house.

Implements the ramp bidding method, with HVAC power as the
bid quantity, and thermostat setting changes as the response
mechanism.
"""
import math
import random
from collections import deque, namedtuple
from datetime import datetime

import helics
from helics import HelicsFederate

from my_auction import Auction
from new_auction import NewAuction


# PEMBid = namedtuple("PEMBid", "bid_price num_packets hvac_power_needed role unresponsive_kw name base_covered")
# self.bid = [self.bid_price, quantity, self.hvac.power_needed, self.role, self.unresponsive_kw, self.name,
#             base_covered]


class HVAC:
    """This agent manages thermostat setpoint and bidding for a house

    Args:
        json_object (dict): dictionary row for this agent from the JSON configuration file
        key (str): name of this agent, also key for its dictionary row
        aucObj (simple_auction): the auction this agent bids into

    Attributes:
        name (str): name of this agent
        control_mode (str): control mode from dict (CN_RAMP or CN_NONE, which still implements the setpoint schedule)
        houseName (str): name of the corresponding house in GridLAB-D, from dict
        meterName (str): name of the corresponding triplex_meter in GridLAB-D, from dict
        period (float): market clearing period, in seconds, from dict
        wakeup_start (float): hour of the day (0..24) for scheduled weekday wakeup period thermostat setpoint, from dict
        daylight_start (float): hour of the day (0..24) for scheduled weekday daytime period thermostat setpoint, from dict
        evening_start (float): hour of the day (0..24) for scheduled weekday evening (return home) period thermostat setpoint, from dict
        night_start (float): hour of the day (0..24) for scheduled weekday nighttime period thermostat setpoint, from dict
        wakeup_set (float): preferred thermostat setpoint for the weekday wakeup period, in deg F, from dict
        daylight_set (float): preferred thermostat setpoint for the weekday daytime period, in deg F, from dict
        evening_set (float): preferred thermostat setpoint for the weekday evening (return home) period, in deg F, from dict
        night_set (float): preferred thermostat setpoint for the weekday nighttime period, in deg F, from dict
        weekend_day_start (float): hour of the day (0..24) for scheduled weekend daytime period thermostat setpoint, from dict
        weekend_day_set (float): preferred thermostat setpoint for the weekend daytime period, in deg F, from dict
        weekend_night_start (float): hour of the day (0..24) for scheduled weekend nighttime period thermostat setpoint, from dict
        weekend_night_set (float): preferred thermostat setpoint for the weekend nighttime period, in deg F, from dict
        deadband (float): thermostat deadband in deg F, invariant, from dict
        offset_limit (float): maximum allowed change from the time-scheduled setpoint, in deg F, from dict
        ramp (float): bidding ramp denominator in multiples of the price standard deviation, from dict
        price_cap (float): the highest allowed bid price in $/kwh, from dict
        bid_delay (float): from dict, not implemented
        use_predictive_bidding (float): from dict, not implemented
        std_dev (float): standard deviation of expected price, determines the bidding ramp slope, initialized from aucObj
        mean (float): mean of the expected price, determines the bidding ramp origin, initialized from aucObj
        Trange (float): the allowed range of setpoint variation, bracketing the preferred time-scheduled setpoint
        air_temp (float): current air temperature of the house in deg F
        hvac_kw (float): most recent non-zero HVAC power in kW, this will be the bid quantity
        mtr_v (float): current line-neutral voltage at the triplex meter
        hvac_on (Boolean): True if the house HVAC is currently running
        base_point (float): the preferred time-scheduled thermostat setpoint in deg F
        set_point (float): the thermostat setpoint, including price response, in deg F
        bid_price (float): the current bid price in $/kwh
        cleared_price (float): the cleared market price in $/kwh
    """

    def __init__(self, helics_federate: HelicsFederate, house_id: int, json_object, auction: NewAuction):
        self.auction = auction
        self.loadType = 'hvac'
        self.control_mode = json_object['control_mode']
        self.houseName = json_object['houseName']
        self.meterName = json_object['meterName']

        self.period = float(json_object['period'])
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
        self.offset_limit = float(json_object['offset_limit'])
        self.ramp = float(json_object['ramp'])
        self.price_cap = float(json_object['price_cap'])
        self.bid_delay = float(json_object['bid_delay'])
        self.use_predictive_bidding = float(json_object['use_predictive_bidding'])

        # state
        self.air_temp = 78.0
        self.hvac_load = 4000
        self.hvac_load_last = self.hvac_load  # save the last power measurement that > 0
        self.hvac_on = False
        self.power_needed = False

        # setpoint related
        self.base_point = 80.6
        self.fix_basepoint = False  # if ture, the base point will not be changed all the time
        self.set_point = 0.0
        self.offset = 0
        self.Trange = abs(2.0 * self.offset_limit)

        # PEM related
        self.update_period = 15  # the time interval that the energy meter measures its load, update state
        self.energy_market = 0  # energy consumed within market period
        self.energy_cumulated = 0  # energy consumed in the co-simulation
        self.energyMarket_window = deque(
            maxlen=3)  # this windows saves latest n energy consumptions during the market period
        self.request_period = 3  # the time interval that a load generate its request according to its probability based on its internal state
        self.energy_packet_length = 1 * 60  # the energy packet total length when the request is accepted
        self.energy_packet_length_now = 0  # current energy packet length (unit: s)
        self.MTTR_base = 60  # initial mean time to request (equal to the packet length)
        self.MTTR_now = self.MTTR_base  # dynamic mean time to request
        self.MTTR_lower = self.MTTR_base * 1 / 2  # lower bound for dynamic MTTR
        self.MTTR_upper = self.MTTR_base * 10  # upper bound for dynamic MTTR

        self.packet_delivered = True  # True if the latest packer has been delivered, False if the packet is on delivery
        self.probability = 0  # current request probability
        self.response_strategy = "setpoint"  # "setpoint" means price response by adjusting the setpoint according to the cleared price
        # "mttr" means price response by adjusting the mean time to request (MTTR)

        # publications and subscriptions
        self.sub_temp = helics_federate.subscriptions[f"gld1/F0_house_A{house_id}#air_temperature"]
        self.sub_hvac_load = helics_federate.subscriptions[f"gld1/F0_house_A{house_id}#hvac_load"]

        self.pub_thermostat_mode = helics_federate.publications[f"sub1/F0_house_A{house_id}_hvac/thermostat_mode"]
        # self.pubThermostatDeadband = helics.helicsFederateRegisterPublication(helics_federate,
        #                                                                   f"F0_house_A{house_id}_hvac/thermostat_mode",
        #                                                                   helics.helics_data_type_string)
        # self.

    def update_state(self):
        self.air_temp = self.sub_temp.double
        # hvac load
        self.hvac_load = max(self.sub_hvac_load.double * 1000, 0)
        # update request probability
        self.update_request_probability()

    def determine_power_needed(self):
        self.set_point = self.base_point + self.offset

        up_bound = self.set_point + 1 / 2 * self.deadband
        lower_bound = self.set_point - 1 / 2 * self.deadband

        if self.air_temp > up_bound:
            self.power_needed = True
        if self.air_temp < lower_bound:
            self.power_needed = False

        # if self.air_temp < (lower_bound - 3) and self.hvac_on:
        #     print(
        #         f"WARN: HVAC {self.name}: Air temperature {self.air_temp} is lower than {lower_bound} - 3, while HVAC is on")

    def predicted_load(self):
        return self.power_needed * 4000

    def auto_control(self):
        self.determine_power_needed()
        self.set_on(self.power_needed)

    def update_request_probability(self):
        """
        Get current request probability for cooling system
        """
        self.set_point = self.base_point + self.offset

        up_bound = self.set_point + 1 / 2 * self.deadband
        lower_bound = self.set_point - 1 / 2 * self.deadband

        mr = 1 / self.MTTR_now
        if self.air_temp >= up_bound:
            mu = float('inf')
        elif self.air_temp <= lower_bound:
            mu = 0
        else:
            mu = mr * (self.air_temp - lower_bound) / (up_bound - self.air_temp) * (up_bound - self.set_point) / (
                    self.set_point - lower_bound)

        self.probability = 1 - math.exp(-mu * self.request_period)

    def set_on(self, on: bool):
        self.pub_thermostat_mode.publish("COOL" if on else "OFF")
        self.hvac_on = on

    def update_energy_market(self):
        d_energy = self.hvac_load * self.update_period / 3600
        self.energy_market += d_energy
        self.energy_cumulated += d_energy

    def clean_energy_market(self):
        self.energy_market = 0

    def price_response(self, price):
        """ Update the thermostat setting if the last bid was accepted
        The last bid is always "accepted". If it wasn't high enough,
        then the thermostat could be turned up.
        """
        if self.response_strategy == 'setpoint':
            self.offset = 0
            # if self.control_mode == 'CN_RAMP' and self.auction.std_dev > 0.0:
            #     self.offset = (price - self.auction.clearing_price) * self.Trange / self.ramp / self.auction.std_dev
            #     self.offset = max(-self.offset_limit, min(self.offset, self.offset_limit))
        if self.response_strategy == 'mttr':
            price_upper = 0.5
            self.MTTR_now = (price - price_upper) / (self.auction.clearing_price - price_upper) * (
                    self.MTTR_base - self.MTTR_upper) + self.MTTR_upper

            self.MTTR_now = max(self.MTTR_lower, min(self.MTTR_now, self.MTTR_upper))

    def change_basepoint(self, hod, dow):
        """ Updates the time-scheduled thermostat setting

        Args:
            hod (float): the hour of the day, from 0 to 24
            dow (int): the day of the week, zero being Monday

        Returns:
            Boolean: True if the setting changed, Falso if not
        """
        if not self.fix_basepoint:
            if dow > 4:  # a weekend
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

        # control parameters
        self.subSolarPower = helics_federate.subscriptions[f"gld1/solar_F0_tpm_A{house_id}#measured_power"]
        # for i in range(helics.helicsFederateGetInputCount(helics_federate)):
        #     print(f"{i} INPUT: ", helics.helicsFederateGetInputByIndex(helics_federate, i))
        self.subSolarDCVOut = helics_federate.subscriptions[f"gld1/solar_F0_house_A{house_id}#V_Out"]
        self.subSolarDCIOut = helics_federate.subscriptions[f"gld1/solar_F0_house_A{house_id}#I_Out"]

        self.pubSolarPOut = helics_federate.publications[f"sub1/solar_inv_F0_house_A{house_id}/P_Out"]
        self.pubSolarQOut = helics_federate.publications[f"sub1/solar_inv_F0_house_A{house_id}/Q_Out"]

    def update_state(self):
        self.solar_power = abs(self.subSolarPower.complex.real)  # unit. kW
        self.solar_DC_V_out = self.subSolarDCVOut.complex.real  # unit. V
        self.solar_DC_I_out = self.subSolarDCIOut.complex.real  # unit. A

    def power_range(self):
        return 0, self.solar_DC_V_out * self.solar_DC_I_out

    # def predicted_power(self):
    #     return self.solar_DC_V_out * self.solar_DC_I_out

    def set_power_out(self, p):
        self.desired_power = p
        self.pubSolarPOut.publish(p)
        self.pubSolarQOut.publish(0.0)


class BATTERY:
    def __init__(self, helics_federate: HelicsFederate, house_id: int):
        # measurements
        self.power = 0
        self.soc = 0.5

        # control parameters
        self.charge_on_threshold = 1500
        self.charge_off_threshold = 1700
        self.discharge_on_threshold = 3000
        self.discharge_off_threshold = 2000

        self.charge_on_threshold_offset = 0
        self.charge_off_threshold_offset = 200
        self.discharge_on_threshold_offset = 3000
        self.discharge_off_threshold_offset = 2500

        self.sub_power = helics_federate.subscriptions[f"gld1/batt_F0_tpm_A{house_id}#measured_power"]
        self.sub_soc = helics_federate.subscriptions[f"gld1/batt_F0_house_A{house_id}#state_of_charge"]

        self.pub_charge_on_threshold = helics_federate.publications[
            f"sub1/batt_inv_F0_house_A{house_id}/charge_on_threshold"]
        self.pub_charge_off_threshold = helics_federate.publications[
            f"sub1/batt_inv_F0_house_A{house_id}/charge_off_threshold"]
        self.pub_discharge_on_threshold = helics_federate.publications[
            f"sub1/batt_inv_F0_house_A{house_id}/disharge_off_threshold"]
        self.pub_discharge_off_threshold = helics_federate.publications[
            f"sub1/batt_inv_F0_house_A{house_id}/disharge_on_threshold"]

    def update_state(self):
        self.power = self.sub_power.complex.real
        self.soc = self.sub_soc.double

    def auto_control(self, unresponsive_kw):
        # unresponsive_kw is the base load
        # charge when load drops below unresponsive load
        self.charge_on_threshold = unresponsive_kw * 1000 + self.charge_on_threshold_offset

        # stop charging when load is coming up to unresponsive load
        self.charge_off_threshold = unresponsive_kw * 1000 + self.charge_off_threshold_offset

        # discharge when the load is above unresponsive load + 3000
        self.discharge_on_threshold = unresponsive_kw * 1000 + self.discharge_on_threshold_offset

        # stop discharging when the load is below unresponsive load + 2500
        self.discharge_off_threshold = unresponsive_kw * 1000 + self.discharge_off_threshold_offset

        self.pub_charge_on_threshold.publish(self.charge_on_threshold)
        self.pub_charge_off_threshold.publish(self.charge_off_threshold)
        self.pub_discharge_on_threshold.publish(self.discharge_on_threshold)
        self.pub_discharge_off_threshold.publish(self.discharge_off_threshold)


class EV:
    def __init__(self, helics_federate: HelicsFederate, house_id: int):
        self.current_time = datetime.min
        self.house_id = house_id

        self.max_discharge_rate = 4000
        self.max_charge_rate = 4000
        self.min_charge_rate = 2500
        self.discharge_hod_ranges = [(0.0, 3.0), (18.0, 24.0)]

        self.location = "home"
        self.stored_energy = 0.0
        self.soc = 0.5
        self.desired_charge_rate = 0.0
        self.charging_load = 0.0
        self.driving_load = 0.0
        self.sub_location = helics_federate.subscriptions[f"ev1/F0_house_A{house_id}_EV/location"]
        self.sub_stored_energy = helics_federate.subscriptions[f"ev1/F0_house_A{house_id}_EV/stored_energy"]
        self.sub_soc = helics_federate.subscriptions[f"ev1/F0_house_A{house_id}_EV/soc"]
        self.sub_charging_load = helics_federate.subscriptions[f"ev1/F0_house_A{house_id}_EV/charging_load"]
        self.sub_driving_load = helics_federate.subscriptions[f"ev1/F0_house_A{house_id}_EV/driving_load"]

        self.pub_desired_charge_rate = helics.helicsFederateGetPublication(helics_federate,
                                                                           f"F0_house_A{house_id}_EV/charge_rate")

    def update_state(self, time):
        self.current_time = time
        self.location = self.sub_location.string
        self.stored_energy = self.sub_stored_energy.double
        self.soc = self.sub_soc.double
        self.charging_load = self.sub_charging_load.complex.real
        self.driving_load = self.sub_driving_load.double
        # print(f"EV {self.house_id} got subs {self.load:3f}, {self.soc:3f}, {self.location}, {self.stored_energy:3f}")

    def load_range(self):
        max_discharge = -self.max_discharge_rate * any(
            a <= (self.current_time.hour + self.current_time.minute / 60.0) < b for a, b in self.discharge_hod_ranges)
        if self.location != "home":
            return 0.0, 0.0
        if .9 <= self.soc:
            return max_discharge, 0
        elif .3 <= self.soc < .9:
            return max_discharge, self.max_charge_rate
        elif .2 <= self.soc < .3:
            return 0, self.max_charge_rate
        else:
            return self.min_charge_rate, self.max_charge_rate

    def set_desired_charge_rate(self, desired_charge_rate):
        min_load, max_load = self.load_range()
        if not min_load <= desired_charge_rate <= max_load:
            print(
                f"WARNING: EV {self.house_id} setting desired_charge_rate {desired_charge_rate:3f}, outside range {min_load}-{max_load}")

        self.desired_charge_rate = desired_charge_rate
        self.pub_desired_charge_rate.publish(self.desired_charge_rate)
        # print(f"EV {self.house_id} published desired_charge_rate {desired_charge_rate:3f}, {self.desired_charge_rate}")

    def predicted_power(self):
        return self.desired_charge_rate

        # print(f"EV {self.house_id} at location {self.location} and stored energy {self.stored_energy}")


class House:
    def __init__(self, helics_federate: HelicsFederate, house_id: int, hvac_config, pv_config, ev_config,
                 battery_config, auction: NewAuction, seed):
        self.number = house_id
        self.name = f"F0_house_A{house_id}"
        self.auction = auction
        self.hvac = HVAC(helics_federate, house_id, hvac_config, auction) if hvac_config else None
        self.pv = PV(helics_federate, house_id) if pv_config else None
        self.ev = EV(helics_federate, house_id) if ev_config else None
        self.battery = BATTERY(helics_federate, house_id) if battery_config else None
        self.role = 'buyer'  # current role: buyer/sellehouse_F0_tpm_r/none-participant

        # market price related
        self.bid = []
        self.bid_price = 0.0
        random.seed(seed)
        self.fixed_seller_price = random.uniform(0.01, 0.015)

        # measurements
        self.mtr_voltage = 120.0
        self.mtr_power = 0.0
        self.total_house_load = 0.0
        self.unresponsive_load = 0.0

        # about packet
        self.packet_unit = 1000

        self.pub_meter_mode = helics.helicsFederateGetPublication(helics_federate, f"F0_tpm_A{house_id}/bill_mode")

        self.pub_meter_monthly_fee = helics_federate.publications[f"sub1/F0_tpm_A{house_id}/monthly_fee"]
        self.pub_meter_price = helics.helicsFederateGetPublication(helics_federate, f"F0_tpm_A{house_id}/price")

        self.sub_meter_voltage = helics_federate.subscriptions[f"gld1/F0_tpm_A{house_id}#measured_voltage_1"]
        self.sub_meter_power = helics_federate.subscriptions[f"gld1/F0_tpm_A{house_id}#measured_power"]
        self.sub_house_load = helics_federate.subscriptions[f"gld1/house_F0_tpm_A{house_id}#measured_power"]

    def set_meter_mode(self):
        self.pub_meter_mode.publish('HOURLY')
        self.pub_meter_monthly_fee.publish(0.0)

    def update_measurements(self, time):
        # for billing meter measurements ==================
        # billing meter voltage
        self.mtr_voltage = abs(self.sub_meter_voltage.complex)
        # billing meter power
        self.mtr_power = self.sub_meter_power.complex.real

        # for house meter measurements ==================
        # house meter power, measured at house_F0_tpm_A(N) (excludes solar + ev)
        self.total_house_load = abs(self.sub_house_load.complex)
        # print(f"{self.name} got house load {self.total_house_load}")

        # for HVAC measurements  ==================
        self.hvac.update_state()  # state update for control
        self.hvac.update_energy_market()  # measured the energy consumed during the market period

        # for unresponsive load ==================
        self.unresponsive_load = self.total_house_load - self.hvac.hvac_load  # for unresponsive load

        if self.pv:
            self.pv.update_state()

        if self.battery:
            self.battery.update_state()

        if self.ev:
            self.ev.update_state(time)

    def real_time_control(self):
        self.pv.set_power_out(self.pv.power_range()[1])
        self.hvac.auto_control()
        if self.total_house_load < self.pv.solar_power:
            # print(
            #     f"EV @ {self.name} setting to diff {self.pv.solar_power} - {self.total_house_load} = {self.pv.solar_power - self.total_house_load}")
            self.ev.set_desired_charge_rate(min(self.pv.solar_power - self.total_house_load, self.ev.load_range()[1]))

    # def predicted_load(self):
    #     return self.unresponsive_kw

    def publish_meter_price(self):
        self.pub_meter_price.publish(self.auction.clearing_price)

    def formulate_bid(self):
        """ formulate the bid for prosumer

        Returns:
            [float, float, boolean, string, float, name]:
            bid price unit. $/kwh,
            bid quantity unit. kw,
            hvac needed,
            unresponsive load unit. kw,
            name of the house
        """
        self.bid.clear()
        min_ev_load, max_ev_load = self.ev.load_range()
        min_pv_power, max_pv_power = self.pv.power_range() if self.pv else (0, 0)
        hvac_load = self.hvac.predicted_load()

        if max_pv_power >= self.unresponsive_load + hvac_load:
            surplus_power = max_pv_power - (self.unresponsive_load + hvac_load + max_ev_load)
            if surplus_power > self.packet_unit:
                self.role = 'seller'
                quantity = (surplus_power // self.packet_unit) * self.packet_unit
                self.bid_price = self.fixed_seller_price
                base_covered = True
            elif surplus_power >= 0:
                self.role = 'none-participant'
                self.bid_price = 0
                quantity = 0
                base_covered = True
            else:
                self.role = 'none-participant'
                self.bid_price = 0
                quantity = 0
                base_covered = True
        elif max_pv_power == self.unresponsive_load + hvac_load:
            self.role = 'none-participant'
            self.bid_price = 0
            quantity = 0
            base_covered = True
        else:  # if PV can't cover base + hvac
            if max_pv_power - min_ev_load >= self.unresponsive_load + hvac_load:
                self.role = 'none-participant'
                self.bid_price = 0
                quantity = 0
                base_covered = True
                # print(f"{self.name} branch 1, {max_pv_power} + {-min_ev_load} covering {self.unresponsive_load} + {hvac_load}")
            else:
                self.role = 'buyer'
                p = self.auction.clearing_price + ((self.hvac.air_temp - self.hvac.base_point) / self.hvac.deadband) * (
                        self.auction.lmp - self.auction.clearing_price)

                self.bid_price = min(self.hvac.price_cap, max(p, 0.0))
                packets_needed = math.ceil(
                    (self.unresponsive_load + hvac_load - (max_pv_power - min_ev_load)) / self.packet_unit)
                quantity = packets_needed * self.packet_unit
                base_covered = max_pv_power - min_ev_load > self.unresponsive_load
                # print(f"{self.name} branch 2, {max_pv_power} + {-min_ev_load} + {packets_needed} packets ({quantity}) covering {self.unresponsive_load} + {hvac_load}")

        self.bid = [self.bid_price, quantity, self.hvac.power_needed, self.role, self.unresponsive_load, self.name,
                base_covered]

        # print(f"{self.name} bidding {self     .bid}, ev={min_ev_load}-{max_ev_load}, pv={min_pv_power}-{max_pv_power}")

        return self.bid

    def demand_response(self):
        self.hvac.price_response(self.auction.clearing_price)

    def post_market_control(self, auction_response):
        role, quantity = auction_response[["role", "quantity"]].values

        # bid_price = self.bid[0]
        # quantity = self.bid[1]
        hvac_power_needed = self.bid[2]
        # role = self.bid[3]
        unresponsive_power = self.bid[4]
        # base_covered = self.bid[6]

        min_ev_load, max_ev_load = self.ev.load_range()
        min_pv_power, max_pv_power = self.pv.power_range() if self.pv else (0, 0)
        hvac_load = self.hvac.predicted_load()

        if role == 'seller':
            # PV control
            self.pv.set_power_out(unresponsive_power + hvac_load + max_ev_load + quantity)
            # if market_condition == 'double-auction':
            #     if bid_price > self.auction.clearing_price:  # rejected
            #         self.pv.set_power_out(unresponsive_power + hvac_load + max_ev_load)
            #     if bid_price < self.auction.clearing_price:  # accepted
            #         self.pv.set_power_out(unresponsive_power + hvac_load + max_ev_load + quantity)
            #     elif bid_price == self.auction.clearing_price:  # some accepted
            #         self.pv.set_power_out(unresponsive_power + hvac_load + max_ev_load + marginal_quantity)
            # else:  # flexible-load (impossible)
            #     print("Invalid seller in flexible-load condition!!")

            # hvac control
            self.hvac.set_on(hvac_power_needed)
            self.ev.set_desired_charge_rate(max_ev_load)
        elif role == "none-participant":
            if max_pv_power >= unresponsive_power + hvac_load + max_ev_load:
                self.hvac.set_on(hvac_power_needed)
                self.ev.set_desired_charge_rate(max_ev_load)
                self.pv.set_power_out(min(max_pv_power, unresponsive_power + hvac_load + max_ev_load))
            elif max_pv_power > unresponsive_power + hvac_load:
                self.hvac.set_on(hvac_power_needed)
                self.pv.set_power_out(max_pv_power)
                self.ev.set_desired_charge_rate(max_pv_power - unresponsive_power - hvac_load)
            elif max_pv_power == unresponsive_power + hvac_load:
                self.pv.set_power_out(unresponsive_power + hvac_load)
                self.ev.set_desired_charge_rate(max(min_ev_load, 0.0))
                # self.ev.set_desired_charge_rate(min_ev_load)
                self.hvac.set_on(hvac_power_needed)
            else:  # max_pv_power < base + hvac
                self.pv.set_power_out(max_pv_power)
                self.ev.set_desired_charge_rate(max_pv_power - unresponsive_power - hvac_load)
                self.hvac.set_on(hvac_power_needed)
        elif role == 'buyer':
            # hvac control and PV control
            power_available = max_pv_power - min_ev_load + quantity
            self.pv.set_power_out(max_pv_power)
            self.ev.set_desired_charge_rate(min_ev_load)
            # self.ev.set_desired_charge_rate(-(unresponsive_power + hvac_load - max_pv_power - quantity))
            self.hvac.set_on(power_available >= unresponsive_power + hvac_load)

            # self.pv.set_power_out(max_pv_power)
            # self.ev.set_desired_charge_rate(min_ev_load)
            # if bid_price >= self.auction.clearing_price:  # accepted
            #     self.pv.set_power_out(max_pv_power)
            #     self.ev.set_desired_charge_rate(min_ev_load)
            #     self.hvac.set_on(hvac_power_needed)
            # else:  # rejected
            #     self.pv.set_power_out(max_pv_power)
            #     self.ev.set_desired_charge_rate(max(max_pv_power - unresponsive_power - hvac_load, min_ev_load))
            #     self.hvac.set_on(hvac_power_needed)
        else:
            print(f"unknown role {role}")
        # print(f"PMC {self.name} set EV={self.ev.desired_charge_rate}, HVAC={self.hvac.hvac_on}, PV={self.pv.p_out}")
        # if hvac_power_needed:
        #     if base_covered:
        #         if bid_price >= self.auction.clearing_price:  # accepted
        #             self.hvac.set_on(True)
        #             if self.pv:
        #                 self.pv.set_power_out(unres_kw + max(3.0 - quantity, 0), 0)
        #         else:
        #             self.hvac.set_on(False)
        #             if self.pv:
        #                 self.pv.set_power_out(unres_kw, 0)
        #     else:
        #         if bid_price >= self.auction.clearing_price:
        #             self.hvac.set_on(True)
        #             if self.pv:
        #                 self.pv.set_power_out(0, 0)
        #         else:
        #             self.hvac.set_on(False)
        #             if self.pv:
        #                 self.pv.set_power_out(0, 0)
        # else:
        #     self.hvac.set_on(False)
        #     if self.pv:
        #         self.pv.set_power_out(0, 0)
        # if role == 'none-participant':
        #     # PV control
        #     if self.pv:
        #         self.pv.set_power_out(unresponsive_power + 3 * int(hvac_power_needed), 0)
        #     # hvac control
        #     self.hvac.set_on(hvac_power_needed)


class VPP:
    def __init__(self, helics_federate: HelicsFederate, auction: NewAuction, enable=True):
        self.enable = enable

        self.vpp_load_p = 0
        self.vpp_load_q = 0
        self.vpp_load = complex(0)
        self.balance_signal = 220  # kVA

        self.request_list = []
        self.response_list = []

        self.sub_vpp_power = helics_federate.subscriptions[f'gld1/F0_triplex_node_A#measured_power']
        self.sub_weather = helics_federate.subscriptions[f"localWeather/temperature"]
        self.sub_lmp = helics_federate.subscriptions("pypower/LMP_B7")

        self.weather_temp = 0
        self.lmp = 0.0

        self.bid = None

    def formulate_bid(self):
        self.bid = [self.lmp, float('inf'), False, "seller", 0, "substation", False]
        return self.bid
        
    def receive_request(self, request):
        # request is a list [name, load type, power, length]
        if len(request) > 0:
            self.request_list.append(request)

    def aggregate_requests(self):
        self.update_load()
        self.update_balance_signal()
        self.response_list.clear()
        self.response_list = self.request_list.copy()  # copy messages

        if len(self.response_list) > 0:
            if not self.enable:  # if VPP coordinator is not enable, all requests will be accepted
                for i in range(len(self.response_list)):
                    self.response_list[i]['response'] = 'YES'
            else:
                arrive_idx_list = list(range(len(self.response_list)))
                random.shuffle(arrive_idx_list)  # randomize the arrive time
                load_est = self.vpp_load_p

                for idx in arrive_idx_list:
                    response = self.response_list[idx]
                    key = response['name']
                    load = response['power']
                    length = response['packet-length']
                    load_est += load
                    if (self.balance_signal - load_est) >= 0:
                        self.response_list[idx]['response'] = 'YES'
                    else:
                        self.response_list[idx]['response'] = 'NO'
        self.request_list.clear()

    def update_load(self):
        self.vpp_load = self.sub_vpp_power.complex
        self.vpp_load_p = self.vpp_load.real
        self.vpp_load_q = self.vpp_load.imag
        self.weather_temp = self.sub_weather.double
        self.lmp = self.sub_lmp.double

    def update_balance_signal(self):
        pass
