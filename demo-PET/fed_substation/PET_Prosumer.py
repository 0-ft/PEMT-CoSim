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

import helics
from helics import HelicsFederate

from my_auction import Auction


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
        basepoint (float): the preferred time-scheduled thermostat setpoint in deg F
        setpoint (float): the thermostat setpoint, including price response, in deg F
        bid_price (float): the current bid price in $/kwh
        cleared_price (float): the cleared market price in $/kwh
    """

    def __init__(self, helics_federate: HelicsFederate, house_id: int, name, json_object, auction: Auction):
        self.name = name
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
        self.hvac_kw = 3.0
        self.hvac_kv_last = self.hvac_kw  # save the last power measurement that > 0
        self.hvac_on = False
        self.power_needed = False

        # setpoint related
        self.basepoint = 80.6
        self.fix_basepoint = False  # if ture, the base point will not be changed all the time
        self.set_point = 0.0
        self.offset = 0
        self.Trange = abs(2.0 * self.offset_limit)

        # PEM related
        self.update_period = 1  # the time interval that the energy meter measures its load, update state
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
        self.subTemp = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                 f"gld1/F0_house_A{house_id}#air_temperature")
        self.subHVACLoad = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                     f"gld1/F0_house_A{house_id}#hvac_load")

        self.pubThermostatMode = helics.helicsFederateRegisterPublication(helics_federate,
                                                                          f"F0_house_A{house_id}_hvac/thermostat_mode",
                                                                          helics.helics_data_type_string)
        # self.pubThermostatDeadband = helics.helicsFederateRegisterPublication(helics_federate,
        #                                                                   f"F0_house_A{house_id}_hvac/thermostat_mode",
        #                                                                   helics.helics_data_type_string)
        # self.

    def update_state(self):
        self.air_temp = helics.helicsInputGetDouble(self.subTemp)
        # hvac load
        self.hvac_kw = max(helics.helicsInputGetDouble(self.subHVACLoad), 0)  # unit kW
        # update request probability
        self.update_request_probability()

    def determine_power_needed(self):
        self.set_point = self.basepoint + self.offset

        up_bound = self.set_point + 1 / 2 * self.deadband
        lower_bound = self.set_point - 1 / 2 * self.deadband

        if self.air_temp > up_bound:
            self.power_needed = True
        if self.air_temp < lower_bound:
            self.power_needed = False

        # if self.air_temp < (lower_bound - 3) and self.hvac_on:
        #     print(
        #         f"WARN: HVAC {self.name}: Air temperature {self.air_temp} is lower than {lower_bound} - 3, while HVAC is on")

    def auto_control(self):
        self.determine_power_needed()
        self.set_on(self.power_needed)

    def update_request_probability(self):
        """
        Get current request probability for cooling system
        """
        self.set_point = self.basepoint + self.offset

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
        helics.helicsPublicationPublishString(self.pubThermostatMode, "COOL" if on else "OFF")
        self.hvac_on = on

    def update_energy_market(self):
        dEnergy = self.hvac_kw * self.update_period / 3600
        self.energy_market += dEnergy
        self.energy_cumulated += dEnergy

    def clean_energy_market(self):
        self.energy_market = 0

    def price_response(self, price):
        """ Update the thermostat setting if the last bid was accepted
        The last bid is always "accepted". If it wasn't high enough,
        then the thermostat could be turned up.
        """
        if self.response_strategy == 'setpoint':
            if self.control_mode == 'CN_RAMP' and self.auction.std_dev > 0.0:
                self.offset = (price - self.auction.clearing_price) * self.Trange / self.ramp / self.auction.std_dev
                self.offset = max(-self.offset_limit, min(self.offset, self.offset_limit))
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
            if abs(self.basepoint - val) > 0.1:
                self.basepoint = val


class PV:
    def __init__(self, helics_federate: HelicsFederate, house_id: int, name: str):
        self.name = name

        # measurements
        self.solar_kw = 0
        self.solar_DC_V_out = 0
        self.solar_DC_I_out = 0

        # control parameters
        self.subSolarPower = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                       f"gld1/solar_F0_tpm_A{house_id}#measured_power")
        self.subSolarDCVOut = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                        f"gld1/solar_F0_house_A{house_id}#V_out")
        self.subSolarDCIOut = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                        f"gld1/solar_F0_house_A{house_id}#I_out")

        self.pubSolarPOut = helics.helicsFederateRegisterPublication(helics_federate,
                                                                     f"solar_inv_F0_house_A{house_id}/P_out",
                                                                     helics.helics_data_type_double)
        self.pubSolarQOut = helics.helicsFederateRegisterPublication(helics_federate,
                                                                     f"solar_inv_F0_house_A{house_id}/Q_out",
                                                                     helics.helics_data_type_double)

    def update_state(self):
        self.solar_kw = abs(helics.helicsInputGetComplex(self.subSolarPower).real * 0.001)  # unit. kW
        self.solar_DC_V_out = helics.helicsInputGetComplex(self.subSolarDCVOut).real  # unit. V
        self.solar_DC_I_out = helics.helicsInputGetComplex(self.subSolarDCIOut).real  # unit. A

    def set_power_out(self, P, Q):
        helics.helicsPublicationPublishDouble(self.pubSolarPOut, P * 1000)
        helics.helicsPublicationPublishDouble(self.pubSolarQOut, Q * 1000)


class BATTERY:
    def __init__(self, helics_federate: HelicsFederate, house_id: int, name):
        self.name = name

        # measurements
        self.battery_kw = 0
        self.battery_soc = 0.5

        # control parameters
        self.charge_on_threshold = 1500
        self.charge_off_threshold = 1700
        self.discharge_on_threshold = 3000
        self.discharge_off_threshold = 2000
        self.charge_on_threshold_offset = 0
        self.charge_off_threshold_offset = 200
        self.discharge_on_threshold_offset = 3000
        self.discharge_off_threshold_offset = 2500

        self.subBattPower = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                      f"gld1/batt_F0_tpm_A{house_id}#measured_power")
        self.subBattSoC = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                    f"gld1/batt_F0_house_A{house_id}#state_of_charge")

        self.pubChargeOnThreshold = helics.helicsFederateRegisterPublication(helics_federate,
                                                                             f"batt_inv_F0_house_A{house_id}/charge_on_threshold",
                                                                             helics.helics_data_type_double)
        self.pubChargeOffThreshold = helics.helicsFederateRegisterPublication(helics_federate,
                                                                              f"batt_inv_F0_house_A{house_id}/charge_off_threshold",
                                                                              helics.helics_data_type_double)
        self.pubDischargeOnThreshold = helics.helicsFederateRegisterPublication(helics_federate,
                                                                                f"batt_inv_F0_house_A{house_id}/disharge_off_threshold",
                                                                                helics.helics_data_type_double)
        self.pubDischargeOffThreshold = helics.helicsFederateRegisterPublication(helics_federate,
                                                                                 f"batt_inv_F0_house_A{house_id}/disharge_on_threshold",
                                                                                 helics.helics_data_type_double)

    def update_state(self):
        self.battery_kw = helics.helicsInputGetComplex(self.subBattPower).real * 0.001  # unit. kW
        self.battery_soc = helics.helicsInputGetDouble(self.subBattSoC)

    def auto_control(self, unresponsive_kw):
        self.charge_on_threshold = unresponsive_kw * 1000 + self.charge_on_threshold_offset
        self.charge_off_threshold = unresponsive_kw * 1000 + self.charge_off_threshold_offset
        self.discharge_on_threshold = unresponsive_kw * 1000 + self.discharge_on_threshold_offset
        self.discharge_off_threshold = unresponsive_kw * 1000 + self.discharge_off_threshold_offset

        helics.helicsPublicationPublishDouble(self.pubChargeOnThreshold, self.charge_on_threshold)
        helics.helicsPublicationPublishDouble(self.pubChargeOffThreshold, self.charge_off_threshold)
        helics.helicsPublicationPublishDouble(self.pubDischargeOnThreshold, self.discharge_on_threshold)
        helics.helicsPublicationPublishDouble(self.pubDischargeOffThreshold, self.discharge_off_threshold)


class EV:
    def __init__(self, helics_federate: HelicsFederate, house_id: int):
        self.location = "home"
        self.stored_energy = 0
        self.charge_rate = 0
        self.sub_location = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                      f"ev1/F0_house_A{house_id}_EV#location")
        self.sub_stored_energy = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                           f"ev1/F0_house_A{house_id}_EV#stored_energy")
        self.sub_load = helics.helicsFederateRegisterSubscription(helics_federate, f"ev1/F0_house_A{house_id}_EV#load")
        self.pub_charge_rate = helics.helicsFederateRegisterPublication(helics_federate,
                                                                        f"F0_house_A{house_id}_EV/charge_rate",
                                                                        helics.helics_data_type_double)

    def update_state(self):
        helics.helicsPublicationPublishDouble(self.pub_charge_rate, 0.0)
        self.location = helics.helicsInputGetString(self.sub_location)
        self.stored_energy = helics.helicsInputGetString(self.sub_stored_energy)


class House:
    def __init__(self, helics_federate: HelicsFederate, house_id: int, info, agents_dict, auction: Auction, seed):
        self.number = house_id
        self.name = f"F0_house_A{house_id}"
        self.auction = auction
        hvac_name = info['HVAC']
        PV_name = info['PV']
        battery_name = info['battery']
        self.hvac = HVAC(helics_federate, house_id, hvac_name, agents_dict['hvacs'][hvac_name], auction)
        self.pv = PV(helics_federate, house_id, PV_name) if PV_name else None
        self.ev = EV(helics_federate, house_id, )
        self.battery = BATTERY(helics_federate, house_id, battery_name) if battery_name else None
        self.role = 'buyer'  # current role: buyer/seller/none-participant

        # market price related
        self.bid = []
        self.bid_price = 0.0
        random.seed(seed)
        self.fix_bid_price_solar = random.uniform(0.01, 0.015)

        # measurements
        self.mtr_voltage = 120.0
        self.mtr_power = 0
        self.house_kw = 0
        self.unresponsive_kw = 0

        # prediction
        self.house_load_predict = 0  # unit. kw
        self.solar_power_predict = 0  # unit. kw

        # about packet
        self.packet_unit = 1.0  # unit. kw

        self.pubMtrMode = helics.helicsFederateRegisterPublication(helics_federate, f"F0_tpm_A{house_id}/bill_mode",
                                                                   helics.helics_data_type_string)

        self.pubMtrMonthly = helics.helicsFederateRegisterPublication(helics_federate,
                                                                      f"F0_tpm_A{house_id}/monthly_fee",
                                                                      helics.helics_data_type_double)
        self.pubMtrPrice = helics.helicsFederateRegisterPublication(helics_federate, f"F0_tpm_A{house_id}/price",
                                                                    helics.helics_data_type_double)

        self.subMtrVoltage = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                       f"gld1/F0_tpm_A{house_id}#measured_voltage_1")
        self.subMtrPower = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                     f"gld1/F0_tpm_A{house_id}#measured_power")
        self.subHousePower = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                       f"gld1/house_F0_tpm_A{house_id}#measured_power")

    def set_meter_mode(self):
        helics.helicsPublicationPublishString(self.pubMtrMode, 'HOURLY')
        helics.helicsPublicationPublishDouble(self.pubMtrMonthly, 0.0)

    def update_measurements(self):
        # for billing meter measurements ==================
        # billing meter voltage
        self.mtr_voltage = abs(helics.helicsInputGetComplex(self.subMtrVoltage))
        # billing meter power
        self.mtr_power = helics.helicsInputGetComplex(self.subMtrPower).real * 0.001  # unit. kW

        # for house meter measurements ==================
        # house meter power
        self.house_kw = helics.helicsInputGetComplex(self.subHousePower).real * 0.001  # unit. kW

        # for HVAC measurements  ==================
        self.hvac.update_state()  # state update for control
        self.hvac.update_energy_market()  # measured the energy consumed during the market period

        # for unresponsive load ==================
        self.unresponsive_kw = max(self.house_kw - self.hvac.hvac_kw, 0)  # for unresponsive load

        if self.pv:
            self.pv.update_state()

        if self.battery:
            self.battery.update_state()

    def predict_house_load(self):
        self.house_load_predict = self.unresponsive_kw + 3.0 * self.hvac.power_needed

    def predict_solar_power(self):
        self.solar_power_predict = self.pv.solar_DC_V_out * self.pv.solar_DC_I_out / 1000 if self.pv else 0

    def publish_meter_price(self):
        helics.helicsPublicationPublishDouble(self.pubMtrPrice, self.auction.clearing_price)

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
        diff = self.solar_power_predict - self.house_load_predict  # estimated the surplus solar generation

        num_packets = int(diff // self.packet_unit)  # estimated the number of surplus PV power packet
        if num_packets >= 1:
            self.role = 'seller'
            quantity = num_packets * self.packet_unit
            self.bid_price = self.fix_bid_price_solar  # fixed bid price for the seller
            base_covered = True
        elif num_packets < 1 and diff >= 0:  # diff is higher than the house load, but is not enough to generate one packet
            self.role = 'none-participant'
            self.bid_price = 0
            quantity = 0
            base_covered = True
        else:
            self.role = 'buyer'
            if self.hvac.power_needed:
                p = self.auction.clearing_price + (
                        self.hvac.air_temp - self.hvac.basepoint) * self.hvac.ramp * self.auction.std_dev / self.hvac.Trange  # * 30
                self.bid_price = min(self.hvac.price_cap, max(p, 0.0))
                if self.solar_power_predict >= self.unresponsive_kw:  # can cover base load
                    quantity = abs(num_packets) * self.packet_unit
                    base_covered = True
                else:
                    quantity = 3.0
                    base_covered = False
            else:
                self.bid_price = 0
                quantity = 0
                base_covered = False

        self.bid = [self.bid_price, quantity, self.hvac.power_needed, self.role, self.unresponsive_kw, self.name,
                    base_covered]

        return self.bid

    def demand_response(self):
        self.hvac.price_response(self.auction.clearing_price)

    def post_market_control(self, market_condition, marginal_quantity):
        bid_price = self.bid[0]
        quantity = self.bid[1]
        hvac_power_needed = self.bid[2]
        role = self.bid[3]
        unres_kw = self.bid[4]
        base_covered = self.bid[6]

        if role == 'seller' and self.pv:
            # PV control
            if market_condition == 'flexible-generation':
                self.pv.set_power_out(unres_kw + 3 * int(hvac_power_needed) + quantity, 0)
            elif market_condition == 'double-auction':
                if bid_price < self.auction.clearing_price:
                    self.pv.set_power_out(unres_kw + 3 * int(hvac_power_needed) + quantity, 0)
                if bid_price > self.auction.clearing_price:
                    self.pv.set_power_out(unres_kw + 3 * int(hvac_power_needed), 0)
                elif bid_price == self.auction.clearing_price:
                    self.pv.set_power_out(unres_kw + 3 * int(hvac_power_needed) + marginal_quantity, 0)
            else:  # flexible-load (impossible)
                print("Invalid seller in flexible-load condition!!")
            # hvac control
            self.hvac.set_on(hvac_power_needed)

        if role == 'buyer':
            # hvac control and PV control
            if hvac_power_needed:
                if base_covered:
                    if bid_price >= self.auction.clearing_price:
                        self.hvac.set_on(True)
                        if self.pv:
                            self.pv.set_power_out(unres_kw + max(3.0 - quantity, 0), 0)
                    else:
                        self.hvac.set_on(False)
                        if self.pv:
                            self.pv.set_power_out(unres_kw, 0)
                else:
                    if bid_price >= self.auction.clearing_price:
                        self.hvac.set_on(True)
                        if self.pv:
                            self.pv.set_power_out(0, 0)
                    else:
                        self.hvac.set_on(False)
                        if self.pv:
                            self.pv.set_power_out(0, 0)
            else:
                self.hvac.set_on(False)
                if self.pv:
                    self.pv.set_power_out(0, 0)
        if role == 'none-participant':
            # PV control
            if self.pv:
                self.pv.set_power_out(unres_kw + 3 * int(hvac_power_needed), 0)
            # hvac control
            self.hvac.set_on(hvac_power_needed)


class VPP:
    def __init__(self, helics_federate: HelicsFederate, name, enable=True):
        self.name = name
        self.enable = enable

        self.vpp_load_p = 0  # kVA
        self.vpp_load_q = 0
        self.balance_signal = 220  # kVA

        self.request_list = []
        self.response_list = []

        self.subVppPower = helics.helicsFederateRegisterSubscription(helics_federate,
                                                                     f'gld1/F0_triplex_node_A#measured_power')

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
        cval = helics.helicsInputGetComplex(self.subVppPower)
        self.vpp_load_p = cval.real * 0.001
        self.vpp_load_q = cval.imag * 0.001

    def update_balance_signal(self):
        pass
