from datetime import datetime, timedelta
from enum import IntEnum
from random import random

import helics
import numpy as np
import pandas
from emobpy import Availability, ModelSpecs, Consumption


class EVChargingState(IntEnum):
    DISCONNECTED = 0
    CHARGING = 1
    DISCHARGING = 2


# MAX_CHARGE_RATE = 3700
# MAX_DISCHARGE_RATE = 4000

# PET EV controller
# uses mobility/grid-availability data from emobpy to simulate an EV responding to PET
# market conditions according to a strategy.
class V2GEV:
    def __init__(self, helics_fed, name: str, start_time: datetime, consumption: Consumption, car_model: ModelSpecs):
        # self.mobility = mobility
        # self.consumption = consumption
        self.helics_fed = helics_fed
        self.name = name
        self.profile = consumption.timeseries
        self.car_model = car_model

        self.battery_capacity = car_model.parameters["battery_cap"] * 1000 * 3600
        self.stored_energy = random() * self.battery_capacity
        self.location = "home"
        self.charge_rate = 0

        self.pub_location = helics.helicsFederateRegisterPublication(self.helics_fed, f"{name}/location",
                                                                     helics.helics_data_type_string)
        self.pub_stored_energy = helics.helicsFederateRegisterPublication(self.helics_fed, f"{name}/stored_energy",
                                                                          helics.helics_data_type_string)
        self.pub_load = helics.helicsFederateRegisterPublication(self.helics_fed, f"{name}/load",
                                                                 helics.helics_data_type_complex, "")

        self.sub_charge_rate = helics.helicsFederateRegisterSubscription(self.helics_fed, f"{name}/charge_rate")

        self.current_time = start_time

    def publish_location(self):
        helics.helicsPublicationPublishString(self.pub_location, self.profile["charging_point"].asof(self.current_time))

    def energy_used_between(self, start_time: datetime, end_time: datetime):
        avg_power = self.profile["average power in W"]
        mask = (avg_power.index >= avg_power.index.asof(start_time)) & (avg_power.index <= end_time)
        rows = self.profile["average power in W"].iloc[mask]

        first_row_time = (rows.index[0] + rows.index.freq - start_time).total_seconds()
        last_row_time = (end_time - rows.index[-1]).total_seconds()
        power_times = np.array([first_row_time] + [
            (rows.index[i + 1] - rows.index[i]).total_seconds()
            for i in range(1, len(rows) - 1)
        ] + [last_row_time])
        if self.name == "F0_house_A4_EV":
            print("POWERTIMES", power_times)
        return sum(power_times * rows.to_numpy())
        # print(rows.index[0], rows.index[1], start_time, end_time)
        # first_row_energy = (rows.index[1] - start_time).total_seconds() * rows[0]
        last_row_energy = (end_time - rows.index[-1]).total_seconds() * rows[-1]
        # middle_rows_energy = sum(rows[i] * (rows.index[i+1] - rows.index[i]).total_seconds() for i in range(1, ))
        # print(first_row_energy)
        # print(rows)

    def update_state(self, new_time):
        time_delta = (new_time - self.current_time).total_seconds()

        # calculate energy used up to now
        energy_used = self.energy_used_between(self.current_time, new_time)

        # calculate charge change from charge/discharge
        charge_delta = time_delta * self.charge_rate

        self.stored_energy += charge_delta - energy_used

        if self.name == "F0_house_A4_EV":
            print(
                f"{self.name} charged {charge_delta / 1000:.3f}kJ, used {energy_used / 1000:.3f}kJ between {self.current_time} and {new_time}, {self.stored_energy / 1000:.3f}kJ remaining")

        self.location = self.profile["state"].asof(new_time)
        new_charge_rate = helics.helicsInputGetDouble(self.sub_charge_rate)
        if self.location != "home" and new_charge_rate != 0.0:
            raise Exception(
                f"{self.name} got charge rate change to {new_charge_rate} while not at home ({self.location})")
        self.charge_rate = new_charge_rate
        helics.helicsPublicationPublishComplex(self.pub_load, -self.charge_rate, 0)

        self.current_time = new_time
