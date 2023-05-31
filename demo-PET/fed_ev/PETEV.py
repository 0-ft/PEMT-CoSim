from datetime import datetime, timedelta
from enum import IntEnum
from random import random

import helics
import numpy as np
import pandas
from emobpy import Availability, ModelSpecs, Consumption
from helics import HelicsFederate


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
    def __init__(self, helics_fed: HelicsFederate, name: str, start_time: datetime, consumption: Consumption,
                 car_model: ModelSpecs):
        # self.mobility = mobility
        # self.consumption = consumption
        self.helics_fed = helics_fed
        self.name = name
        self.profile = consumption.timeseries
        self.car_model = car_model

        self.battery_capacity = car_model.parameters["battery_cap"] * 1000 * 3600
        self.stored_energy = (1.0 - random() * 0.1) * self.battery_capacity
        self.location = "home"
        self.desired_charge_rate = 0
        self.charge_rate = 0
        self.time_to_full_charge = float('inf')
        self.pub_location = helics.helicsFederateGetPublication(self.helics_fed, f"{name}/location")
        self.pub_stored_energy = helics.helicsFederateGetPublication(self.helics_fed, f"{name}/stored_energy")
        self.pub_load = helics.helicsFederateGetPublication(self.helics_fed, f"{name}/load")
        self.pub_soc = helics.helicsFederateGetPublication(self.helics_fed, f"{name}/soc")

        self.sub_charge_rate = helics.helicsFederateGetSubscription(self.helics_fed, f"sub1/{name}/charge_rate")

        self.current_time = start_time
        self.history = []

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
        return sum(power_times * rows.to_numpy())
        # print(rows.index[0], rows.index[1], start_time, end_time)
        # first_row_energy = (rows.index[1] - start_time).total_seconds() * rows[0]
        # last_row_energy = (end_time - rows.index[-1]).total_seconds() * rows[-1]
        # middle_rows_energy = sum(rows[i] * (rows.index[i+1] - rows.index[i]).total_seconds() for i in range(1, ))
        # print(first_row_energy)
        # print(rows)

    def publish_state(self):
        self.history.append([self.current_time, self.location, self.stored_energy, self.charge_rate,
                             self.stored_energy / self.battery_capacity])
        self.pub_location.publish(self.location)
        self.pub_stored_energy.publish(self.stored_energy)
        self.pub_load.publish(complex(self.charge_rate, 0))
        self.pub_soc.publish(self.stored_energy / self.battery_capacity)

    def update_state(self, new_time):
        time_delta = (new_time - self.current_time).total_seconds()

        new_desired_charge_rate = self.sub_charge_rate.double
        if new_desired_charge_rate != self.desired_charge_rate:
            print(f"{self.name} desired charge rate updated from {self.desired_charge_rate} to {new_desired_charge_rate}")
        self.desired_charge_rate = new_desired_charge_rate

        if (self.location != "home") and self.desired_charge_rate != 0.0:
            raise Exception(
                f"{self.name} desired charge rate is {self.desired_charge_rate} while at ({self.location})")

        # calculate energy used up to now
        energy_used = self.energy_used_between(self.current_time, new_time)

        if self.desired_charge_rate > 0:
            self.charge_rate = self.desired_charge_rate if self.stored_energy / self.battery_capacity < 0.9999 else 0
        elif self.desired_charge_rate < 0:
            self.charge_rate = self.desired_charge_rate if self.stored_energy / self.battery_capacity > 0.0001 else 0
        else:
            self.charge_rate = 0

        # calculate charge change from charge/discharge
        charge_delta = time_delta * self.charge_rate
        charge_delta = min(charge_delta, self.battery_capacity - self.stored_energy)

        self.stored_energy += charge_delta - energy_used

        if 0.9999 < self.stored_energy / self.battery_capacity and self.stored_energy != self.battery_capacity:
            print(f"{self.name} reached full charge")
            self.stored_energy = self.battery_capacity

        self.time_to_full_charge = ((self.battery_capacity - self.stored_energy) / self.charge_rate) \
            if self.charge_rate > 0 and self.stored_energy < self.battery_capacity else float('inf')

        self.location = self.profile["state"].asof(new_time)
        self.current_time = new_time
