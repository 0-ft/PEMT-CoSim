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
        self.location_change_times = (self.profile["state"].shift() != self.profile["state"]).loc[
            lambda x: x].index
        self.time_to_location_change = 0  # (self.location_change_times[0] - start_time).total_seconds()

        # parameters
        self.car_model = car_model
        self.max_discharge_rate = 4000
        self.max_home_charge_rate = 4500
        self.min_home_charge_rate = 2500
        self.battery_capacity = car_model.parameters["battery_cap"] * 1000 * 3600
        self.work_charge_rate = 7000

        # state
        self.stored_energy = random() * self.battery_capacity
        self.location = self.profile["state"].asof(start_time)
        self.desired_charge_rate = 0.0
        self.charging_load = 0.0
        self.driving_load = 0.0
        self.time_to_full_charge = float('inf')
        self.charging_load_range = 0.0, 0.0

        self.pub_location = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#location")
        self.pub_stored_energy = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#stored_energy")
        self.pub_charging_load = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#charging_load")
        self.pub_soc = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#soc")
        self.pub_driving_load = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#driving_load")

        self.pub_max_charging_load = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#max_charging_load")
        self.pub_min_charging_load = helics.helicsFederateGetPublication(self.helics_fed, f"{name}#min_charging_load")

        self.sub_charge_rate = helics.helicsFederateGetSubscription(self.helics_fed, f"pet1/{name}#charge_rate")
        self.prev_time = start_time
        self.current_time = start_time
        self.history = []
        self.enable_movement = True
        self.enable_charging = True
        self.enable_discharging = True

    def energy_used_between(self, start_time: datetime, end_time: datetime):
        if not self.enable_movement:
            return 0.0
        avg_power = self.profile["average power in W"]
        t = start_time
        energy = 0.0
        while t < end_time:
            future_indices = avg_power.loc[avg_power.index > t].index
            next_index = future_indices[0] if len(future_indices) else end_time
            next_t = min(next_index, end_time)
            delta = (next_t - t).total_seconds()
            energy += delta * avg_power.asof(t)
            t = next_t
        return energy
        # mask = (avg_power.index >= avg_power.index.asof(start_time)) & (avg_power.index <= end_time)
        # rows = self.profile["average power in W"].iloc[mask]
        #
        # first_row_time = (rows.index[0] + rows.index.freq - start_time).total_seconds()
        # last_row_time = (end_time - rows.index[-1]).total_seconds()
        # power_times = np.array([first_row_time] + [
        #     (rows.index[i + 1] - rows.index[i]).total_seconds()
        #     for i in range(1, len(rows) - 1)
        # ] + [last_row_time])
        # return sum(power_times * rows.to_numpy())
        # print(rows.index[0], rows.index[1], start_time, end_time)
        # first_row_energy = (rows.index[1] - start_time).total_seconds() * rows[0]
        # last_row_energy = (end_time - rows.index[-1]).total_seconds() * rows[-1]
        # middle_rows_energy = sum(rows[i] * (rows.index[i+1] - rows.index[i]).total_seconds() for i in range(1, ))
        # print(first_row_energy)
        # print(rows)

    def publish_state(self):
        self.history.append([self.current_time, self.location, self.stored_energy, self.charging_load,
                             self.stored_energy / self.battery_capacity])
        self.pub_location.publish(self.location)
        self.pub_stored_energy.publish(self.stored_energy)
        self.pub_charging_load.publish(complex(self.charging_load, 0))
        self.pub_driving_load.publish(self.driving_load)
        self.pub_soc.publish(self.stored_energy / self.battery_capacity)
        self.pub_max_charging_load.publish(self.charging_load_range[1])
        self.pub_min_charging_load.publish(self.charging_load_range[0])

    def time_to_next_location_change(self):
        future_changes = self.location_change_times[self.location_change_times > self.current_time]
        return (future_changes[0] - self.current_time).total_seconds() if len(future_changes) else float('inf')

    def load_range(self):
        # max_discharge = -self.max_discharge_rate * any(
        #     a <= (self.current_time.hour + self.current_time.minute / 60.0) < b for a, b in self.discharge_hod_ranges)
        max_discharge = -self.max_discharge_rate
        soc = self.stored_energy / self.battery_capacity
        if self.location != "home":
            return 0.0, 0.0
        elif .9 <= soc:
            return max_discharge, 0
        elif .3 <= soc < .9:
            return max_discharge, self.max_home_charge_rate
        elif .2 <= soc < .3:
            return 0, self.max_home_charge_rate
        else:
            return self.min_home_charge_rate, self.max_home_charge_rate

    def update_state(self, new_time):
        self.prev_time = self.current_time
        self.current_time = new_time
        time_delta = (self.current_time - self.prev_time).total_seconds()
        self.charging_load_range = self.load_range()

        new_desired_charge_rate = self.sub_charge_rate.double
        # if new_desired_charge_rate != self.desired_charge_rate:
        #     print(
        #         f"{self.name} desired charge rate updated from {self.desired_charge_rate} @ {self.prev_time} to {new_desired_charge_rate} @ {self.current_time}")
        self.desired_charge_rate = new_desired_charge_rate

        # if (self.location != "home") and self.desired_charge_rate != 0.0:
        #     raise Exception(
        #         f"{self.name} desired charge rate is {self.desired_charge_rate} while at ({self.location})")

        # calculate energy used up to now
        energy_used = self.energy_used_between(self.prev_time, self.current_time)
        self.driving_load = energy_used / time_delta if time_delta > 0 else 0

        if self.location == "home":
            if (self.desired_charge_rate > 0 and (
                    self.stored_energy / self.battery_capacity < 0.9999) and self.enable_charging) \
                    or (self.desired_charge_rate < 0 and (
                    self.stored_energy / self.battery_capacity > 0.0001) and self.enable_discharging):
                self.charging_load = self.desired_charge_rate
            else:
                self.charging_load = 0
        else:
            if abs(self.desired_charge_rate) > 0:
                print(f"EV {self.name} unable to set DCR {self.desired_charge_rate} as at {self.location}")
            self.charging_load = 0
        # if self.desired_charge_rate > 0 and self.location == "home":
        #     self.charge_rate = self.desired_charge_rate if (
        #                                                            self.stored_energy / self.battery_capacity < 0.9999) and self.enable_charging else 0.0
        # elif self.desired_charge_rate < 0 and self.location == "home":
        #     self.charge_rate = self.desired_charge_rate if (
        #                                                            self.stored_energy / self.battery_capacity > 0.0001) and self.enable_discharging else 0.0

        # calculate charge change from charge/discharge
        home_charge_delta = time_delta * self.charging_load * (
            self.car_model.parameters["battery_charging_eff"]
            if self.charging_load > 0 else
            self.car_model.parameters["battery_discharging_eff"]
        )
        # home_charge_delta = min(home_charge_delta, self.battery_capacity - self.stored_energy)

        work_charge_delta = time_delta * self.work_charge_rate * (self.location == "workplace")
        if work_charge_delta > 0:
            print(f"EV {self.name} charging at work {work_charge_delta} @ {self.current_time}")
        total_charge_delta = min(home_charge_delta + work_charge_delta, self.battery_capacity - self.stored_energy)

        self.stored_energy += total_charge_delta - energy_used

        if 0.9999 < self.stored_energy / self.battery_capacity and self.stored_energy != self.battery_capacity:
            print(f"{self.name} reached full charge")
            self.stored_energy = self.battery_capacity

        self.time_to_full_charge = ((self.battery_capacity - self.stored_energy) / self.charging_load) \
            if self.charging_load > 0 and self.stored_energy < self.battery_capacity else float('inf')

        self.location = self.profile["state"].asof(new_time) if self.enable_movement else "home"
        self.time_to_location_change = self.time_to_next_location_change()
