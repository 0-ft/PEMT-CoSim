import json
import pickle
from collections import namedtuple
from datetime import datetime, timedelta
from random import randint

import helics
import numpy as np
import pandas
from helics import HelicsFederate

from EVProfiles import EVProfiles, EVProfile
from fed_ev.PETEV import V2GEV

EVPublications = namedtuple("EVPublications", "location load")


class EVFederate:
    def __init__(self, start_time: str, num_evs, hour_stop):
        self.quant = None
        self.fed_name = None
        self.time_period_hours = 0.125
        self.time_period_seconds = self.time_period_hours * 3600
        self.num_evs = num_evs
        self.helics_fed: HelicsFederate = None
        self.hour_stop = hour_stop
        self.start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        self.current_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        self.stop_time = self.start_time + timedelta(hours=self.hour_stop)

        self.ev_profiles = EVProfiles(self.start_time, self.hour_stop, self.time_period_hours, num_evs,
                                      "emobpy_data/profiles").load_from_saved()

        self.evs: list[V2GEV] = []

        self.stop_seconds = int(self.hour_stop * 3600)  # co-simulation stop time in seconds
        self.enabled = True

    def create_federate(self):

        print(f"Creating EV federate")
        fed_json = {
            "name": "ev1",
            "uninterruptible": False,
            "publications": [
                p for i in range(self.num_evs) for p in [
                    {
                        "key": f"H{i}_ev#location",
                        "type": "string",
                        "global": False
                    },
                    {
                        "key": f"H{i}_ev#stored_energy",
                        "type": "double",
                        "global": False
                    },
                    {
                        "key": f"H{i}_ev#soc",
                        "type": "double",
                        "global": False
                    },
                    {
                        "key": f"H{i}_ev#charging_load",
                        "type": "complex",
                        "global": False
                    },
                    {
                        "key": f"H{i}_ev#driving_load",
                        "type": "double",
                        "global": False
                    }
                ]
            ],
            "subscriptions": [
                {
                    "key": f"pet1/H{i}_ev#charge_rate",
                    "type": "double"
                } for i in range(self.num_evs)
            ]
        }
        with open("ev_helics_config.json", "w") as config_file:
            json.dump(fed_json, config_file, indent=4)

        self.helics_fed = helics.helicsCreateValueFederateFromConfig("ev_helics_config.json")
        self.fed_name = self.helics_fed.name
        print(f"EV federate {self.fed_name} created", flush=True)
        self.evs = [
            V2GEV(self.helics_fed, f"H{i}_ev", self.current_time, profile.consumption, profile.car_model)
            for i, profile in enumerate(self.ev_profiles.profiles)
        ]

        print("EV federate publications registered")

    def state_summary(self):
        data = pandas.DataFrame(
            [
                [ev.location, ev.stored_energy, ev.charging_load, ev.stored_energy / ev.battery_capacity,
                 ev.time_to_full_charge]
                for ev in self.evs if ev.charging_load > 0.0
            ],
            columns=["location", "stored_energy", "charge_rate", "soc", "time_to_full_charge"])

        return f"{self.current_time}: {len(data)} EVs charging, next battery full in {min(data['time_to_full_charge']) if len(data) else '<inf>'}s"
        return data

    def save_data(self):
        nparr = np.array([ev.history for ev in self.evs])
        with open("arr.pkl", "wb") as out:
            pickle.dump(nparr, out)
        data = pandas.concat(
            [pandas.DataFrame(ev.history, columns=["time", "location", "stored_energy", "charge_rate", "soc"]) for ev in
             self.evs], axis=1,
            keys=range(self.num_evs))
        print(data)
        data.to_csv(f"out.csv")

    def run(self):
        print("EV federate to enter initializing mode", flush=True)
        self.helics_fed.enter_initializing_mode()
        print("EV federate entered initializing mode", flush=True)
        for ev in self.evs:
            # ev.update_state(self.start_time)
            ev.publish_state()
        print("published locations", list(enumerate([ev.location for ev in self.evs])))

        print("EV federate to enter execution mode")
        self.helics_fed.enter_executing_mode()
        print("EV federate entered execution mode")
        # pub_count = helics.helicsFederateGetPublicationCount(self.helics_fed)
        # pub_ID = helics.helicsFederateGetPublicationByIndex(self.helics_fed, 1)
        # pub_name = helics.helicsPublicationGetKey(pub_ID)
        # print(f"EV federate has {pub_count} publications, pub_ID = {pub_name}")

        while self.current_time < self.stop_time:
            current_time_s = (self.current_time - self.start_time).total_seconds()
            next_full_charge = min([ev.time_to_full_charge for ev in self.evs]) + current_time_s
            next_location_change = min([ev.time_to_location_change for ev in self.evs]) + current_time_s
            # delta_to_request = max(1.0, min([ev.time_to_full_charge for ev in self.evs] + [self.time_period_seconds]))
            # time_to_request = (self.current_time - self.start_time).total_seconds() + delta_to_request
            time_to_request = min(next_location_change, next_full_charge, self.hour_stop * 3600)
            delta_to_request = time_to_request - current_time_s
            time_granted_seconds = self.helics_fed.request_time(time_to_request)
            new_time = self.start_time + timedelta(seconds=time_granted_seconds)
            print(f"requested time {time_to_request} (delta +{delta_to_request}), got {time_granted_seconds}")
            for ev in self.evs:
                ev.update_state(new_time)
                ev.publish_state()
            print(f"published EVS: {[f'{i}: {ev.location}, SOC {ev.stored_energy / ev.battery_capacity:3f}, {ev.charging_load:3f}, {ev.desired_charge_rate}' for i, ev in enumerate(self.evs)]}")
            # print("published locations", list(enumerate([ev.location for ev in self.evs])))

            self.current_time = new_time
            print(self.state_summary(), flush=True)
        self.save_data()
        print("EV federate finished + saved", flush=True)
        # self.publish_locations()


federate = EVFederate("2013-07-01 00:00:00", 30, 192)
federate.create_federate()
federate.enabled = True
federate.run()
