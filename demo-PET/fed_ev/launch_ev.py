from datetime import datetime, timedelta

import helics

from EVProfiles import EVProfiles, EVProfile
from pytz import UTC


class EVFederate:
    def __init__(self, start_time: str, fed_name, num_evs):
        self.quant = None
        # self.helics_config_file = helics_config_file
        self.fed_name = fed_name
        self.time_period_hours = 0.125
        self.num_evs = num_evs
        self.helics_fed = None
        # self.helics_config_file = helics_config_file
        # with open(helics_config_file, "r") as f:
        #     self.helics_config = json.load(f)
        self.hour_stop = 48  # simulation duration (default 48 hours)
        self.start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        self.current_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

        self.ev_profiles = EVProfiles(self.start_time, self.hour_stop, self.time_period_hours, num_evs,
                                      "emobpy_data/profiles").load_from_saved()
        self.ev_pubs = []

        self.stop_seconds = int(self.hour_stop * 3600)  # co-simulation stop time in seconds

    def create_federate(self):
        print(f"Creating EV federate")
        fed_info = helics.helicsCreateFederateInfo()
        helics.helicsFederateInfoSetCoreName(fed_info, self.fed_name)
        helics.helicsFederateInfoSetTimeProperty(fed_info, helics.helics_property_time_period,
                                                 self.time_period_hours * 3600)
        helics.helicsFederateInfoSetFlagOption(fed_info, helics.helics_flag_uninterruptible, True)
        self.helics_fed = helics.helicsCreateValueFederate(self.fed_name, fed_info)
        print(f"EV federate {self.fed_name} created")
        self.ev_pubs = [
            helics.helicsFederateRegisterPublication(self.helics_fed, f"F0_house_A{i}_EV/load",
                                                     helics.helics_data_type_double, "")
            for i in range(self.num_evs)
        ]

        print("EV federate publications registered")

    def publish_loads(self):
        print(f"EV federate publishing loads for t={self.current_time}")
        current_loads = self.ev_profiles.get_loads_at_time(time=self.current_time)
        for i, pub in enumerate(self.ev_pubs):
            helics.helicsPublicationPublishDouble(pub, current_loads[i])
        print(f"EV federate published loads {current_loads} at time {self.current_time}")

    def run_federate(self):
        helics.helicsFederateEnterExecutingMode(self.helics_fed)
        print("EV federate entered execution mode")
        pub_count = helics.helicsFederateGetPublicationCount(self.helics_fed)
        # pub_ID = helics.helicsFederateGetPublicationByIndex(self.helics_fed, 1)
        # pub_name = helics.helicsPublicationGetKey(pub_ID)
        # print(f"EV federate has {pub_count} publications, pub_ID = {pub_name}")

        for t in range(0, self.stop_seconds, int(self.time_period_hours * 3600)):
            time_granted_seconds = int(helics.helicsFederateRequestTime(self.helics_fed, self.time_period_hours * 3600))
            self.current_time = self.start_time + timedelta(seconds=time_granted_seconds)
            print(f"EV federate granted time {time_granted_seconds} ({self.current_time})")
            self.publish_loads()

        # for i in range(3001):
        #     print(f"EV federate granted time {time_granted}")
        #     # val = helics.helicsInputGetDouble(self.quant)
        #     # print(val)
        #     for ev in self.evs:
        #         ev["load"] = 10000.0
        #     self.publish_loads()


federate = EVFederate("2013-07-01 00:00:00", "ev1", 30)
federate.create_federate()
federate.run_federate()
