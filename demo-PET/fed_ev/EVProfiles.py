import collections
import glob
import gzip
import os
import pickle
import sys
from collections import namedtuple
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from random import choices

import numpy as np
import pandas as pd
from emobpy import Mobility, Consumption, HeatInsulation, BEVspecs, Availability, Charging, ModelSpecs
from emobpy.tools import set_seed
from plotly.subplots import make_subplots

sys.path.append('..')
from my_tesp_support_api.utils import DotDict

set_seed(seed=200, dir="emobpy_data/config_files")

# Dictionary with charging stations type probability distribution per the purpose of the trip (location or destination)
STATION_DISTRIBUTION = {
    'prob_charging_point': {
        'errands': {'none': 1},
        'escort': {'none': 1},
        'leisure': {'none': 1},
        'shopping': {'none': 1},
        'home': {'home': 1},
        'workplace': {'workplace': 0.5},
        # If the vehicle is at the workplace, it will always find a charging station available (assumption)
        'driving': {'none': 1}
    },
    # with the low probability given to fast charging is to ensure fast charging only for very long trips (assumption)
    'capacity_charging_point': {  # Nominal power rating of charging station in kW
        'public': 22,
        'home': 3.7,
        'workplace': 11,
        'none': 0,  # dummy station
        'fast75': 75,
        'fast150': 150
    }
}

CAR_MODELS_DISTRIBUTION = np.array([
    [BEVspecs().model(('Tesla', 'Model 3 Long Range AWD', 2020)), 0.6],
    [BEVspecs().model(('Volkswagen', 'ID.3', 2020)), 0.4],
])

EVProfile = namedtuple("EVProfile", ["mobility", "consumption", "availability", "demand", "car_model"])


class EVProfiles:
    def __init__(self, start_date: datetime, hours, time_step, num_evs, output_folder,
                 station_distribution=STATION_DISTRIBUTION, car_models_distribution=CAR_MODELS_DISTRIBUTION):
        self.demand_df = None
        self.start_date = start_date
        self.hours = hours
        self.time_step = time_step
        self.num_evs = num_evs
        self.profiles = []
        self.output_folder = output_folder
        self.car_models_distribution = car_models_distribution
        self.station_distribution = station_distribution

    def load_from_saved(self):
        print("Loading EV profiles from saved")
        files = glob.glob(f"{self.output_folder}/*")
        if len(files) < self.num_evs:
            raise Exception(f"Not enough saved EV profiles. Found {len(files)}, expected {self.num_evs}")
        for f in files:
            with gzip.open(f, 'rb') as handle:
                self.profiles.append(pickle.load(handle))
        self.demand_df = pd.concat([profile.demand.timeseries for profile in self.profiles], axis=1,
                                   keys=range(self.num_evs))

        print("Finished loading EV profiles from saved")
        print("Loaded " + ", ".join(
            [f"{n}x{m}" for m, n in collections.Counter(p.car_model.name for p in self.profiles).items()]))
        return self

    def clear_output_folder(self):
        print(f"Clearing output folder {self.output_folder}")
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        files = glob.glob(f"{self.output_folder}/*")
        for f in files:
            os.remove(f)

    def run(self, pool_size=2):
        self.profiles = []
        self.clear_output_folder()
        print("Generating EV profiles")
        models = choices(population=self.car_models_distribution[:, 0], weights=self.car_models_distribution[:, 1],
                         k=self.num_evs)

        print("Generating " + ", ".join([f"{n}x{m.name}" for m, n in collections.Counter(models).items()]))
        with Pool(pool_size) as p:
            self.profiles = p.starmap(self.create_ev_profile, zip(range(self.num_evs), models))
            print("Finished generating EV profiles")
            self.demand_df = pd.concat([profile.demand.timeseries for profile in self.profiles], axis=1,
                                       keys=range(self.num_evs))
        return self.profiles

    def create_ev_profile(self, i, car_model: ModelSpecs):
        print(f"Creating EV profile {i}, car model {car_model.name}")
        try:
            mobility = self.create_mobility_timeseries()
            consumption = self.create_consumption_timeseries(mobility, car_model)
            availability = self.create_availability_timeseries(consumption)
            demand = self.create_demand_timeseries(availability)
            result = EVProfile(mobility, consumption, availability, demand, car_model)
            print(f"EV profile {i} created, saving now")

            with gzip.open(f"{self.output_folder}/{i}.pkl", 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"EV profile {i} saved")
            return result
        except Exception as e:
            print(e)
            print(f"Failed to create mobility timeseries. Retrying...")
            return self.create_ev_profile(i, car_model)

    def create_mobility_timeseries(self):
        print("Creating mobility timeseries")
        m = Mobility(config_folder='emobpy_data/config_files')
        m.set_params(
            name_prefix="evprofile",
            total_hours=self.hours,  # one week
            time_step_in_hrs=self.time_step,  # 15 minutes
            category="user_defined",
            reference_date=self.start_date.strftime("%Y-%m-%d")
        )

        m.set_stats(
            stat_ntrip_path="TripsPerDay.csv",
            stat_dest_path="DepartureDestinationTrip.csv",
            stat_km_duration_path="DistanceDurationTrip.csv",
        )
        m.set_rules(rule_key="user_defined")
        m.run()
        print(f"Mobility timeseries created {m.name}")
        return m

    def create_consumption_timeseries(self, mobility: Mobility, car_model: ModelSpecs):
        print(f"Creating consumption timeseries for mobility {mobility.name}")
        c = Consumption(mobility.name, car_model)
        c.load_setting_mobility(DotDict({"db": {mobility.name: mobility.__dict__}}))
        hi = HeatInsulation(True)
        c.run(
            heat_insulation=hi,
            weather_country='DE',
            weather_year=2016,
            passenger_mass=75,  # kg
            passenger_sensible_heat=70,  # W
            passenger_nr=1.5,  # Passengers per vehicle including driver
            air_cabin_heat_transfer_coef=20,  # W/(m2K). Interior walls
            air_flow=0.02,  # m3/s. Ventilation
            driving_cycle_type='WLTC',  # Two options "WLTC" or "EPA"
            road_type=0,  # For rolling resistance, Zero represents a new road.
            road_slope=0
        )
        print(f"Consumption timeseries created {c.name}")
        return c

    def create_availability_timeseries(self, consumption: Consumption):
        print(f"Creating availability timeseries for consumption {consumption.name}")
        ga = Availability(consumption.name, DotDict({"db": {consumption.name: consumption.__dict__}}))
        ga.set_scenario(self.station_distribution)
        ga.run()
        print(f"Availability timeseries created {ga.name}")
        return ga

    @staticmethod
    def create_demand_timeseries(availability: Availability):
        print(f"Creating demand timeseries for availability {availability.name}")
        ged = Charging(availability.name)
        ged.load_scenario(DotDict({"db": {availability.name: availability.__dict__}}))
        ged.set_sub_scenario("immediate")
        ged.run()
        print(f"Demand timeseries created {ged.name}")
        return ged

    def get_loads_at_time(self, time=None, time_hours=None):
        if not time:
            time = self.start_date + timedelta(hours=time_hours)
        row = self.demand_df.xs('charge_grid', level=1, axis=1).asof(time)
        return row.values * 1000  # convert to W

    def get_stored_power(self):
        socs = self.demand_df.xs('actual_soc', level=1, axis=1)
        battery_capacities = np.array([p.car_model.parameters["battery_cap"] for p in self.profiles]) * 1000
        stored = pd.concat([(socs * battery_capacities)], axis=1, keys=['stored_power']).swaplevel(0, 1, 1)
        return stored.join(pd.concat([socs], axis=1, keys=["soc"]).swaplevel(0, 1, 1))

    def get_stored_power_at_time(self, time=None, time_hours=None):
        if not time:
            time = self.start_date + timedelta(hours=time_hours)
        return self.get_stored_power().asof(time)

    # def save_profiles(self):
    #     for i, p in enumerate(self.profiles):
    #         p = EVProfile(p[0], p[1], p[2], p[3])
    #         with gzip.open(f"{self.output_folder}/{i}.pkl", 'wb') as handle:
    #             pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     print(f"Profiles saved to {self.output_folder}")

    def draw_figures(self):
        fig = make_subplots(rows=4, cols=1,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{}]])
        fig.update_layout(title_text="EV Grid Demand")

        # EV Locations
        states = self.demand_df.xs('state', level=1, axis=1).apply(lambda x: collections.Counter(x), axis=1,
                                                                   result_type="expand")
        fig.add_traces([
            {
                "type": "scatter",
                "x": states.index,
                "y": states[column],
                "name": f"{column} count",
                "showlegend": False,
            }
            for column in states.columns
        ], rows=1, cols=1)

        fig.update_yaxes(title_text="EV Locations", row=1, col=1)

        # EVs at Charging Points

        num_at_charging_points = self.demand_df.xs('charging_point', level=1, axis=1).apply(
            lambda x: x.str.count('^((?!none).)*$'), axis=1).sum(axis=1)
        fig.add_trace({
            "type": "scatter",
            "x": num_at_charging_points.index,
            "y": num_at_charging_points,
            "name": f"EVs at charging points",
            "showlegend": False,
        }, row=2, col=1)

        num_charging = self.demand_df.xs('charge_grid', level=1, axis=1).apply(
            lambda x: (x > 1).sum(), axis=1)

        fig.add_trace({
            "type": "scatter",
            "x": num_charging.index,
            "y": num_charging,
            "name": f"EVs at charging points",

            "showlegend": False,
        }, row=2, col=1, secondary_y=True)

        fig.update_yaxes(title_text="EVs at charging points", row=2, col=1)
        fig.update_yaxes(title_text="EVs charging", row=2, col=1, secondary_y=True)

        # EV Grid Demand

        fig.add_traces([
            {
                "type": "scatter",
                "x": profile.demand.timeseries.index,
                "y": profile.demand.timeseries['charge_grid'],
                "name": f"EV {i} Grid Demand",
                "showlegend": False,
                "line": {"color": "#bbb"}
            }
            for i, profile in enumerate(self.profiles)
        ], rows=3, cols=1)

        fig.update_yaxes(title_text="Individual EV Demand (kW)", secondary_y=False, row=3, col=1)

        total_charge_grid = self.demand_df.xs('charge_grid', level=1, axis=1).sum(axis=1)
        fig.add_trace({
            "type": "scatter",
            "x": total_charge_grid.index,
            "y": total_charge_grid,
            "name": f"Sum EV Grid Demand",
            "showlegend": False,
            "line": {"color": "#2594f4", "width": 4}
        }, secondary_y=True, row=3, col=1)
        fig.update_yaxes(title_text="Sum EV Grid Demand (kW)", secondary_y=True, row=3, col=1)

        stored_power = self.get_stored_power().xs('stored_power', level=1, axis=1)
        #
        fig.add_traces([
            {
                "type": "scatter",
                "x": stored_power.index,
                "y": stored_power[i],
                "name": f"EV {i} Stored Power",
                "showlegend": False,
                "stackgroup": "stored_power",
                "line": {"width": 0}
            }
            for i, profile in enumerate(self.profiles)
        ], rows=4, cols=1)

        for i in range(1, 5):
            fig.update_xaxes(title_text="Time", tickformat="%Hh", tickmode="linear",
                             tick0=self.profiles[0].demand.timeseries.index[0], dtick=7200e3, row=i, col=1)

        # fig.write_image("ev_grid_demand.svg")
        # fig.write_html("ev_grid_demand.html")
        fig.show()

        # fig2 = px.area(stored_power, x=stored_power.index, y=stored_power.columns,
        #                labels={str(int(i)): f"EV {i}" for i in stored_power.columns})
        # fig2.show()


if __name__ == '__main__':
    start_t = datetime.strptime("2013-07-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    ev_profiles = EVProfiles(start_t, 48, 0.125, 30, "emobpy_data/profiles")
    # ev_profiles.run()
    ev_profiles.load_from_saved()
    t = start_t + timedelta(hours=14)
    sp = ev_profiles.get_stored_power()
    ev_profiles.get_loads_at_time(t)
    ev_profiles.draw_figures()
    # ev_profile.save_profiles()
    # ev_profile.run()

    # mobility1, consumption1, availability1, demand1 = ev_profile.profiles[0]
    # PLT = NBplot(DotDict({"db": {availability1.name: availability1.__dict__, demand1.name: demand1.__dict__}}))
    # PLT.sgplot_ga(availability1.name, rng=None, to_html=False, path=None).write_image("test.png")
    # PLT.sgplot_ged(demand1.name, rng=None, to_html=False, path=None).write_image("test2.png")

    # print("PROFILE")
    # print(demand1.profile)
    # models = [CAR_MODELS_DISTRIBUTION[i, 0] for i in np.random.choice([0,1], size=30, p=CAR_MODELS_DISTRIBUTION[:,1])]
