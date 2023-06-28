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
import plotly.express as px

set_seed(seed=200, dir="emobpy_data/config_files")

# Dictionary with charging stations type probability distribution per the purpose of the trip (location or destination)
STATION_DISTRIBUTION = {
    'prob_charging_point': {
        'errands': {'none': 1},
        'escort': {'none': 1},
        'leisure': {'none': 1},
        'shopping': {'none': 1},
        'home': {'home': 1},
        'workplace': {'workplace': 0.8},
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
    [BEVspecs().model(('Tesla', 'Model Y Long Range AWD', 2020)), 0.6],
    [BEVspecs().model(('Volkswagen', 'ID.3', 2020)), 0.4],
])

EVProfile = namedtuple("EVProfile", ["mobility", "consumption", "availability", "demand", "car_model"])


def layout(fig, w=None, h=None):
    fig.update_layout(
        width=w or 1000, height=h or 500, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            # yanchor="bottom",
            # y=1.02,
            # y=1,
            # traceorder="normal",
        ),
        font=dict(size=18))


class EVProfiles:
    def __init__(self, start_date: datetime, hours, time_step, num_evs, output_folder,
                 station_distribution=STATION_DISTRIBUTION, car_models_distribution=CAR_MODELS_DISTRIBUTION):
        self.consumption_df = None
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
        if self.num_evs == 0:
            return self
        print("Loading EV profiles from saved")
        files = glob.glob(f"{self.output_folder}/*")
        if len(files) < self.num_evs:
            raise Exception(f"Not enough saved EV profiles. Found {len(files)}, expected {self.num_evs}")
        for f in files[:self.num_evs]:
            with gzip.open(f, 'rb') as handle:
                self.profiles.append(pickle.load(handle))
        self.consumption_df = pd.concat([profile.consumption.timeseries for profile in self.profiles], axis=1,
                                        keys=range(self.num_evs))

        # self.demand_df = pd.concat([profile.demand.timeseries for profile in self.profiles], axis=1,
        #                            keys=range(self.num_evs))

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
        self.consumption_df = pd.concat([profile.consumption.timeseries for profile in self.profiles], axis=1,
                                        keys=range(self.num_evs))
        # self.demand_df = pd.concat([profile.demand.timeseries for profile in self.profiles], axis=1,
        #                            keys=range(self.num_evs))
        return self.profiles

    def create_ev_profile(self, i, car_model: ModelSpecs):
        print(f"Creating EV profile {i}, car model {car_model.name}")
        try:
            mobility = self.create_mobility_timeseries(np.random.uniform(0, 1) < 0.8, np.random.uniform(0, 1) < 0.8)
            consumption = self.create_consumption_timeseries(mobility, car_model)
            # availability = self.create_availability_timeseries(consumption)
            # demand = self.create_demand_timeseries(availability)
            result = EVProfile(mobility, consumption, None, None, car_model)
            print(f"EV profile {i} created, saving now")

            with gzip.open(f"{self.output_folder}/{i}.pkl", 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"EV profile {i} saved")
            return result
        except Exception as e:
            print(e)
            print(f"Failed to create mobility timeseries. Retrying...")
            return self.create_ev_profile(i, car_model)

    def create_mobility_timeseries(self, worker=True, full_time=True):
        full_time = True
        print(f"Creating mobility timeseries, worker={worker}, full_time={full_time}")
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
            stat_dest_path=f"DepartureDestinationTrip_{'Worker' if worker else 'Free'}.csv",
            stat_km_duration_path="DistanceDurationTrip.csv",
        )
        rule_key = ("fulltime" if full_time else "parttime") if worker else "freetime"
        m.set_rules(rule_key=rule_key)
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

    def get_locations_at_time(self, time=None, time_hours=None):
        if not time:
            time = self.start_date + timedelta(hours=time_hours)
        row = self.demand_df.xs('charging_point', level=1, axis=1).asof(time)
        return row.values  # convert to W

    # def save_profiles(self):
    #     for i, p in enumerate(self.profiles):
    #         p = EVProfile(p[0], p[1], p[2], p[3])
    #         with gzip.open(f"{self.output_folder}/{i}.pkl", 'wb') as handle:
    #             pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     print(f"Profiles saved to {self.output_folder}")

    def draw_figures(self):
        fig = make_subplots(rows=1, cols=1,
                            specs=[[{}]])
        fig.update_layout(title_text="EV Locations")

        # EV Locations
        states = self.consumption_df.xs('state', level=1, axis=1).apply(lambda x: collections.Counter(x), axis=1,
                                                                        result_type="expand")
        START_TIME = datetime.strptime('2013-07-02 00:00:00', '%Y-%m-%d %H:%M:%S')
        END_TIME = datetime.strptime('2013-07-06 00:00:00', '%Y-%m-%d %H:%M:%S')
        states = states[(states.index >= START_TIME) & (states.index <= END_TIME)]
        fig.add_traces([
            {
                "type": "scatter",
                "x": states.index,
                "y": states[column],
                "name": f"{column}",
                "stackgroup": "location",
                "line": {"width": 1}
            }
            for column in states.columns
        ], rows=1, cols=1)

        fig.update_xaxes(title_text="", row=1, col=1, tickformat="%H:%M")
        fig.update_yaxes(title_text="EV Locations", row=1, col=1)
        layout(fig, 1200, 400)
        fig.write_html("ev_locations.html")
        fig.write_image("ev_locations.png")

        fig = make_subplots(rows=1, cols=1,
                            specs=[[{"secondary_y": True}]])
        fig.update_layout(title_text="EV Driving Load")

        # EV Driving Loads
        driving_loads = self.consumption_df.xs('average power in W', level=1, axis=1).apply(lambda x: x, axis=1,
                                                                                            result_type="expand")

        START_TIME = datetime.strptime('2013-07-02 00:00:00', '%Y-%m-%d %H:%M:%S')
        END_TIME = datetime.strptime('2013-07-06 00:00:00', '%Y-%m-%d %H:%M:%S')
        driving_loads = driving_loads[(driving_loads.index >= START_TIME) & (driving_loads.index <= END_TIME)]
        bcol = lambda x: f"rgb({255 - x * 5}, 50, {50 + x * 5})"
        # fig.add_traces([
        #     {
        #         "type": "scatter",
        #         "x": driving_loads.index,
        #         "y": driving_loads[ev],
        #         "name": f"EV {ev} Driving Load",
        #         "stackgroup": "dload",
        #         "showlegend": False,
        #         "line": {"width": 0, "color": bcol(ev)}
        #     }
        #     for ev in range(30)
        # ], rows=1, cols=1)

        driving_load_total = driving_loads.sum(axis=1)
        fig.add_trace(
            {
                "type": "scatter",
                "x": driving_load_total.index,
                "y": driving_load_total,
                "name": f"Total driving load",
                # "showlegend": False,
                # "line": {"color": k}
            }, row=1, col=1)
        driving_energy_total = driving_load_total.cumsum()
        fig.add_trace(
            {
                "type": "scatter",
                "x": driving_energy_total.index,
                "y": driving_energy_total,
                "name": f"Cumulative driving energy use",
                # "showlegend": False,
                # "line": {"color": k}
            }, row=1, col=1, secondary_y=True)

        fig.update_xaxes(title_text="", row=1, col=1, tickformat="%H:%M")
        fig.update_yaxes(title_text="EV Driving Load (W)", row=1, col=1)
        fig.update_yaxes(title_text="EV Energy Used (J)", row=1, col=1, secondary_y=True)
        layout(fig, 1200, 400)
        fig.write_html("ev_driving_loads.html")
        fig.write_image("ev_driving_loads.png")
    # fig.show()

    # fig2 = px.area(stored_power, x=stored_power.index, y=stored_power.columns,
    #                labels={str(int(i)): f"EV {i}" for i in stored_power.columns})
    # fig2.show()


def energy_used_between(ts, start_time: datetime, end_time: datetime):
    # print("ss", start_time)
    avg_power = ts["average power in W"]
    # mask = (avg_power.index >= avg_power.index.asof(start_time)) & (avg_power.index <= end_time)
    # rows = ts["average power in W"].iloc[mask]
    # first_row_time = (rows.index[0] + rows.index.freq - start_time).total_seconds()
    # last_row_time = (end_time - rows.index[-1]).total_seconds()
    # power_times = np.array([first_row_time] + [
    #     (rows.index[i + 1] - rows.index[i]).total_seconds()
    #     for i in range(1, len(rows) - 1)
    # ] + [last_row_time])
    # print(rows)
    # print(power_times)
    # return sum(power_times * rows.to_numpy())
    t = start_time
    print("start", start_time)
    energy = 0.0
    while t < end_time:
        next_index = avg_power.loc[avg_power.index > t].index[0]
        next_t = min(next_index, end_time)
        delta = (next_t - t).total_seconds()
        energy += delta * avg_power.asof(t)
        t = next_t
    print(energy, start_time, end_time)
    # new_t = min()


def total_between(ss, start_time: datetime, end_time: datetime):
    t = start_time
    energy = 0.0
    while t < end_time:
        future_indices = ss.loc[ss.index > t].index
        next_index = future_indices[0] if len(future_indices) else end_time
        next_t = min(next_index, end_time)
        delta = (next_t - t).total_seconds()
        # print("QA", ss.asof(t))
        energy += delta * ss.asof(t)
        t = next_t
    return energy


if __name__ == '__main__':
    start_t = datetime.strptime("2013-07-01 00:00:00", '%Y-%m-%d %H:%M:%S')
    # BEVspecs().show_models()
    ev_profiles = EVProfiles(start_t, 192, 0.125, 30, "emobpy_data/profiles")
    # ev_profiles.run(pool_size=1)
    ev_profiles.load_from_saved()
    # ev_profiles.draw_figures()
    # START_TIME = datetime.strptime('2013-07-03 00:00:00', '%Y-%m-%d %H:%M:%S')
    END_TIME = start_t + timedelta(days=8)
    # END_TIME = datetime.strptime('2013-07-05 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
    avg_powers = sum(p.consumption.timeseries["average power in W"] for p in ev_profiles.profiles)
    avg_powers = avg_powers[(avg_powers.index >= start_t) & (avg_powers.index <= END_TIME)]

    rpt = avg_powers[:-1].repeat(2).set_axis(avg_powers.index.repeat(2)[1:-1])
    idx = pd.date_range(start_t, END_TIME, freq=f'300S')
    # total_between(rpt, start_t+timedelta(days=1,hours=23,minutes=45), start_t+timedelta(days=1,hours=23,minutes=50))
    totals = [total_between(rpt, s, s + timedelta(seconds=300)) / 300 for s in idx]
    totals = pd.Series(totals, index=idx)
    pickle.dump(totals, open("driving_power.pkl", "wb"))
    print("\n", totals)


    # idx = pd.date_range(start_t, END_TIME, freq=f'150S')
    # rpt = rpt.reindex(idx   )
    # print(rpt)
    # idx = pd.date_range(start_t, END_TIME, freq=f'300S')

    def time_weighted_average(group):
        return (group['value'] * group['weights']).sum() / group['weights'].sum()


    df = pd.DataFrame(
        {"weights": rpt.index.to_series().diff().shift(-1).fillna(pd.Timedelta(seconds=0)).dt.total_seconds(),
         "value": rpt.values})
    # print(df)
    fig = px.line(avg_powers)
    fig.write_html("test.html")
    t = start_t + timedelta(hours=14)
    # sp = ev_profiles.get_stored_power()
    # ev_profiles.get_loads_at_time(t)
    # ev_profiles.get_locations_at_time(t)
    # c = ev_profiles.profiles[2].consumption.timeseries
    # print(c[c["state"] == "workplace"])
    # print(set(s for p in ev_profiles.profiles for s in set(p.mobility.timeseries["state"])))
    # print(ev_profiles.profiles[2].consumption.timeseries.to_string())
    # print(ev_profiles.profiles[2].car_model.parameters)
    # c = ev_profiles.profiles[2].consumption.timeseries
    # loc_changes = (c["state"].shift() != c["state"]).loc[lambda x: x].index
    # print("FF", loc_changes[loc_changes > datetime.strptime("2013-07-01 09:00:00", '%Y-%m-%d %H:%M:%S')][0])
    # print(c.to_string())
    # print(energy_used_between(c, start_t + timedelta(seconds=47940), start_t + timedelta(seconds=47955)))
    # print(ev_profiles.profiles[0].consumption.timeseries.index.freq)
    # ev_profiles.draw_figures()
    # ev_profile.save_profiles()
    # ev_profile.run()

    # mobility1, consumption1, availability1, demand1 = ev_profile.profiles[0]
    # PLT = NBplot(DotDict({"db": {availability1.name: availability1.__dict__, demand1.name: demand1.__dict__}}))
    # PLT.sgplot_ga(availability1.name, rng=None, to_html=False, path=None).write_image("test.png")
    # PLT.sgplot_ged(demand1.name, rng=None, to_html=False, path=None).write_image("test2.png")

    # print("PROFILE")
    # print(demand1.profile)
    # models = [CAR_MODELS_DISTRIBUTION[i, 0] for i in np.random.choice([0,1], size=30, p=CAR_MODELS_DISTRIBUTION[:,1])]
