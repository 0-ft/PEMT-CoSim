# file: prepare_case.py
"""
Function:
        generate a co-simulation testbed based on user-defined configurations
last update time: 2021-11-11
modified by Yuanliang Li

"""
import argparse
import json
import pickle
from datetime import datetime

from fed_weather.TMY3toCSV import weathercsv
from glmhelper import GlmGenerator
from helics_config_helper import HelicsConfigHelper
from scenario import PETScenario

"""0. generate a glm file (TE_Challenge.glm) according to user's preference"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_case',
        description='Generate a PET scenario for simulation')
    parser.add_argument("-n", "--num_houses", type=int, default=30)
    parser.add_argument("-e", "--num_ev", type=int, default=30)
    parser.add_argument("-p", "--num_pv", type=int, default=30)
    parser.add_argument("-g", "--grid_cap", type=int, default=200000)
    args = parser.parse_args()

    scenario = PETScenario(
        num_houses=args.num_houses,
        num_ev=args.num_ev,
        num_pv=args.num_pv,
        grid_power_cap=args.grid_cap,
        start_time=datetime(2013, 7, 1, 0, 0, 0),
        end_time=datetime(2013, 7, 9, 0, 0, 0),
    )
    with open("scenario.pkl", "wb") as f:
        pickle.dump(scenario, f)

    GlmGenerator(scenario).save("fed_gridlabd")

    # configure weather data
    weathercsv(f"fed_weather/tesp_weather/AZ-Tucson_International_Ap.tmy3", 'weather.dat', scenario.start_time,
               scenario.end_time,
               scenario.start_time.year)

    # generate HELICS configs for fed_gridlabd and fed_substation
    helics_config_helper = HelicsConfigHelper(scenario)
    with open("fed_gridlabd/TE_Challenge_HELICS_gld_msg.json", "w") as f:
        json.dump(helics_config_helper.gridlab_config, f, indent=4)

    with open("fed_substation/TE_Challenge_HELICS_substation.json", "w") as f:
        json.dump(helics_config_helper.pet_config, f, indent=4)

    weather_config = json.load(open("fed_weather/TE_Challenge_HELICS_Weather_Config.json", "r"))
    weather_config["time_stop"] = f"{int((scenario.end_time - scenario.start_time).total_seconds() / 60)}m"
    json.dump(weather_config, open("fed_weather/TE_Challenge_HELICS_Weather_Config.json", "w"), indent=4)
