# generate a scenario file with the provided arguments and immediately run the cosimulation

import json
import subprocess

from fed_weather.TMY3toCSV import weathercsv
from glmhelper import GlmGenerator
from helics_config_helper import HelicsConfigHelper
from scenario import PETScenario


class PETRunner:
    def __init__(self, scenario: PETScenario):
        self.scenario = scenario

    def generate_auxiliary_files(self):
        # generate gridlabd config
        GlmGenerator(self.scenario).save("fed_gridlabd")

        # configure weather data
        weathercsv(f"fed_weather/tesp_weather/AZ-Tucson_International_Ap.tmy3", 'weather.csv', self.scenario.start_time,
            self.scenario.end_time,
            self.scenario.start_time.year)
        print(f"wrote weather data for scenario")

        # generate HELICS configs for fed_gridlabd and fed_substation
        helics_config_helper = HelicsConfigHelper(self.scenario)
        with open("fed_gridlabd/gridlabd_helics_config.json", "w") as f:
            json.dump(helics_config_helper.gridlab_config, f, indent=4)
        print(f"wrote GridLab-D HELICS config")

        with open("fed_substation/substation_helics_config.json", "w") as f:
            json.dump(helics_config_helper.pet_config, f, indent=4)
        print(f"wrote substation HELICS config")

        # update weather config
        weather_config_path = "fed_weather/weather_helics_config.json"
        weather_config = json.load(open(weather_config_path, "r"))
        weather_config["time_stop"] = f"{int((self.scenario.end_time - self.scenario.start_time).total_seconds() / 60)}m"
        json.dump(weather_config, open(weather_config_path, "w"), indent=4)
        print(f"wrote weather HELICS config")

        # update pypower config
        pypower_config_path = "fed_pypower/te30_pp.json"
        pypower_config = json.load(open(pypower_config_path, "r"))
        pypower_config["Tmax"] = int((self.scenario.end_time - self.scenario.start_time).total_seconds())
        json.dump(pypower_config, open(pypower_config_path, "w"), indent=4)
        print(f"wrote pypower HELICS config")

    def run(self):
        self.scenario.save("scenario.pkl")
        self.generate_auxiliary_files()
        subprocess.call(("helics", "run", f"--path=runner.json"))
