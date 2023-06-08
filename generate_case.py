# file: prepare_case.py
"""
Function:
        generate a co-simulation testbed based on user-defined configurations
last update time: 2021-11-11
modified by Yuanliang Li

"""
import json
import pickle

from fed_weather.TMY3toCSV import weathercsv
from glmhelper import GlmGenerator
from helics_config_helper import HelicsConfigHelper
from scenario import PETScenario

"""0. generate a glm file (TE_Challenge.glm) according to user's preference"""

scenario = PETScenario(num_houses=30, num_ev=30, num_pv=30)
with open("scenario.pkl", "wb") as f:
    pickle.dump(scenario, f)

glm = GlmGenerator(scenario)
# glm.generate_glm()
glm.save("fed_gridlabd")
"""1. configure simulation time period"""
year = 2013
start_time = '2013-07-01 00:00:00'
stop_time = '2013-07-09 00:00:00'

"""2. configure weather data"""
tmy_file_name = 'AZ-Tucson_International_Ap.tmy3'  # choose a .tmy3 file to specify the weather in a specific location

weathercsv(f"fed_weather/tesp_weather/{tmy_file_name}", 'weather.dat', start_time, stop_time,
           year)  # it will output weather.dat in the weather fold as the input of the weather federate

"""3. generate configuration files for gridlabd, substation, pypower, and weather"""
# tesp.glm_dict('TE_Challenge', te30=True)
#
# tesp.prep_substation('TE_Challenge', scenario)

hch = HelicsConfigHelper(scenario)
with open("fed_gridlabd/TE_Challenge_HELICS_gld_msg.json", "w") as f:
    json.dump(hch.gridlab_config, f, indent=4)

with open("fed_substation/TE_Challenge_HELICS_substation.json", "w") as f:
    json.dump(hch.pet_config, f, indent=4)
