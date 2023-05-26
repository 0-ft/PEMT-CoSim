# file: prepare_case.py
"""
Function:
        generate a co-simulation testbed based on user-defined configurations
last update time: 2021-11-11
modified by Yuanliang Li

"""

import numpy as np

import my_tesp_support_api.api as tesp
from fed_weather.TMY3toCSV import weathercsv
from glmhelper import GLM_HELPER

"""0. generate a glm file (TE_Challenge.glm) according to user's preference"""


class GLOBAL_Configuration:
    """ a class which defines the configurations of the glm for GridLAB-D
    """

    minimum_timestep = 1  # simulation time step
    market_period = 300  # market running period
    helics_connected = True

    num_VPPs = 1  # number of VPPs, each VPP manages a number of houses
    VPP_phase_list = ['A']  # the phase managed by a VPP
    num_house_phase_list = [30]  # number of houses of each phase for each VPP
    num_house_list = np.array(num_house_phase_list)  # number of houses for each VPP
    ratio_PV_only_list = [0 / 10]  # ratio of houses that have only PV installed for each VPP
    # in this case, each house must have battery installed, so ratio_PV_only_list = [0,..]
    ratio_Bat_only_list = [0 / 30]  # ratio of houses that have only battery installed for each VPP
    ratio_PV_Bat_list = [0 / 30]  # ratio oh houses that have both PV and battery installed for each VP
    # Note: ratio_PV_only + ratio_Bat_only + ratio_PV_Bat <= 1,
    # the remaining is the houses without PV and battery installed.
    # It is suggested that ratio_PV_only = 0, which means all houses should at least have battery installed
    ratio_PV_generation_list = [100 / 100]  # PV generation ratio for each VPP
    battery_mode = 'LOAD_FOLLOWING'  # CONSTANT_PQ


global_config = GLOBAL_Configuration()
glm = GLM_HELPER(global_config)
glm.generate_glm()

"""1. configure simulation time period"""
year = 2013
start_time = '2013-07-01 00:00:00'
stop_time = '2013-07-03 00:00:00'

"""2. configure weather data"""
tmy_file_name = 'AZ-Tucson_International_Ap.tmy3'  # choose a .tmy3 file to specify the weather in a specific location

weathercsv(f"fed_weather/tesp_weather/{tmy_file_name}", 'weather.dat', start_time, stop_time,
           year)  # it will output weather.dat in the weather fold as the input of the weather federate

"""3. generate configuration files for gridlabd, substation, pypower, and weather"""
tesp.glm_dict('TE_Challenge', te30=True)
tesp.prep_substation('TE_Challenge', global_config)
