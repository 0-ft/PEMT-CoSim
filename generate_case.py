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
from datetime import datetime, timedelta

from fed_weather.TMY3toCSV import weathercsv
from glmhelper import GlmGenerator
from helics_config_helper import HelicsConfigHelper
from case_runner import PETRunner
from scenario import PETScenario

"""0. generate a glm file (TE_Challenge.glm) according to user's preference"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python3 generate_case.py',
        description='Generate a PET scenario for simulation')
    parser.add_argument("-a", "--name", type=str, default=None, help="scenario name")
    parser.add_argument("-n", "--num_houses", type=int, default=30, help="number of houses")
    parser.add_argument("-e", "--num_ev", type=int, default=30, help="number of EVs")
    parser.add_argument("-p", "--num_pv", type=int, default=30, help="number of PVs")
    parser.add_argument("-g", "--grid_cap", type=int, default=200000, help="grid power capacity (W)")
    # parser.add_argument("-w", "--work_charge_rate", type=int, default=7000, help="work charge rate")
    parser.add_argument("-l", "--length", type=float, default=6, help="simulation length in hours")
    parser.add_argument("-f", "--figure_period", type=int, default=3600*24, help="figure drawing period (seconds)")
    parser.add_argument("-b", "--ev_buy_iqr_ratio", type=float, default=0.3, help="EV buy IQR ratio")
    parser.add_argument("-i", "--input_file", type=argparse.FileType('rb'), required=False, help="scenario file to use")
    parser.add_argument("-r", "--run", action="store_true", help="run the simulation immediately")

    args = parser.parse_args()

    if args.input_file:
        scenario = pickle.load(args.input_file)
    else:
        start_time = datetime(2013, 7, 1, 0, 0, 0)
        end_time = start_time + timedelta(hours=args.length)
        scenario = PETScenario(
            scenario_name=args.name if args.name else f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            num_houses=args.num_houses,
            num_ev=args.num_ev,
            num_pv=args.num_pv,
            grid_power_cap=args.grid_cap,
            start_time=start_time,
            end_time=end_time,
            workplace_charge_capacity=0,
            figure_period=args.figure_period,
            ev_buy_iqr_ratio=args.ev_buy_iqr_ratio
        )
        pickle.dump(scenario, open(f"scenarios/{scenario.name}.pkl", "wb"))

    if(args.run):
        print(f"running cosimulation immediately")
        PETRunner(scenario).run()
        print(f"cosimulation finished")