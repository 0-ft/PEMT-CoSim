import pickle
from datetime import datetime, timedelta
from sys import argv

from scipy.integrate import trapezoid
from scipy.stats import iqr

import numpy as np
from pandas import DataFrame
import plotly.express as px

with open(argv[1], "rb") as f:
    h = pickle.load(f)
start_time = datetime.strptime('2013-07-01 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
end_time = datetime.strptime('2013-07-09 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
houses = h["houses"][(start_time <= h["houses"].index) & (h["houses"].index < end_time)]
# bids = h["bids"][(start_time <= h["bids"].index) & (h["bids"].index < end_time)]
auction = h["auction"][(start_time <= h["auction"].index) & (h["auction"].index < end_time)]
grid = h["grid"][(start_time <= h["grid"].index) & (h["grid"].index < end_time)]

hvac_loads = houses["sum.hvac.hvac_load"]
unresp_loads = houses["sum.unresponsive_load"]
seconds = (houses.index - start_time).total_seconds()

hvac_j_per_day = trapezoid(y=hvac_loads.values, x=seconds) / 8 / 30
unresp_j_per_day = trapezoid(y=unresp_loads.values, x=seconds) / 8 / 30
j_per_day = hvac_j_per_day + unresp_j_per_day
w = j_per_day / (3600 * 24)

print(w * 30)
# time = auction.index
# lmp = auction["lmp"]
#
# mean = (np.ones_like(time) * (np.mean(lmp))).astype(float)
# median = (np.ones_like(time) * (np.median(lmp))).astype(float)
#
# mean_until = np.array([
#     np.mean(lmp[(lmp.index >= (i - timedelta(hours=24))) & (lmp.index <= i)])
#     for i in lmp.index
# ])
#
# median_until = np.array([
#     np.median(lmp[(lmp.index >= (i - timedelta(hours=24))) & (lmp.index <= i)])
#     for i in lmp.index
# ])
# iqr_until = np.array([
#     iqr(lmp[(lmp.index >= (i - timedelta(hours=24))) & (lmp.index <= i)])
#     for i in lmp.index
# ])
#
# # print(iqr, lmp, time)
# df = DataFrame({
#     # "lmp": lmp,
#     "lmp": lmp,
#     "mean": mean,
#     "median": median,
#     "iqr_until": iqr_until,
#     "median_until": median_until,
#     "bt": mean_until - iqr_until * 0.1,
#     "st": mean_until + iqr_until * 0.1,
# })
# print(df)
# print(df.dtypes)
# df.index = time
# fig = px.scatter(df, render_mode="svg")
# fig.write_html("price.html")

# time = np.arange(0, 8000)
# t = time * np.pi * 2 / 1000
#
# grid_cap = 80
# max_lmp = 0.03
# min_lmp = 0.015
# lmp = np.power(np.sin(t / 2), 6) * (max_lmp - min_lmp) + min_lmp
#
# max_pv = 84
# pv_availability = np.power(np.clip(np.sin(t - np.pi / 4), 0, 1), 2) * max_pv
# pv_price = lmp * 0.95
#
# hvac_cap = 0.03
#
# hvac_desired = np.clip(np.sin(t - 1.5) + 0.5, 0, 1.5) / 1.5 * 110
# unresp = (np.abs(np.sin(t / 2 - 0.5)) / 2 + 0.5) * 50
#
# grid_left = np.ones_like(t) * grid_cap
# grid_left = grid_left - unresp
#
# print((lmp <= hvac_cap).astype(int) * hvac_desired)
# hvac_from_grid = np.minimum(grid_left, hvac_desired * (lmp <= hvac_cap).astype(int))
#
# grid_left -= hvac_from_grid
# hvac_left = hvac_desired - hvac_from_grid
# hvac_from_pv = np.minimum(grid_left, hvac_desired * (lmp <= hvac_cap).astype(int))
#
#
# df = DataFrame({
#     # "lmp": lmp,
#     "pv power": pv_availability,
#     "hvac desired": hvac_desired,
#     # "unresp": unresp,
#     # "grid_left": grid_left,
#     # "hvac from grid": hvac_from_grid,
#     "hvac left": hvac_left,
#     # "pv price": pv_price
# }, index=time)
# # df.index = time
# fig = px.line(df, render_mode="svg")
# fig.write_html("price.html")
