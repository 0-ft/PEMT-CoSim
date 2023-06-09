import pickle
from datetime import datetime

import autosklearn.regression
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import plotly.express as px
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

with open("metrics_cv.pkl", "rb") as f:
    h = pickle.load(f)
start_time = datetime.strptime('2013-07-01 00:00:20 -0800', '%Y-%m-%d %H:%M:%S %z')
end_time = datetime.strptime('2013-07-05 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
houses = h["houses"][(start_time <= h["houses"].index) & (h["houses"].index < end_time)]
bids = h["bids"][(start_time <= h["bids"].index) & (h["bids"].index < end_time)]
auction = h["auction"][(start_time <= h["auction"].index) & (h["auction"].index < end_time)]
grid = h["grid"][(start_time <= h["grid"].index) & (h["grid"].index < end_time)]

houses.index = pd.DatetimeIndex(houses.index)
houses = houses.resample("300S", origin=start_time).first()

grid.index = pd.DatetimeIndex(grid.index)
grid = grid.resample("300S", origin=start_time).first()

air_temps = houses["values.hvac.air_temp"]
set_points = houses["values.hvac.set_point"]
hvac_loads = houses["values.hvac.hvac_load"]
weather_temps = grid["weather_temp"]

times = np.tile(np.array(houses.index.to_list()), 30)
# print(times)
# sps = pd.DataFrame(set_points.to_list(), index=times)
# fig = px.scatter(sps)
# fig.write_image("te.svg")

set_points = np.array(set_points.to_list()).flatten("F")
air_temps = np.array(air_temps.to_list()).flatten("F")
weather_temps = np.tile(np.array(weather_temps.to_list()), 30)
print(set_points.size)
print(air_temps.size)
print(weather_temps.size)
# print(set_points)
# hvac_loads = np.amax(np.array(hvac_loads.to_list()), axis=1)
max_hvac_loads = np.tile(np.amax(np.array(hvac_loads.to_list()), axis=1), 30)
mean_hvac_loads = np.tile(np.mean(np.array(hvac_loads.to_list()), axis=1), 30)
hvac_loads = np.array(hvac_loads.to_list()).flatten("F")
print(hvac_loads)
# air_temps = np.array(air_temps.to_list())[:,0]
# set_points = np.array(set_points.to_list())[:,0]
# hvac_loads = np.array(hvac_loads.to_list())[:,0]

# hvac_ons = hvac_loads > 0.0
# set_points = set_points[hvac_ons]
# hvac_loads = hvac_loads[hvac_ons]
# air_temps = air_temps[hvac_ons]
# weather_temps = weather_temps[hvac_ons]

# print(len(hvac_loads))
# print(len(air_temps))
# print(diff.flatten("F"))
# print(unpack(hvac_ons))
# fig = px.scatter(x=air_temps - set_points, y=hvac_loads)
# fig.write_image("pred.svg")
# diff = diff.reshape(-1,1)
# model = LinearRegression()
# model.fit(ins, max_hvac_loads)
# r_sq = model.score(ins, max_hvac_loads)
# print(f"coefficient of determination: {r_sq}")
# print(model.intercept_, model.coef_)


X = np.stack([air_temps, weather_temps, set_points, air_temps * set_points, air_temps * weather_temps, weather_temps * set_points], axis=1)
y = mean_hvac_loads
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
#     X, y, random_state=1
# )
#
# automl = autosklearn.regression.AutoSklearnRegressor(
#     time_left_for_this_task=120,
#     per_run_time_limit=30,
#     tmp_folder="/tmp/autosklearn_regression_example_tmp",
# )
#
# automl.fit(X_train, y_train)
# print(automl.leaderboard())
# train_predictions = automl.predict(X_train)
# print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
# test_predictions = automl.predict(X_test)
# print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

m1: LinearRegression = LinearRegression().fit(X, y)
print(m1.intercept_, m1.coef_)
print("s1", m1.score(X, y))
# m2 = DecisionTreeRegressor(max_depth=2).fit(X, y)
# print("s2", m2.score(X, y))
# pickle.dump(m2, open("hvac_load_predictor.pkl", "wb"))
# #
preds1 = m1.predict(X)

# ts = X[:,0] * 1000
# wa = X[:,1] * 1000
# rm = X[:,0] * X[:,1] * 1000
# df = pd.DataFrame({"ts": ts, "wa": wa, "rm": rm, "actual": y}, index=times).sort_index()
df = pd.DataFrame({"preds1": preds1, "actual": y}, index=times).sort_index()
px.line(df).write_image("pred.svg")
# m3 = SVR().fit(ins, max_hvac_loads)
# print("s3", m3.score(ins, max_hvac_loads))

