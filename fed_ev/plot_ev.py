import pickle

import pandas
import pandas as pd
from plotly.subplots import make_subplots

with open("arr.pkl", "rb") as f:
    data = pickle.load(f)

data = [pandas.DataFrame(x, columns=["time", "location", "stored_energy", "charge_rate", "soc"]) for x in data]
print(data)

fig = make_subplots(rows=3, cols=1,
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])
fig.update_layout(title_text="EV Grid Demand")
fig.add_traces([
    {
        "type": "scatter",
        "x": profile["time"],
        "y": profile["stored_energy"],
        "name": f"EV {i} Stored Power",
        "showlegend": False,
        "stackgroup": "stored_power",
        "line": {"width": 0}
    }
    for i, profile in enumerate(data)
], rows=1, cols=1)

fig.add_traces([
    {
        "type": "scatter",
        "x": profile["time"],
        "y": profile["charge_rate"],
        "name": f"EV {i} Charge Power",
        "showlegend": False,
        "stackgroup": "charge_rate",
        "line": {"width": 0}
    }
    for i, profile in enumerate(data)
], rows=2, cols=1)

fig.add_traces([
    {
        "type": "scatter",
        "x": profile["time"],
        "y": profile["location"],
        "name": f"EV {i} Location",
        "showlegend": False,
    }
    for i, profile in enumerate(data)
], rows=3, cols=1)

fig.show()
# data = pd.read_csv("out383.csv")
# print(data.iloc[:, data.columns.get_level_values(1)=='time'])
#
# data.index = pd.to_datetime(data["time"])
# print(data)