import pickle
from datetime import datetime, timedelta
from sys import argv
from time import time as millis

import pandas as pd
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly
from scipy.integrate import trapezoid

colors = plotly.colors.DEFAULT_PLOTLY_COLORS

START_TIME = datetime.strptime('2013-07-02 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
END_TIME = datetime.strptime('2013-07-04 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')


def rate_integ(series):
    total_s = (series.index.max() - series.index.min()).total_seconds()
    seconds = (series.index - series.index.min()).total_seconds()
    trap = trapezoid(y=series.values, x=seconds)
    return trap / total_s


def oneplot(h, keys, scenario_name, ax_names, scale=None):
    has_y2 = any([axis for _, _, axis, _, _ in keys])
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": has_y2}]])
    print(keys)
    for varnum, (table, key, axis, name, stackgroup) in enumerate(keys):
        print(
            f"{name}: min={h[table][key].min()}, max={h[table][key].max()}, mean={h[table][key].mean()} range={h[table][key].max() - h[table][key].min()}")
        fig.add_trace(
            {
                "type": "scatter",
                "x": h[table][key].index,
                "y": h[table][key].map(scale) if scale else h[table][key],
                "name": f"{name}",
                "line": {"color": colors[varnum], "width": 2},
                "stackgroup": stackgroup,
            }, row=1, col=1, secondary_y=axis)

    fig.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    fig.update_yaxes(title_text=ax_names[0], row=1, col=1)
    if len(ax_names) > 1:
        fig.update_yaxes(title_text=ax_names[1], row=1, col=1, secondary_y=True)
    return fig


def sameplot(hs, keys, scenario_names, ax_names):
    has_y2 = any([axis for _, _, axis, _ in keys])
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": has_y2}]])
    for varnum, (table, key, axis, name) in enumerate(keys):
        fig.add_traces([
            {
                "type": "scatter",
                "x": h[table].index,
                "y": h[table][key],
                "name": f"{scenario_names[i]}",
                "legendgroup": name,
                "legendgrouptitle_text": name,
                "line": {"color": colors[i], "width": 2},
            }
            for i, h in enumerate(hs)
        ], rows=1, cols=1, secondary_ys=[axis] * len(hs))
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text=ax_names[0], row=1, col=1)
    if len(ax_names) > 1:
        fig.update_yaxes(title_text=ax_names[1], row=1, col=1, secondary_y=True)
    return fig


def multiplot(hs, keys, scenario_names, ax_names, layout, size=1000):
    rows, cols = layout
    has_y2 = any([axis for _, _, axis, _ in keys])
    specs = [
        [
            {"secondary_y": has_y2} for col in range(cols)
        ] for row in range(rows)
    ]
    # print(specs)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, specs=specs, subplot_titles=scenario_names)
    for i, h in enumerate(hs):
        row, col = i // cols, i % cols
        for varnum, (table, key, axis, name) in enumerate(keys):
            print(
                f"{name}: min={h[table][key].min()}, max={h[table][key].max()}, mean={h[table][key].mean()}, range={h[table][key].max() - h[table][key].min()}}}")
            fig.add_traces([
                {
                    "type": "scatter",
                    "x": h[table].index,
                    "y": h[table][key],
                    "name": name,
                    "line": {"color": colors[varnum], "width": 2},
                    "showlegend": i == 0,
                }
            ], rows=row + 1, cols=col + 1, secondary_ys=[axis] * len(hs))

        fig.update_xaxes(title_text="Time", row=row + 1, col=col + 1)
        fig.update_yaxes(title_text=ax_names[0], row=row + 1, col=col + 1)
        if len(ax_names) > 1:
            fig.update_yaxes(title_text=ax_names[1], row=row + 1, col=col + 1, secondary_y=True)

    return fig


def days_mean(hs: list[dict[str: DataFrame]], cols, resample=False):
    hs = [
        {
            k: list(df[cols[k]].groupby(df.index.dayofyear))
            for k, df in h.items() if k in cols
        }
        for h in hs
    ]
    hs = [
        {
            k: pd.concat([df.set_index(df.index.time) for doy, df in groups])
            for k, groups in h.items()
        }
        for h in hs
    ]
    hs = [
        {
            k: stack.groupby(stack.index).mean()
            for k, stack in h.items()
        }
        for h in hs
    ]
    hs = [
        {
            k: ts.set_index(ts.index.map(lambda t: datetime.combine(START_TIME.date(), t)))
            for k, ts in h.items()
        }
        for h in hs
    ]
    if resample:
        hs = [
            {
                k: ts.resample(timedelta(minutes=3)).mean()
                for k, ts in h.items()
            }
            for h in hs
        ]
    return hs


def layout(fig):
    fig.update_layout(
        width=1000, height=500, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
        ),
        font=dict(size=18))


def one_figs_capped(hs):
    house_means = days_mean(hs, {
        "houses": [
            "mean.hvac.air_temp", "max.hvac.air_temp", "min.hvac.air_temp", "mean.hvac.set_point",
            "sum.hvac.hvac_load", "sum.unresponsive_load"
        ],
        "grid": ["weather_temp"],
    }, resample=True)
    hvac = oneplot(house_means[0], [
        ("houses", "mean.hvac.air_temp", False, "Mean House Air Temperature", None),
        ("houses", "max.hvac.air_temp", False, "Max House Air Temperature", None),
        ("houses", "min.hvac.air_temp", False, "Min House Air Temperature", None),
        ("houses", "mean.hvac.set_point", False, "Mean House Set Point", None),
        ("grid", "weather_temp", False, "Weather Temperature", None),
    ], argv[1], ["Temperature (°C)"], scale=lambda x: (x - 32) * 5 / 9)
    layout(hvac)
    hvac.write_html(f"figs/{argv[1]}_hvac.html")
    hvac.write_image(f"figs/{argv[1]}_hvac.png", scale=1)

    load = oneplot(house_means[0], [
        ("houses", "sum.unresponsive_load", False, "Unresponsive Load", "load"),
        ("houses", "sum.hvac.hvac_load", False, "HVAC Load", "load"),
    ], argv[1], ["Load (W)"])
    layout(load)
    load.write_html(f"figs/{argv[1]}_load.html")
    load.write_image(f"figs/{argv[1]}_load.png", scale=1)

    hvac_rate = rate_integ(house_means[0]["houses"]["sum.hvac.hvac_load"])
    print(f"HVAC {hvac_rate} W = {hvac_rate * 3600 * 24} J/day = {hvac_rate * 3600 * 24 / 3.6e6} kWh/day")
    unresp_rate = rate_integ(house_means[0]["houses"]["sum.unresponsive_load"])
    print(f"Unresp {unresp_rate} W = {unresp_rate * 3600 * 24} J/day = {unresp_rate * 3600 * 24 / 3.6e6} kWh/day")
    total_rate = unresp_rate + hvac_rate
    print(f"Total {total_rate} W = {total_rate * 3600 * 24} J/day = {total_rate * 3600 * 24 / 3.6e6} kWh/day")

    price_means = days_mean(hs, {
        "auction": ["average_price", "lmp"]
    })
    # price_means = hs

    price = oneplot(price_means[0], [
        ("auction", "average_price", False, "VWAP", None),
        # ("auction", "lmp", False, "VWAP \= LMP", None),
    ], argv[1], ["Price ($)"])
    layout(price)
    price.write_html(f"figs/{argv[1]}_price.html")
    price.write_image(f"figs/{argv[1]}_price.png", scale=1)
    # print(max(total_l), min(total_l), max(total_l) - min(total_l))

def one_figs_uncapped(hs):
    house_means = days_mean(hs, {
        "houses": [
            "mean.hvac.air_temp", "max.hvac.air_temp", "min.hvac.air_temp", "mean.hvac.set_point",
            "sum.hvac.hvac_load", "sum.unresponsive_load",
            "sum.pv.solar_power"
        ],
        "grid": ["weather_temp"],
    }, resample=True)
    hvac = oneplot(house_means[0], [
        ("houses", "mean.hvac.air_temp", False, "Mean House Air Temperature", None),
        ("houses", "max.hvac.air_temp", False, "Max House Air Temperature", None),
        ("houses", "min.hvac.air_temp", False, "Min House Air Temperature", None),
        ("houses", "mean.hvac.set_point", False, "Mean House Set Point", None),
        ("grid", "weather_temp", False, "Weather Temperature", None),
    ], argv[1], ["Temperature (°C)"], scale=lambda x: (x - 32) * 5 / 9)
    layout(hvac)
    hvac.write_html(f"figs/{argv[1]}_hvac.html")
    hvac.write_image(f"figs/{argv[1]}_hvac.png", scale=1)

    load = oneplot(house_means[0], [
        ("houses", "sum.unresponsive_load", False, "Unresponsive Load", "load"),
        ("houses", "sum.hvac.hvac_load", False, "HVAC Load", "load"),
        ("houses", "sum.pv.solar_power", False, "PV Supply", "load"),
    ], argv[1], ["Load (W)"])
    layout(load)
    load.write_html(f"figs/{argv[1]}_load.html")
    load.write_image(f"figs/{argv[1]}_load.png", scale=1)

    hvac_rate = rate_integ(house_means[0]["houses"]["sum.hvac.hvac_load"])
    print(f"HVAC {hvac_rate} W = {hvac_rate * 3600 * 24} J/day = {hvac_rate * 3600 * 24 / 3.6e6} kWh/day")
    unresp_rate = rate_integ(house_means[0]["houses"]["sum.unresponsive_load"])
    print(f"Unresp {unresp_rate} W = {unresp_rate * 3600 * 24} J/day = {unresp_rate * 3600 * 24 / 3.6e6} kWh/day")
    total_rate = unresp_rate + hvac_rate
    print(f"Total {total_rate} W = {total_rate * 3600 * 24} J/day = {total_rate * 3600 * 24 / 3.6e6} kWh/day")

    price_means = days_mean(hs, {
        "auction": ["average_price", "lmp"]
    })
    # price_means = hs

    price = oneplot(price_means[0], [
        ("auction", "average_price", False, "VWAP", None),
        # ("auction", "lmp", False, "VWAP \= LMP", None),
    ], argv[1], ["Price ($)"])
    layout(price)
    price.write_html(f"figs/{argv[1]}_price.html")
    price.write_image(f"figs/{argv[1]}_price.png", scale=1)
    # print(max(total_l), min(total_l), max(total_l) - min(total_l))


if __name__ == "__main__":
    # print(argv[1])
    hs = [pickle.load(open(f"metrics/{a}.pkl", "rb")) for a in argv[1:]]
    print(hs[0]["houses"].index)
    end_time = min(END_TIME, hs[0]["houses"].index.max(), hs[0]["houses"].index.max())
    hs = [
        {k: h[k][(START_TIME <= h[k].index) & (h[k].index < end_time)] for k in h.keys()}
        for h in hs
    ]
    # samegraph([h1, h2], [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], [argv[1], argv[2]], ["HVAC Load (W)"])
    if len(hs) == 1:
        one_figs_uncapped(hs)
    else:
        sameplot(hs, [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], argv[1:], ["HVAC Load (W)"])
        multiplot(hs, [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], argv[1:], ["HVAC Load (W)"],
                  layout=(1, 2))
    # ev_history = pickle.load(open(f"../fed_ev/{argv[1]}_ev_history.pkl", "rb"))

    # SubstationRecorder.make_figure_bids(history, "bids_metrics")
    # fig
    # fig_compare(history, argv[1], make_html=True, ev_history=ev_history)
