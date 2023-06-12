import pickle
from datetime import datetime, timedelta
from sys import argv
from time import time as millis

import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly
from scipy.integrate import trapezoid

colors = plotly.colors.DEFAULT_PLOTLY_COLORS
colors = {
    "pv": "#2ebde8",
    "ev": "#31c480",
    "grid": "#df65f0",
    "hvac": "#f5be3d",
    "unresp": colors[4],
    "total": "black",
}

START_TIME = datetime.strptime('2013-07-02 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
END_TIME = datetime.strptime('2013-07-06 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')


def rate_integ(series):
    total_s = (series.index.max() - series.index.min()).total_seconds()
    seconds = (series.index - series.index.min()).total_seconds()
    trap = trapezoid(y=series.values, x=seconds)
    return trap / total_s


def load_plot(h, grid_power_cap=100000):
    solar_supply = np.abs(h["houses"]["sum.pv.solar_power"])
    grid_supply = np.maximum(h["grid"]["vpp_load_p"], 0)
    supply_breakdown = make_subplots(rows=1, cols=1)
    print(supply_breakdown.layout.colorway)
    supply_breakdown.add_traces([
        {
            "type": "scatter",
            "x": supply.index,
            "y": supply,
            "name": f"{name}",
            "line": {"width": 0, "color": color},
            "stackgroup": "supply",
        } for name, supply, color in [
            ("Grid Supply", grid_supply, colors["grid"]),
            ("PV Supply", solar_supply, colors["pv"]),
            ("EV Supply", -np.minimum(h["houses"]["sum.ev.charging_load"], 0), colors["ev"])
        ]
    ], rows=1, cols=1)

    total_load = h["houses"]["sum.unresponsive_load"] + h["houses"][
        "sum.hvac.hvac_load"] + np.maximum(h["houses"]["sum.ev.charging_load"], 0)

    supply_breakdown.add_trace({
        "type": "scatter",
        "x": total_load.index,
        "y": total_load,
        "name": f"Total Load",
        "line": {"color": colors["total"]},
    }, row=1, col=1)

    grid_cap = np.ones_like(h["grid"].index, dtype=float) * grid_power_cap
    max_pv = h["houses"]["sum.pv.max_power"]

    supply_breakdown.add_traces([
        {
            "type": "scatter",
            "x": supply.index,
            "y": supply,
            "name": f"{name}",
            "line": {"width": 2, "color": color, "dash": "dash"},
        } for name, supply, color in [
            ("Available Grid Supply", grid_cap, colors["grid"]),
            ("Available EV Supply", max_pv, colors["pv"])
        ]
    ], rows=1, cols=1)

    # supply_breakdown.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    supply_breakdown.update_yaxes(title_text="Power (W)", row=1, col=1, rangemode="tozero")

    load_breakdown = make_subplots(rows=1, cols=1)

    load_breakdown.add_traces([
        {
            "type": "scatter",
            "x": load.index,
            "y": load,
            "name": f"{name}",
            "line": {"width": 0, "color": color},
            "stackgroup": "supply",
        } for name, load, color in [
            ("Unresponsive Load", h["houses"]["sum.unresponsive_load"], colors["unresp"]),
            ("HVAC Load", h["houses"]["sum.hvac.hvac_load"], colors["hvac"]),
            ("EV Load", np.maximum(h["houses"]["sum.ev.charging_load"], 0), colors["ev"])
        ]
    ], rows=1, cols=1)

    total_supply = grid_supply + solar_supply - np.minimum(h["houses"]["sum.ev.charging_load"], 0)

    load_breakdown.add_trace({
        "type": "scatter",
        "x": total_supply.index,
        "y": total_supply,
        "name": f"Total Supply",
        "line": {"color": colors["total"]},
    }, row=1, col=1)
    # load_breakdown.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    load_breakdown.update_yaxes(title_text="Power (W)", row=1, col=1, rangemode="tozero")
    layout(supply_breakdown, 1200, 400)
    layout(load_breakdown, 1200, 400)
    return supply_breakdown, load_breakdown
    # return fig

def hvac_plot(h):
    hvac = make_subplots(rows=1, cols=1)

    ftoc = lambda x : (x - 32) * 5 / 9
    hvac.add_traces([
        {
            "type": "scatter",
            "x": quant.index,
            "y": quant,
            "name": f"{name}",
        } for name, quant in [
            ("Min House Air Temp", ftoc(h["houses"]["min.hvac.air_temp"])),
            ("Max House Air Temperature", ftoc(h["houses"]["max.hvac.air_temp"])),
            ("Mean House Air Temperature", ftoc(h["houses"]["mean.hvac.air_temp"])),
            ("Mean House Set Point", ftoc(h["houses"]["mean.hvac.set_point"])),
            ("Weather Temperature", ftoc(h["grid"]["weather_temp"])),
        ]
    ], rows=1, cols=1)

    setpoint = ftoc(h["houses"]["mean.hvac.set_point"])
    meantemp = ftoc(h["houses"]["mean.hvac.air_temp"])
    diff = np.maximum(meantemp - setpoint, 0)
    seconds = (diff.index - diff.index.min()).total_seconds()
    excess_quant = trapezoid(y=diff.values, x=seconds)
    print(f"EXC: {excess_quant} Ks = {excess_quant / 60 / 24} Kmin/day")
    # hvac.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    hvac.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    layout(hvac, 1200, 400)
    return hvac

def price_plot(h):
    price = make_subplots(rows=1, cols=1)

    price.add_traces([
        {
            "type": "scatter",
            "x": quant.index,
            "y": quant,
            "name": f"{name}",
        } for name, quant in [
            ("LMP", h["auction"]["lmp"]),
            ("VWAP", h["auction"]["average_price"])
        ]
    ], rows=1, cols=1)

    # price.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    price.update_yaxes(title_text="Price ($)", row=1, col=1)
    layout(price, 1200, 400)
    return price


def oneplot(h, keys, scenario_name, ax_names):
    has_y2 = any([axis for _, _, axis, _, _, _ in keys])
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": has_y2}]])
    print(keys)
    for varnum, (table, key, axis, name, stackgroup, scale) in enumerate(keys):
        print(
            f"{name}: min={h[table][key].min()}, max={h[table][key].max()}, mean={h[table][key].mean()} range={h[table][key].max() - h[table][key].min()}")
        fig.add_trace(
            {
                "type": "scatter",
                "x": h[table][key].index,
                "y": h[table][key].map(scale) if scale else h[table][key],
                "name": f"{name}",
                "line": {"color": colors[varnum], "width": 0 if stackgroup else 2},
                "stackgroup": stackgroup,
            }, row=1, col=1, secondary_y=axis)

    fig.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    fig.update_yaxes(title_text=ax_names[0], row=1, col=1, rangemode="tozero")
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


def one_figs_uncapped(hs):
    house_means = days_mean(hs, {
        "houses": [
            "mean.hvac.air_temp", "max.hvac.air_temp", "min.hvac.air_temp", "mean.hvac.set_point",
            "sum.hvac.hvac_load", "sum.unresponsive_load"
        ],
        "grid": ["weather_temp"],
    }, resample=True)
    # hvac = oneplot(house_means[0], [
    #     ("houses", "mean.hvac.air_temp", False, "Mean House Air Temperature", None, lambda x: (x - 32) * 5 / 9),
    #     ("houses", "max.hvac.air_temp", False, "Max House Air Temperature", None, lambda x: (x - 32) * 5 / 9),
    #     ("houses", "min.hvac.air_temp", False, "Min House Air Temperature", None, lambda x: (x - 32) * 5 / 9),
    #     ("houses", "mean.hvac.set_point", False, "Mean House Set Point", None, lambda x: (x - 32) * 5 / 9),
    #     ("grid", "weather_temp", False, "Weather Temperature", None, lambda x: (x - 32) * 5 / 9),
    # ], argv[1], ["Temperature (°C)"])
    # layout(hvac)
    # hvac.write_html(f"figs/{argv[1]}_hvac.html")
    # hvac.write_image(f"figs/{argv[1]}_hvac.png", scale=1)

    load = oneplot(house_means[0], [
        ("houses", "sum.unresponsive_load", False, "Unresponsive Load", "load", None),
        ("houses", "sum.hvac.hvac_load", False, "HVAC Load", "load", None),
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
        ("auction", "average_price", False, "VWAP", None, None),
        # ("auction", "lmp", False, "VWAP \= LMP", None),
    ], argv[1], ["Price ($)"])
    layout(price)
    price.write_html(f"figs/{argv[1]}_price.html")
    price.write_image(f"figs/{argv[1]}_price.png", scale=1)
    # print(max(total_l), min(total_l), max(total_l) - min(total_l))


def one_figs_capped(hs):

    house_means = days_mean(hs, {
        "houses": [
            "mean.hvac.air_temp", "max.hvac.air_temp", "min.hvac.air_temp", "mean.hvac.set_point",
            "sum.hvac.hvac_load", "sum.unresponsive_load",
            "sum.pv.solar_power", "sum.ev.charging_load"
        ],
        "grid": ["weather_temp", "vpp_load_p"],
    }, resample=True)
    hvac = hvac_plot(house_means[0])
    hvac.write_html(f"figs/{argv[1]}_hvac.html")
    hvac.write_image(f"figs/a_{argv[1]}_hvac.png", scale=1)

    der_supply = house_means[0]["houses"]["sum.pv.solar_power"] - np.minimum(
        house_means[0]["houses"]["sum.ev.charging_load"], 0)
    total_load = house_means[0]["houses"]["sum.unresponsive_load"] + house_means[0]["houses"][
        "sum.hvac.hvac_load"] + np.maximum(house_means[0]["houses"]["sum.ev.charging_load"], 0)
    grid_supply = house_means[0]["grid"]["vpp_load_p"]

    supply_breakdown, load_breakdown = load_plot(house_means[0])
    supply_breakdown.write_html(f"figs/{argv[1]}_supply.html")
    supply_breakdown.write_image(f"figs/a_{argv[1]}_supply.png")
    load_breakdown.write_html(f"figs/{argv[1]}_load.html")
    load_breakdown.write_image(f"figs/a_{argv[1]}_load.png")

    pv_rate = rate_integ(hs[0]["houses"]["sum.pv.solar_power"])
    print(f"PV {pv_rate} W = {pv_rate * 3600 * 24} J/day = {pv_rate * 3600 * 24 / 3.6e6} kWh/day")
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

    price = price_plot(price_means[0])
    price.write_html(f"figs/{argv[1]}_price.html")
    price.write_image(f"figs/a_{argv[1]}_price.png", scale=1)
    # print(max(total_l), min(total_l), max(total_l) - min(total_l))


if __name__ == "__main__":
    print(argv[1])

    hs = [pickle.load(open(f"metrics/{a}.pkl", "rb")) for a in argv[1:]]
    end_time = min(END_TIME, hs[0]["houses"].index.max(), hs[0]["houses"].index.max())
    hs = [
        {k: h[k][(START_TIME <= h[k].index) & (h[k].index < end_time)] for k in h.keys()}
        for h in hs
    ]
    # samegraph([h1, h2], [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], [argv[1], argv[2]], ["HVAC Load (W)"])
    if len(hs) == 1:
        one_figs_capped(hs)

    else:
        sameplot(hs, [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], argv[1:], ["HVAC Load (W)"])
        multiplot(hs, [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], argv[1:], ["HVAC Load (W)"],
                  layout=(1, 2))
    # ev_history = pickle.load(open(f"../fed_ev/{argv[1]}_ev_history.pkl", "rb"))

    # SubstationRecorder.make_figure_bids(history, "bids_metrics")
    # fig
    # fig_compare(history, argv[1], make_html=True, ev_history=ev_history)
