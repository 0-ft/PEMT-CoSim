import pickle
from datetime import datetime, timedelta
from os import listdir
from sys import argv
from time import time as millis

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
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

START_TIME = datetime.strptime('2013-07-05 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')
END_TIME = datetime.strptime('2013-07-09 00:00:00 -0800', '%Y-%m-%d %H:%M:%S %z')

def rate_integ(series):
    total_s = (series.index.max() - series.index.min()).total_seconds()
    seconds = (series.index - series.index.min()).total_seconds()
    trap = trapezoid(y=series.values, x=seconds)
    return trap / total_s


def load_plot(h, grid_power_cap=100000):
    solar_supply = np.abs(h["houses"]["sum.pv.solar_power"])
    grid_supply = np.maximum(h["grid"]["vpp_load_p"], 0)
    supply_breakdown = make_subplots(rows=1, cols=1)

    w_to_kwhd = lambda x: x / 1000 * 24

    hvac_rate = rate_integ(h["houses"]["sum.hvac.hvac_load"])
    print(f"HVAC Load {hvac_rate} W = {w_to_kwhd(hvac_rate)} kWh/day")
    unresp_rate = rate_integ(h["houses"]["sum.unresponsive_load"])
    print(f"Unresp Load {unresp_rate} W = {w_to_kwhd(unresp_rate)} kWh/day")
    total_rate = unresp_rate + hvac_rate
    print(f"Total Load {total_rate} W = {w_to_kwhd(total_rate)} kWh/day")

    pv_rate = rate_integ(h["houses"]["sum.pv.solar_power"])
    print(f"PV Supply {pv_rate} W = {w_to_kwhd(pv_rate)} kWh/day")

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

    grid_cap = pd.Series(np.ones_like(h["grid"].index, dtype=float) * grid_power_cap, index=h["grid"].index)
    pv_average_capacity = rate_integ(h["houses"]["sum.pv.max_power"])
    print(f"PV Capacity Average: {pv_average_capacity} = {w_to_kwhd(pv_average_capacity)} kWh/day")

    grid_cap_average = rate_integ(grid_cap)
    print(f"Grid Capacity Average: {grid_cap_average} = {w_to_kwhd(grid_cap_average)} kWh/day")

    pv_surp = rate_integ(h["houses"]["sum.pv.max_power"] - solar_supply)
    print(f"PV Surplus: {pv_surp} = {w_to_kwhd(pv_surp)} kWh/day")

    supply_breakdown.add_traces([
        {
            "type": "scatter",
            "x": supply.index,
            "y": supply,
            "name": f"{name}",
            "line": {"width": 2, "color": color, "dash": "dash"}
        } for name, supply, color in [
            ("Grid Supply Capacity", grid_cap, colors["grid"]),
            ("PV Supply Capacity", h["houses"]["sum.pv.max_power"], colors["pv"])
        ]
    ], rows=1, cols=1)

    supply_breakdown.update_xaxes(row=1, col=1, tickformat="%H:%M")
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
    load_breakdown.update_xaxes(title_text="", row=1, col=1, tickformat="%H:%M")
    load_breakdown.update_yaxes(title_text="Power (W)", row=1, col=1, rangemode="tozero")
    layout(supply_breakdown, 1200, 400)
    layout(load_breakdown, 1200, 400)
    return supply_breakdown, load_breakdown
    # return fig


def hvac_plot(day_means, h):
    hvac = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    ftoc = lambda x: (x - 32) * 5 / 9
    setpoints = h["houses"]["values.hvac.set_point"].apply(lambda x: Series(ftoc(np.array(x))))
    airtemps = h["houses"]["values.hvac.air_temp"].apply(lambda x: Series(ftoc(np.array(x))))
    diffs = np.maximum(airtemps - setpoints, 0)
    diffs_sq = np.power(diffs, 2)
    mean_diffs_sq = diffs_sq.mean(axis=1)
    mean_diffs_sq = df_days_mean(DataFrame({"diffs": mean_diffs_sq}), True)
    # mean_diffs_sq = DataFrame({"diffs": mean_diffs_sq})
    hvac.add_traces([
        {
            "type": "scatter",
            "x": quant.index,
            "y": quant,
            "name": f"{name}",
        } for name, quant in [
            ("Min House Air Temp", ftoc(day_means["houses"]["min.hvac.air_temp"])),
            ("Max House Air Temperature", ftoc(day_means["houses"]["max.hvac.air_temp"])),
            ("Mean House Air Temperature", ftoc(day_means["houses"]["mean.hvac.air_temp"])),
            ("Mean House Set Point", ftoc(day_means["houses"]["mean.hvac.set_point"])),
            ("Weather Temperature", ftoc(day_means["grid"]["weather_temp"])),
        ]
    ], rows=1, cols=1)

    hvac.add_trace(
        {
            "type": "scatter",
            "x": mean_diffs_sq.index,
            "y": mean_diffs_sq["diffs"],
            "name": "$\mkern 1.5mu\overline{\mkern-1.5mu T_{excess}^2 \mkern-1.5mu}(t)\mkern 1.5mu$",
            "line": {"dash": "dash"},
            "showlegend": True
        }, row=1, col=1, secondary_y=True
    )

    # seconds = (diff.index - diff.index.min()).total_seconds()
    # excess_quant = trapezoid(y=diff.values, x=seconds)
    # print(f"EXC: {excess_quant} Ks = {excess_quant / 60 / 24} Kmin/day")

    # hvac.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    hvac.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    hvac.update_yaxes(title_text="$\mkern 1.5mu\overline{\mkern-1.5mu T_{excess}^2 \mkern-1.5mu}(t)\mkern 1.5mu$", row=1,
                      col=1, secondary_y=True)
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


def ev_plot(h):
    ev = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    ev.add_traces([
        {
            "type": "scatter",
            "x": quant.index,
            "y": quant,
            "name": f"{name}",
            "stackgroup": "load",
            "showlegend": True
        } for name, quant in [
            ("Driving", h["houses"]["sum.ev.driving_load"]),
            ("Charging/Discharging", -h["houses"]["sum.ev.charging_load"])
        ]
    ], rows=1, cols=1)

    ev.add_trace(
        {
            "type": "scatter",
            "x": h["houses"]["sum.ev.stored_energy"].index,
            "y": h["houses"]["sum.ev.stored_energy"],
            "name": f"Total EV Stored Energy",
            "showlegend": True
        }, row=1, col=1, secondary_y=True
    )

    # price.update_xaxes(title_text="Time", row=1, col=1, tickformat="%H:%M")
    ev.update_yaxes(title_text="Power (W)", row=1, col=1)
    ev.update_yaxes(title_text="Energy (J)", row=1, col=1, secondary_y=True)
    layout(ev, 1200, 400)
    return ev


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


def df_days_mean(df: DataFrame, resample=False):
    groups = list(df.groupby(df.index.dayofyear))
    stack = pd.concat([df.set_index(df.index.time) for doy, df in groups])
    means = stack.groupby(stack.index).mean()
    dated = means.set_index(means.index.map(lambda t: datetime.combine(START_TIME.date(), t)))
    if resample:
        dated = dated.resample(timedelta(seconds=300)).mean()
    return dated


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
                k: ts.resample(timedelta(seconds=300)).mean()
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


def one_figs_capped(hs, name):
    house_means = days_mean(hs, {
        "houses": [
            "mean.hvac.air_temp", "max.hvac.air_temp", "min.hvac.air_temp", "mean.hvac.set_point",
            "sum.hvac.hvac_load", "sum.unresponsive_load",
            "sum.pv.solar_power", "sum.ev.charging_load",
            "sum.pv.max_power", "sum.ev.driving_load", "sum.ev.stored_energy"
        ],
        "grid": ["weather_temp", "vpp_load_p"],
    }, resample=True)
    house_means = hs

    hvac = hvac_plot(house_means[0], hs[0])
    hvac.write_html(f"figs/{name}_hvac.html")
    hvac.write_image(f"figs/{name}_hvac.png", scale=1)

    # der_supply = house_means[0]["houses"]["sum.pv.solar_power"] - np.minimum(
    #     house_means[0]["houses"]["sum.ev.charging_load"], 0)
    # total_load = house_means[0]["houses"]["sum.unresponsive_load"] + house_means[0]["houses"][
    #     "sum.hvac.hvac_load"] + np.maximum(house_means[0]["houses"]["sum.ev.charging_load"], 0)
    # grid_supply = house_means[0]["grid"]["vpp_load_p"]

    supply_breakdown, load_breakdown = load_plot(house_means[0])
    supply_breakdown.write_html(f"figs/{name}_supply.html")
    supply_breakdown.write_image(f"figs/{name}_supply.png")
    load_breakdown.write_html(f"figs/{name}_load.html")
    load_breakdown.write_image(f"figs/{name}_load.png")

    price_means = days_mean(hs, {
        "auction": ["average_price", "lmp"]
    }, resample=True)
    # price_means = hs

    price = price_plot(price_means[0])
    price.write_html(f"figs/{name}_price.html")
    price.write_image(f"figs/{name}_price.png", scale=1)
    # print(max(total_l), min(total_l), max(total_l) - min(total_l))

    ev = ev_plot(house_means[0])
    ev.write_html(f"figs/{name}_ev.html")
    ev.write_image(f"figs/{name}_ev.png")

def all_single_figs():
    pkls = [f.replace(".pkl", "") for f in listdir("metrics") if "pkl" in f]
    for pkl in pkls:
        print(f"\n\n{pkl}")
        hs = [pickle.load(open(f"metrics/{pkl}.pkl", "rb"))]
        end_time = min(END_TIME, hs[0]["houses"].index.max(), hs[0]["houses"].index.max())
        hs = [
            {k: h[k][(START_TIME <= h[k].index) & (h[k].index < end_time)] for k in h.keys()}
            for h in hs
        ]
        one_figs_capped(hs, pkl)

if __name__ == "__main__":
    # samegraph([h1, h2], [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], [argv[1], argv[2]], ["HVAC Load (W)"])
    if len(argv) == 1:
        all_single_figs()
    elif len(argv[1:]) == 1:
        hs = [pickle.load(open(f"metrics/{a}.pkl", "rb")) for a in argv[1:]]
        end_time = min(END_TIME, hs[0]["houses"].index.max(), hs[0]["houses"].index.max())
        hs = [
            {k: h[k][(START_TIME <= h[k].index) & (h[k].index < end_time)] for k in h.keys()}
            for h in hs
        ]
        one_figs_capped(hs, argv[1])

    else:
        hs = [pickle.load(open(f"metrics/{a}.pkl", "rb")) for a in argv[1:]]
        end_time = min(END_TIME, hs[0]["houses"].index.max(), hs[0]["houses"].index.max())
        hs = [
            {k: h[k][(START_TIME <= h[k].index) & (h[k].index < end_time)] for k in h.keys()}
            for h in hs
        ]
        sameplot(hs, [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], argv[1:], ["HVAC Load (W)"])
        multiplot(hs, [("houses", "sum.hvac.hvac_load", False, "HVAC Load")], argv[1:], ["HVAC Load (W)"],
                  layout=(1, 2))
    # ev_history = pickle.load(open(f"../fed_ev/{argv[1]}_ev_history.pkl", "rb"))

    # SubstationRecorder.make_figure_bids(history, "bids_metrics")
    # fig
    # fig_compare(history, argv[1], make_html=True, ev_history=ev_history)
