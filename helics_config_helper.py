import json

from scenario import PETScenario


def pub(obj, prop, prop_type, info=False):
    return {
        "global": False,
        "key": f"{obj}#{prop}",
        "type": prop_type
    } | ({"info": {"object": obj, "property": prop}} if info else {})


def sub(source, obj, prop, prop_type, info=False, target_name=None):
    return {
        "key": f"{source}/{obj}#{prop}",
        "type": prop_type
    } | ({"info": {"object": obj, "property": target_name or prop}} if info else {})


class HelicsConfigHelper:
    def __init__(self, scenario: PETScenario):
        self.gridlab_config = {
            "name": "gridlabd",
            "period": scenario.minimum_timestep,
            "uninterruptible": False,
            "wait_for_current_time_update": True,
            "offset": -0.01,
            "publications": self.gridlabd_other_pubs(), "subscriptions": self.gridlabd_other_subs()
        }
        self.pet_config = {
            "name": "substation",
            # "period": 1,
            "uninterruptible": False,
            "publications": self.pet_other_pubs(), "subscriptions": self.pet_other_subs()
        }
        for i in range(scenario.num_houses):
            self.add_house(i, has_ev=i < scenario.num_ev, has_pv=i < scenario.num_pv)

    def add_meter(self, meter_name, billing=False):
        gridlab_pubs_meter = [
            pub(meter_name, prop, prop_type, True)
            for prop, prop_type in
            [("measured_real_power", "double")]#, ("measured_voltage_1", "complex")]#, ("measured_real_power", "double")]
        ]

        gridlab_subs_meter = [
            sub("substation", meter_name, prop, prop_type, True)
            for prop, prop_type in
            [("bill_mode", "string"), ("monthly_fee", "double"), ("price", "double")]
        ] if billing else []

        self.gridlab_config["publications"] += gridlab_pubs_meter
        self.gridlab_config["subscriptions"] += gridlab_subs_meter
        pet_subs_meter = [
            sub("gridlabd", meter_name, prop, prop_type, False)
            for prop, prop_type in
            [("measured_real_power", "double")]#, ("measured_voltage_1", "complex")]#, ("measured_real_power", "double")]
        ]

        pet_pubs_meter = [
            pub(meter_name, prop, prop_type, False)
            for prop, prop_type in
            [("bill_mode", "string"), ("monthly_fee", "double"), ("price", "double")]
        ] if billing else []

        self.pet_config["subscriptions"] += pet_subs_meter
        self.pet_config["publications"] += pet_pubs_meter

    def add_pv(self, solar_name):
        self.add_meter(f"{solar_name}_meter", billing=False)
        pet_subs_pv = [
            sub("gridlabd", solar_name, prop, prop_type, False)
            for prop, prop_type in [("I_Out", "double"), ("V_Out", "double")]
        ]

        pet_pubs_pv = [
            pub(f"{solar_name}_inv", prop, prop_type, False)
            for prop, prop_type in [("P_Out", "double"), ("Q_Out", "double")]
        ]

        self.pet_config["subscriptions"] += pet_subs_pv
        self.pet_config["publications"] += pet_pubs_pv

        gridlabd_subs_pv = [
            sub("substation", f"{solar_name}_inv", prop, prop_type, True)
            for prop, prop_type in [("P_Out", "double"), ("Q_Out", "double")]
        ]

        gridlabd_pubs_pv = [
            pub(solar_name, prop, prop_type, True)
            for prop, prop_type in [("V_Out", "double"), ("I_Out", "double")]
        ]
        self.gridlab_config["subscriptions"] += gridlabd_subs_pv
        self.gridlab_config["publications"] += gridlabd_pubs_pv

    def add_ev(self, ev_name):
        self.add_meter(f"{ev_name}_meter", billing=False)

        pet_subs_ev = [
            sub("ev", ev_name, prop, prop_type, False)
            for prop, prop_type in
            [("location", "string"), ("charging_load", "double"), ("soc", "double"), ("stored_energy", "double"),
             ("max_charging_load", "double"), ("min_charging_load", "double")]
        ]

        pet_pubs_ev = [
            pub(ev_name, prop, prop_type, False)
            for prop, prop_type in [("charge_rate", "double")]
        ]

        self.pet_config["subscriptions"] += pet_subs_ev
        self.pet_config["publications"] += pet_pubs_ev

        gridlabd_subs_ev = [
            sub("ev", f"{ev_name}", "charging_load", "complex", True, "constant_power_A")
        ]

        self.gridlab_config["subscriptions"] += gridlabd_subs_ev

    def add_house(self, house_index, has_ev, has_pv):
        house_name = f"H{house_index}"
        self.add_meter(f"H{house_index}_meter_billing", billing=True)
        self.add_meter(f"H{house_index}_meter_house", billing=False)

        gridlab_pubs_house = [
            pub(house_name, prop, prop_type, True)
            for prop, prop_type in [("hvac_load", "double"), ("air_temperature", "double"), ("total_load", "double"),
                                    ("power_state", "string")]
        ]

        gridlab_subs_hvac = [
            sub("substation", f"{house_name}", prop, prop_type, True)
            for prop, prop_type in
            [("cooling_setpoint", "double"), ("thermostat_mode", "string")]
        ]

        # gridlab_subs_billing = [
        #     sub(f"{house_name}_meter_billing", prop, prop_type, True)
        #     for prop, prop_type in [("bill_mode", "string"), ("monthly_fee", "double"), ("price", "double")]
        # ]

        self.gridlab_config["publications"] += gridlab_pubs_house
        self.gridlab_config["subscriptions"] += gridlab_subs_hvac

        pet_pubs_house = [
            pub(house_name, prop, prop_type, False)
            for prop, prop_type in [("thermostat_mode", "string"), ("cooling_setpoint", "double")]
        ]

        pet_subs_house = [
            sub("gridlabd", house_name, prop, prop_type, False)
            for prop, prop_type in [("hvac_load", "double"), ("air_temperature", "double"), ("total_load", "double"),
                                    ("power_state", "string")]
        ]

        self.pet_config["publications"] += pet_pubs_house
        self.pet_config["subscriptions"] += pet_subs_house
        if has_ev:
            self.add_ev(f"{house_name}_ev")

        if has_pv:
            self.add_pv(f"{house_name}_solar")

    def gridlabd_other_pubs(self):
        return [
            {
                "global": False,
                "key": "distribution_load",
                "type": "complex",
                "info": {
                    "object": "network_node",
                    "property": "distribution_load"
                }
            },
            {
                "global": False,
                "key": "grid_meter#measured_real_power",
                "type": "double",
                "info": {
                    "object": "grid_meter",
                    "property": "measured_real_power"
                }
            },
        ]

    def gridlabd_other_subs(self):
        return [
            {
                "key": "pypower/three_phase_voltage_B7",
                "type": "complex",
                "info": {
                    "object": "network_node",
                    "property": "positive_sequence_voltage"
                }
            },
            {
                "key": "weather/temperature",
                "type": "double",
                "info": {
                    "object": "weather",
                    "property": "temperature"
                }
            },
            {
                "key": "weather/humidity",
                "type": "double",
                "info": {
                    "object": "weather",
                    "property": "humidity"
                }
            },
            {
                "key": "weather/solar_direct",
                "type": "double",
                "info": {
                    "object": "weather",
                    "property": "solar_direct"
                }
            },
            {
                "key": "weather/solar_diffuse",
                "type": "double",
                "info": {
                    "object": "weather",
                    "property": "solar_diffuse"
                }
            },
            {
                "key": "weather/pressure",
                "type": "double",
                "info": {
                    "object": "weather",
                    "property": "pressure"
                }
            },
            {
                "key": "weather/wind_speed",
                "type": "double",
                "info": {
                    "object": "weather",
                    "property": "wind_speed"
                }
            }
        ]

    def pet_other_subs(self):
        return [
            {
                "key": "pypower/LMP_B7",
                "type": "double"
            },
            {
                "key": "gridlabd/distribution_load",
                "type": "complex"
            },
            {
                "key": "gridlabd/grid_meter#measured_real_power",
                "type": "double"
            },
            {
                "key": "weather/temperature",
                "type": "double"
            }
        ]

    def pet_other_pubs(self):
        return [
            {
                "key": "clear_price",
                "type": "double",
                "global": False
            },
            {
                "key": "unresponsive_mw",
                "type": "double",
                "global": False
            },
            {
                "key": "responsive_max_mw",
                "type": "double",
                "global": False
            },
            {
                "key": "responsive_c2",
                "type": "double",
                "global": False
            },
            {
                "key": "responsive_c1",
                "type": "double",
                "global": False
            },
            {
                "key": "responsive_deg",
                "type": "integer",
                "global": False
            }
        ]
