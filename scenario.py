import pickle

import numpy as np

wakeup_start_lo = 5.0
wakeup_start_hi = 6.5
daylight_start_lo = 8.0
daylight_start_hi = 9.0
evening_start_lo = 17.0
evening_start_hi = 18.5
night_start_lo = 22.0
night_start_hi = 23.5

wakeup_set_lo = 78.0
wakeup_set_hi = 80.0
daylight_set_lo = 84.0
daylight_set_hi = 86.0
evening_set_lo = 78.0
evening_set_hi = 80.0
night_set_lo = 72.0
night_set_hi = 74.0

weekend_day_start_lo = 8.0
weekend_day_start_hi = 9.0
weekend_day_set_lo = 76.0
weekend_day_set_hi = 84.0
weekend_night_start_lo = 22.0
weekend_night_start_hi = 24.0
weekend_night_set_lo = 72.0
weekend_night_set_hi = 74.0

ramp_lo = 0.5
ramp_hi = 3.0
deadband_lo = 2.0
deadband_hi = 3.0
offset_limit_lo = 2.0
offset_limit_hi = 4.0
ctrl_cap_lo = 1.0
ctrl_cap_hi = 3.0


class PETScenario:
    def __init__(self, minimum_timestep=1, market_period=300, num_houses=30, num_pv=30, num_ev=30):
        self.minimum_timestep = minimum_timestep
        self.market_period = market_period
        self.num_houses = num_houses
        self.num_pv = num_pv
        self.num_ev = num_ev
        self.hvac_configs = [self.generate_hvac_config() for _ in range(num_houses)]

    def generate_hvac_config(self):
        wakeup_start = np.random.uniform(wakeup_start_lo, wakeup_start_hi)
        daylight_start = np.random.uniform(daylight_start_lo, daylight_start_hi)
        evening_start = np.random.uniform(evening_start_lo, evening_start_hi)
        night_start = np.random.uniform(night_start_lo, night_start_hi)
        wakeup_set = np.random.uniform(wakeup_set_lo, wakeup_set_hi)
        daylight_set = np.random.uniform(daylight_set_lo, daylight_set_hi)
        evening_set = np.random.uniform(evening_set_lo, evening_set_hi)
        night_set = np.random.uniform(night_set_lo, night_set_hi)
        weekend_day_start = np.random.uniform(weekend_day_start_lo, weekend_day_start_hi)
        weekend_day_set = np.random.uniform(weekend_day_set_lo, weekend_day_set_hi)
        weekend_night_start = np.random.uniform(weekend_night_start_lo, weekend_night_start_hi)
        weekend_night_set = np.random.uniform(weekend_night_set_lo, weekend_night_set_hi)
        deadband = np.random.uniform(deadband_lo, deadband_hi)
        return {
            "wakeup_start": wakeup_start, "daylight_start": daylight_start, "evening_start": evening_start,
            "night_start": night_start, "wakeup_set": wakeup_set, "daylight_set": daylight_set,
            "evening_set": evening_set, "night_set": night_set, "weekend_day_start": weekend_day_start,
            "weekend_day_set": weekend_day_set, "weekend_night_start": weekend_night_start,
            "weekend_night_set": weekend_night_set, "deadband": deadband
        }

    def save(self, path):
        with open(f"{path}/scenario.pkl", "wb") as f:
            pickle.dump(self, f)
