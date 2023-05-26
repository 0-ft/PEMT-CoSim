from emobpy import Availability, ModelSpecs


# PET EV controller
# uses mobility/grid-availability data from emobpy to simulate an EV responding to PET
# market conditions according to a strategy.
class V2GEV:
    def __init__(self, grid_availability: Availability, car_model: ModelSpecs):
        # self.mobility = mobility
        # self.consumption = consumption
        self.grid_availability = grid_availability
        self.car_model = car_model
