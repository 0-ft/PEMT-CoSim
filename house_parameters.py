import pickle
import random

with open("template_houses.pkl", "rb") as f:
    template_houses: list = pickle.load(f)

def generate_house_parameters():
    template_house = random.choice(template_houses)
    air_temperature = float(template_house['attributes']['air_temperature']) + round(random.uniform(-1, 1))
    skew = int(template_house['attributes']['schedule_skew']) + random.randint(-10, 10)
    ZIP_code = ""
    for child in template_house['children']:
        if child['name'] == 'ZIPload':
            ZIP_code += "object ZIPload {\n"
            for attr in child['attributes']:
                if attr == 'schedule_skew':
                    ZIP_code += '  ' + attr + ' ' + str(skew) + ';\n'
                else:
                    ZIP_code += '  ' + attr + ' ' + child['attributes'][attr] + ';\n'
            ZIP_code += '};\n'

    return {
        "skew": skew,
        "Rroof": float(template_house['attributes']['Rroof']) + round(random.uniform(-1, 1), 2),
        "Rwall": float(template_house['attributes']['Rwall']) + round(random.uniform(-1, 1), 2),
        "Rfloor": float(template_house['attributes']['Rfloor']) + round(random.uniform(-1, 1), 2),
        "Rdoors": int(template_house['attributes']['Rdoors']),
        "Rwindows": float(template_house['attributes']['Rwindows']) + round(random.uniform(-0.1, 0.1), 2),
        "airchange_per_hour": float(template_house['attributes']['airchange_per_hour']) + round(
            random.uniform(-0.1, 0.1), 2),
        "total_thermal_mass_per_floor_area": float(
            template_house['attributes']['total_thermal_mass_per_floor_area']) + round(
            random.uniform(-0.2, 0.2), 2),
        "cooling_COP": float(template_house['attributes']['cooling_COP']) + round(random.uniform(-0.1, 0.1), 2),
        "floor_area": float(template_house['attributes']['floor_area']) + round(random.uniform(-20, 20), 2),
        "number_of_doors": int(template_house['attributes']['number_of_doors']),
        "air_temperature": air_temperature,
        "mass_temperature": air_temperature,
        "ZIP_code": ZIP_code
    }
