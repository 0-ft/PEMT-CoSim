"""Generate a gridlabd json file (dict)
Generate a dict file including loads/generators based on a template file.
It is able to scale
"""

import json

import glm

"""1. load a template file from a existing system"""
data = glm.load("./TE_Challenge.glm")
# houses_dict = {
#     obj['attributes']['name']: {
#         'attributes': obj['attributes'],
#         'children': obj['children']
#     }
#     for obj in data['objects']
#     if obj['name'] == 'house'
# }

with open('objects.json', 'w') as f:
    json.dump({'houses_list': [obj for obj in data['objects'] if obj['name'] == 'house']}, f, indent=4)

# area_list = [float(house['attributes']['floor_area']) for house in houses_list]
