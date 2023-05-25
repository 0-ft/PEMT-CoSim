import collections
import json
data = json.load(open("TE_Challenge_HELICS_gld_msg.json", "r"))

keys = [x["key"] for x in data["publications"]]
print(keys)
print(len(keys), len(set(keys)))
print([item for item, count in collections.Counter(keys).items() if count > 1])