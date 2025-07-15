import pandas as pd
import json

"""
Dados obtidos pela CardioID durante cada trail
- Drowsiness.csv (niveis de sonolência)
- ECG (eletrocardiograma)
- IBI.csv (Inter Bit Intreval)
- Ignition.csv
- info
- LOD_Event
"""

# # Load the CSV file
# file_path = '../datasetsPart2/valu3s/sim/simulator_1_elvagar/cardioID/2023_03_06T15_42_44__UYVc3/Drowsiness.csv'
# data = pd.read_csv(file_path, delimiter=';')

# path to the JSON file
file_path = '../datasetsPart2/valu3s/sim/simulator_1_elvagar/cardioID/2023_03_06T15_42_44__UYVc3/info.json'

# eead the JSON file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# content of JSON file
print("Content of info.json:")
print(json_data)

# info
print("UUID:", json_data["uuid"])
print("Trip start:", json_data["trip_start"])
print("Trip end:", json_data["trip_end"])
print("Distance:", json_data["distance"])
print("Duration:", json_data["duration"])
print("Data types:", json_data["dtypes"])
print("Vehicle UUID:", json_data["vehicle"]["uuid"])
print("Vehicle name:", json_data["vehicle"]["name"])
print("Vehicle owner UUID:", json_data["vehicle"]["owner"]["uuid"])
print("Vehicle owner name:", json_data["vehicle"]["owner"]["name"])
print("Gateway UUID:", json_data["vehicle"]["gw_uuid"])
print("Inactive parameters:", json_data["vehicle"]["inactive_params"])
print("Inactive parameters timestamp:", json_data["vehicle"]["inactive_params_ts"])