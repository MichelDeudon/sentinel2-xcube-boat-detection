# Get the Coordinates of the chosen months, dates and times
from pathlib import Path
import json
import datetime
import pandas as pd
import glob
import os
# def getLatLonColor(selectedData, month, day):
#     listCoords = totalList[month][day]
#
#     # No times selected, output all times for chosen month and date
#     if selectedData is None or len(selectedData) is 0:
#         return listCoords
#     listStr = "listCoords["
#     for time in selectedData:
#         if selectedData.index(time) is not len(selectedData) - 1:
#             listStr += "(totalList[month][day].index.hour==" + str(int(time)) + ") | "
#         else:
#             listStr += "(totalList[month][day].index.hour==" + str(int(time)) + ")]"
#     return eval(listStr)

def get_selection(barSelection, locationSelection):
    # breakpoint()
    # print(os.path.join("data", str(locationSelection).lower() + "*.json"))
    if locationSelection:
        json_file = glob.glob(os.path.join("data", str(locationSelection).lower() + "*.json"))[0]
        # print(json_file)
        with open(str(json_file)) as f:
            json_obj = json.load(f)
            x_vals = []
            y_vals =[]
            for each in json_obj:
                for k, v in each.items():
                    date_picked = datetime.datetime.strptime(k, "%Y-%m-%d")
                    x_vals.append(date_picked)
                    y_vals.append(round(v, 2))
            color_vals = ["#24D249" for _ in range(len(x_vals))]
            return x_vals, y_vals, color_vals
    else:
        return [], [], []

def get_monthly_aggregation():
    df = pd.DataFrame(columns=["location", "time", "count"])
    json_files = Path("data").glob("*.json")

    for location_json_file in json_files:
        # print(location_json_file)
        with open(str(location_json_file)) as f:
            json_obj = json.load(f)
            for each in json_obj:
                for time, count in each.items():
                    df = df.append({"location": location_json_file.stem, "time": time, "count": count}, ignore_index=True)

    df["time"] = pd.to_datetime(df["time"])
    mean_df = df.set_index("time").resample("M").mean()
    color_vals = ["#24D249" for _ in range(len(mean_df))]
    mean_df["color"] = mean_df.index.map(lambda t:  "#FF5050" if t.year == 2020 else "#24D249")
    # breakpoint()
    return mean_df.index,  mean_df["count"],  mean_df["color"]

# get_monthly_aggregation()