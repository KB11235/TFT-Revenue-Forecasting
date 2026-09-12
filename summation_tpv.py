import pyarrow.parquet as pq
import pandas as pd
import csv
import os
import re
import glob
import numpy as np

table = pq.read_table("training_data.parquet")

df_train = table.to_pandas(ignore_metadata=True)
df_train["request_date"] = pd.to_datetime(df_train["request_date"])

min_date = df_train["request_date"].min()
df_train["time_idx"] = (df_train["request_date"] - min_date).dt.days.astype(int)

col_names = ["merchant_id", "time_idx", "date", "q35", "q50", "q65", "expected_gbp"]

csv_file_path = "predictions_final.csv"
with open(csv_file_path, mode = 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(col_names)

input_folder = r"C:\Users\kirin\OneDrive\Desktop\Forecast\v2_out\forecast_merchant"
output_file = "predictions_final.csv"

input_files = glob.glob(os.path.join(input_folder, "forecast_merchant_*.csv"))

for input_file in input_files:
    filename = os.path.basename(input_file)
    match = re.search(r"forecast_merchant_(\d+)\.csv", filename)
    if not match:
        continue
    merchant_id = int(match.group(1))

    with open(input_file, mode = "r", newline = "") as infile:
        reader = csv.reader(infile)
        next(reader)
        with open(output_file, mode = "a", newline = "") as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow([merchant_id] + row)

df_pred = pd.read_csv("predictions_final.csv")
df_pred["date"] = pd.to_datetime(df_pred["date"])
df_pred = df_pred.rename(columns={"date": "request_date"})

def day_month_year(frame):
    frame["request_month"] = (frame["request_date"]).dt.month.astype(int)
    frame["request_day"] = (frame["request_date"]).dt.day.astype(int)
    frame["request_year"] = (frame["request_date"]).dt.year.astype(int)
    return frame

df_train = day_month_year(df_train)
df_pred = day_month_year(df_pred)

df_train["request_date"] = pd.to_datetime(df_train["request_date"])
df_pred["request_date"] = pd.to_datetime(df_pred["request_date"])

max_date_train = df_train["request_date"].max()

merchant_max_dates = df_train.groupby("merchant_id")["request_date"].max()

inactive_merchants = merchant_max_dates[max_date_train - merchant_max_dates > pd.Timedelta(days = 7)].index

df_train = df_train[~df_train["merchant_id"].isin(inactive_merchants)]
df_pred = df_pred[~df_pred["merchant_id"].isin(inactive_merchants)]

df_pred["dispersion_q35"] = df_pred["expected_gbp"]/df_pred["q35"]
df_pred["dispersion_q50"] = df_pred["expected_gbp"]/df_pred["q50"]
df_pred["dispersion_q65"] = df_pred["expected_gbp"]/df_pred["q65"]

def sorting(frame):
    frame = frame.sort_values(by = ["request_date", "merchant_id"])
    return frame

df_train = sorting(df_train)
df_pred = sorting(df_pred)

df_train_add = df_train[["request_date", "merchant_id", "time_idx", "request_year", "request_month", "request_day"]].copy()

df_train_add["daily_expected_q35"] = df_train["daily_successful_amount_gbp"].values
df_train_add["daily_expected_q50"] = df_train["daily_successful_amount_gbp"].values
df_train_add["daily_expected_q65"] = df_train["daily_successful_amount_gbp"].values

df_pred_add = df_pred[["request_date", "merchant_id", "time_idx", "request_year", "request_month", "request_day"]].copy()
df_pred_add["daily_expected_q35"] = np.select(
    [df_pred["q35"] == 0, df_pred["dispersion_q35"] >= 6],
    [df_pred["expected_gbp"], (df_pred["expected_gbp"] + df_pred["q35"])/2],
    default = df_pred["q35"]
)
df_pred_add["daily_expected_q50"] = np.select(
    [df_pred["q50"] == 0, df_pred["dispersion_q50"] >= 3],
    [df_pred["expected_gbp"], (df_pred["expected_gbp"] + df_pred["q50"])/2],
    default = df_pred["q50"]
)
df_pred_add["daily_expected_q65"] = np.select(
    [df_pred["q65"] == 0, df_pred["dispersion_q65"] >= 2],
    [df_pred["expected_gbp"], (df_pred["expected_gbp"] + df_pred["q65"])/2],
    default = df_pred["q65"]
)

df_pred_add = df_pred_add[
    [
        "request_date",
        "merchant_id",
        "daily_expected_q35",
        "daily_expected_q50",
        "daily_expected_q65",
        "time_idx",
        "request_year",
        "request_month",
        "request_day"
    ]
]

inputs = [df_train_add, df_pred_add]
df_res = pd.concat(inputs, axis = 0)

df_res["daily_expected_q35"] = df_res["daily_expected_q35"].astype(float)
df_res["daily_expected_q50"] = df_res["daily_expected_q50"].astype(float)
df_res["daily_expected_q65"] = df_res["daily_expected_q65"].astype(float)

merchant_max_dates = df_train.groupby("merchant_id")["request_date"].max().reset_index()
min_max_date = merchant_max_dates["request_date"].min()
min_max_cutoff_date = min_max_date.replace(day = 1)

df_res = df_res[df_res["request_date"] >= min_max_cutoff_date]

df_res.loc[df_res["daily_expected_q35"] < 0.01, "daily_expected_q35"] = 0
df_res.loc[df_res["daily_expected_q50"] < 0.01, "daily_expected_q50"] = 0
df_res.loc[df_res["daily_expected_q65"] < 0.01, "daily_expected_q65"] = 0

df_monthly_pred = (df_res
    .groupby(["merchant_id", "request_year", "request_month"])
    [["daily_expected_q35", "daily_expected_q50", "daily_expected_q65"]]
    .sum()
    .reset_index()
)

df_monthly_pred.to_csv("predictions_by_month.csv")