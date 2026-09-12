#!/usr/bin/env python3
"""
Temporal Fusion Transformer for daily merchant revenue (v2).

BigQuery query -> preprocess -> TimeSeriesDataSet -> dataloaders -> baseline ->
Trainer.fit -> best checkpoint -> validation plots -> interpretation -> one future
forecast per merchant (plus a verbose single-merchant report for whichever id
``--forecast-merchant`` names)
The reasoning behind every change is in MODEL_NOTES.md.  The short version:

  1. The target is log-transformed *in the dataframe* (``revenue_log``) instead of
     inside the normaliser.  pytorch_forecasting inverts the normaliser's
     ``transformation`` *before* the loss is evaluated, so a QuantileLoss on a
     log1p/expm1 GroupNormalizer is still a QuantileLoss in raw GBP - and raw GBP
     pinball loss is owned entirely by a handful of whale merchants.
  2. Every merchant is reindexed onto a complete daily calendar and missing days
     are filled with zero revenue, because "no transactions" is an observation of
     zero, not a missing value.  ``allow_missing_timesteps`` is therefore off.
  3. All calendar covariates are derived from the date by one function that is
     used for history *and* for the future horizon, so the encoder and the decoder
     can never disagree about what "day_of_week = 5" means.
  4. Normalisation is genuinely per merchant (``groups=["merchant_id"]``); the
     default ``GroupNormalizer()`` has ``groups=None`` and is global.
  5. Train / validation / test are three disjoint time blocks, and validation is
     restricted to merchants that are actually alive in the validation window.

Usage
-----
    python3 tft_revenue_v2.py --selftest              # fast sanity checks, no training
    python3 tft_revenue_v2.py --smoke                 # tiny end-to-end run (~minutes)
    python3 tft_revenue_v2.py                         # full run
    python3 tft_revenue_v2.py --refit-full --forecast-merchant 171
    python3 tft_revenue_v2.py --checkpoint <ckpt> --skip-train --forecast-merchant 171
    python3 tft_revenue_v2.py --refit-full --forecast-plots 25   # PNGs for the 25 biggest only
    python3 tft_revenue_v2.py --no-forecast-all       # only the one --forecast-merchant

Every merchant that survives filter_series gets a forecast: one
``v2_out/forecast_merchant/forecast_merchant_<id>.csv`` each (time_idx, date, p35, p50, p65,
expected value), a chart beside it, and one row per merchant in ``v2_out/forecast_summary.csv``
saying how far behind that merchant's data is, whether it had a trained embedding, and whether
its level and dispersion checks pass.  Longer horizons than the trained 30 days are a separate
question with a separate answer - see future_considerations.md.
"""

from __future__ import annotations

from google.cloud import bigquery
import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd

# BigQuery NUMERIC/DATE columns come back as db_dtypes extension dtypes.  The
# original script only got away with reading the parquet because importing
# google.cloud.bigquery registers them as a side effect; be explicit instead.
try:  # pragma: no cover - environment dependent
    import db_dtypes  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

import matplotlib
import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.data.encoders import Expm1Transform


from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

torch.set_float32_matmul_precision("high")


# ======================================================================================
# 0. The BigQuery feed
# ======================================================================================
# Changes from ``TFT model 1 final.py`` are marked CHANGED / NEW.  Everything this script
# needs comes from these columns plus the extract's own calendar fields, so the parquet is
# self-sufficient.
#
# IMPORTANT: the payouts exclusion below changes the TARGET DEFINITION, so it only takes
# effect once you re-run this query.  ``training_data2.parquet`` was produced by the old
# query and still has payouts inside daily_successful_amount_gbp - roughly 9% of the
# target amount population-wide, and 100% of it for merchant 763.  Nothing in the parquet
# lets the script subtract them after the fact: the extract carries payouts_amount, which
# is a transaction COUNT, not the GBP value.  See MODEL_NOTES.md Part 4.2.

# CHANGED: payouts are money leaving the merchant, not inbound processing volume, so
# they no longer count as revenue.  Mean successful payout is GBP 509.55 against 
# GBP 50.06 for card, so despite being only ~1% of transactions they are ~9% of the amount.
TRANSACTION_QUERY = """SELECT
    request_date,
    merchant_id,
    COUNT(*) AS daily_transaction_amount,
    SUM(CASE WHEN UPPER(response_result) = 'SUCCESS' THEN 1 ELSE 0 END) AS daily_successful_transaction_count,
    SUM(CASE WHEN UPPER(response_result) = 'SUCCESS' AND UPPER(payment_type) != 'PAYOUTS' 
    AND UPPER(transaction_type) != "REFUND" OR "VOID" THEN amount_gbp ELSE 0 END) AS daily_successful_amount_gbp,

    SUM(CASE WHEN UPPER(response_result) = 'SUCCESS' AND UPPER(payment_type) = 'PAYOUTS' THEN amount_gbp ELSE 0 END) AS daily_payout_amount_gbp,
    SUM(CASE WHEN UPPER(transaction_type) = 'SUCCESS' AND UPPER(payment_type) != 'PAYOUTS' 
    AND UPPER(transaction_type) = 'REFUND' THEN amount_gbp ELSE 0 END) AS daily_refund_amount_gbp,
    SUM(CASE WHEN UPPER(transaction_type) = 'SUCCESS' AND UPPER(payment_type) != 'PAYOUTS' 
    AND UPPER(transaction_type) = 'VOID' THEN amount_gbp ELSE 0 END) AS daily_void_amount_gbp,
    SUM(CASE WHEN UPPER(response_result) != 'SUCCESS' THEN amount_gbp ELSE 0 END) AS daily_unsuccessful_amount_gbp,

    ANY_VALUE(calendar_year) AS calendar_year,
    ANY_VALUE(calendar_month) AS calendar_month,
    ANY_VALUE(month_name) AS month_name,
    ANY_VALUE(calendar_quarter) AS calendar_quarter,
    ANY_VALUE(calendar_week_of_year) AS calendar_week_of_year,
    ANY_VALUE(is_uk_bank_holiday) AS is_uk_bank_holiday,
    ANY_VALUE(day_of_week) AS day_of_week,
    ANY_VALUE(day_of_month) AS day_of_month,
    ANY_VALUE(day_name) AS day_name,
    ANY_VALUE(day_type) AS day_type,
    ANY_VALUE(merchant_name) AS merchant_name,
    ANY_VALUE(merchant_category) AS merchant_category,
    ANY_VALUE(sector_group) AS sector_group,
    ANY_VALUE(iso_id) AS iso_id,
    ANY_VALUE(iso_name) AS iso_name,
    ANY_VALUE(merchant_created_datetime) AS merchant_created_datetime,

    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'MIGRATION' THEN 1 ELSE 0 END) AS migration_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'ACQUIRED.COM' THEN 1 ELSE 0 END) AS acquired_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'TRUST PAYMENTS' THEN 1 ELSE 0 END) AS trust_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'CASHFLOWS' THEN 1 ELSE 0 END) AS cashflows_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'BARCLAYCARD' THEN 1 ELSE 0 END) AS barclaycard_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'LLOYDS CARDNET' THEN 1 ELSE 0 END) AS lloyds_amount,
    SUM(CASE WHEN UPPER(acquiring_bank_name) = 'SHIFT4' THEN 1 ELSE 0 END) AS shift4_amount,

    SUM(CASE WHEN UPPER(response_result) = 'DECLINED' THEN 1 ELSE 0 END) AS declined_amount,
    SUM(CASE WHEN UPPER(response_result) = 'EXPIRED' THEN 1 ELSE 0 END) AS expired_amount,
    SUM(CASE WHEN UPPER(response_result) = 'CANCELLED' THEN 1 ELSE 0 END) AS cancelled_amount,
    SUM(CASE WHEN UPPER(response_result) = 'PENDING' THEN 1 ELSE 0 END) AS pending_amount,
    SUM(CASE WHEN UPPER(response_result) = 'ERROR' THEN 1 ELSE 0 END) AS error_amount,
    SUM(CASE WHEN UPPER(response_result) = 'QUARANTINED' THEN 1 ELSE 0 END) AS quarantined_amount,
    SUM(CASE WHEN UPPER(response_result) = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_response_amount,
    SUM(CASE WHEN UPPER(response_result) = 'BLOCKED' THEN 1 ELSE 0 END) AS blocked_amount,

    SUM(CASE WHEN UPPER(payment_type) = 'CARD' THEN 1 ELSE 0 END) AS card_amount,
    SUM(CASE WHEN UPPER(payment_type) = 'DIRECT_DEBIT' THEN 1 ELSE 0 END) AS dd_amount,
    SUM(CASE WHEN UPPER(payment_type) = 'PAY_BY_BANK' THEN 1 ELSE 0 END) AS pbb_amount,
    SUM(CASE WHEN UPPER(payment_type) = 'PAYOUTS' THEN 1 ELSE 0 END) AS payouts_amount,
    SUM(CASE WHEN UPPER(payment_type) = 'BACS_RECEIVE' THEN 1 ELSE 0 END) AS bacs_amount,
    SUM(CASE WHEN UPPER(payment_type) = 'VARIABLE_RECURRING_PAYMENT' THEN 1 ELSE 0 END) AS vrp_amount,

    SUM(CASE WHEN UPPER(integration_api) = 'CORE' THEN 1 ELSE 0 END) AS core_amount,
    SUM(CASE WHEN UPPER(integration_api) = 'REST' THEN 1 ELSE 0 END) AS rest_amount,

    SUM(CASE WHEN has_fraud_notification IS TRUE THEN 1 ELSE 0 END) AS daily_fraud_amount,
    SUM(CASE WHEN has_dispute IS TRUE THEN 1 ELSE 0 END) AS daily_dispute_amount

FROM `prj-p-dat-datawarehouse-90m9.analytics.transaction`
GROUP BY
    request_date,
    merchant_id
ORDER BY
    request_date,
    merchant_id;"""

# WHERE NOT (month_name IN ('June', 'July') AND calendar_year = 2026) --> for presentation
# WHERE NOT (month_name = 'July' AND calendar_year = 2026) --> for presentation

client = bigquery.Client(project = 'prj-p-dat-datawarehouse-90m9')
df = client.query(TRANSACTION_QUERY).to_dataframe()
# df.to_parquet("training_datajunjul.parquet") OR "training_datajul.parquet" OR "training_data.parquet"

# 1. Configuration
# ======================================================================================
@dataclass
class Config:
    data_path: str = "training_data.parquet" # Column names rather than actual variables, except data_path
    raw_target: str = "daily_successful_amount_gbp"   # tpv in GBP, as extracted
    target: str = "tpv_log"                      # log1p of it - what the model fits
    group_col: str = "merchant_id"                   # one time series per value
    date_col: str = "request_date"

    horizon: int = 91 # length of the forecast horizon
    max_encoder_length: int = 120
    min_encoder_length: int = 28  # 4 whole weeks: the least that identifies weekday shape
    val_days: int = 180  # length of the validation block (2 horizons of origins)

    # --- series filtering -------------------------------------------------------------
    min_series_days: int = 30
    min_log_std: float = 0.20  # drop merchants whose log-tpv barely moves
    min_active_days: int = 20  # at least this many non-zero-tpv days
    val_stride: int = 7  # keep every 7th validation origin (all weekday phases)
    # False keeps each merchant's pre-first-sale zero days, so the ramp-up from zero is
    # in the training data.
    trim_leading_zeros: bool = True

    # --- model ------------------------------------------------------------------------
    hidden_size: int = 96
    attention_head_size: int = 4
    hidden_continuous_size: int = 32
    lstm_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 5e-3
    # Weight decay pulls the rarely-updated ones back toward zero in order to reduce overfitting risk
    weight_decay: float = 5e-3
    optimizer: str = "adamw"
    # Larger than the previous 0.1, deliberately.  0.1 was compensating for a loss that
    # could reach 25,000 in GBP; now that the loss is O(1) in log space, clipping at 0.1
    # truncates almost every honest gradient and slows learning to a crawl.  0.5 still
    # catches a genuine spike.
    gradient_clip_val: float = 0.5
    group_id_dropout: float = 0.15  # cold-start robustness, see RevenueTFT
    # 0.35 and 0.65 are here because they are what the forecast CSV reports: a narrow band
    # either side of the median is the only part of a log-space predictive distribution that
    # survives expm1 as a usable number.  They have to be in the trained set - QuantileLoss
    # gives the head one output channel per level, so a quantile that was not trained cannot
    # be interpolated afterwards.  The outer levels are kept for the coverage diagnostics in
    # gbp_metrics(), which reference 0.1/0.9 and the extremes.
    quantiles: tuple[float, ...] = (0.02, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 0.98)
    # Levels written to forecast_merchant_<id>.csv, in order.
    report_quantiles: tuple[float, ...] = (0.35, 0.5, 0.65)

    # --- optimisation -----------------------------------------------------------------
    batch_size: int = 128
    max_epochs: int = 60 
    patience: int = 8 
    train_batches_per_epoch: int = 400  # fixed step budget so epochs are comparable
    reduce_on_plateau_patience: int = 3
    num_workers: int | None = None
    seed: int = 42 

    # --- output -----------------------------------------------------------------------
    # Terminal output still happens - every metric is printed.  The directory holds what a
    # terminal cannot: predictions_holdout.csv (one row per merchant per forecast day with
    # all seven quantiles - this is what you sort to find the merchants the model is worst
    # at), forecast_merchant/forecast_merchant_<id>.csv for every merchant plus
    # forecast_summary.csv over them, forecast_merchant_<id>.csv/.png for the one merchant
    # reported verbatim on stdout, the example and interpretation plots, config.json,
    # metrics.json, and the TensorBoard/CSV logs.  Nothing lands there that is not also
    # summarised on stdout.
    outdir: str = "v2_out"
    # The one merchant that gets the long-form treatment: the full quantile table printed to
    # stdout, the per-merchant sanity checks in prose, and a csv/png at the top level of
    # outdir.  It is NOT what decides which merchants are forecast - forecast_all_merchants
    # does every merchant including this one - it only decides which one is worth reading in
    # the terminal, and therefore which one gets a duplicate pair of files outside
    # forecast_dir.  "none" turns it off and leaves only the all-merchant pass.
    forecast_merchant: str = "171"
    # Sub-directory of outdir holding one forecast_merchant_<id>.csv per merchant.  Its own
    # folder because there are ~380 of them and they would otherwise bury the ~15 files that
    # describe the run itself.
    forecast_dir: str = "forecast_merchant"
    # PNGs written by the all-merchant pass: -1 = every merchant, 0 = none, N > 0 = the N
    # largest by mean daily revenue.  Measured on this data (384 merchants, 882-day
    # histories): the batched forward pass that produces all 384 forecasts takes 4s, and
    # rendering all 384 charts takes 50s at 0.13s each for 58 MB of png.  So the charts do
    # cost an order of magnitude more than the forecasts and are still under a minute against
    # a training run of tens of minutes - which is why they are ON by default.  Set 0 if you
    # only want the csvs, or N to cap it at the merchants that matter by revenue.
    forecast_plots: int = -1
    n_example_plots: int = 10

    def as_dict(self) -> dict: # configures quantiles into a plain dict
        as_plain_dict = asdict(self)
        as_plain_dict["quantiles"] = list(self.quantiles)
        as_plain_dict["report_quantiles"] = list(self.report_quantiles)
        return as_plain_dict

    def __post_init__(self) -> None:
        missing = [t for t in self.report_quantiles if t not in self.quantiles]
        assert not missing, (
            f"report_quantiles {missing} are not in quantiles {list(self.quantiles)} - the "
            f"loss only produces an output channel per trained level, so a level that was "
            f"not trained cannot be reported"
        )


# ======================================================================================
# 2. Calendar features - ONE function, used for history and for the future horizon
# ======================================================================================
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# The extract's own calendar columns.  These are taken verbatim wherever the extract has a
# row for the date; nothing here is recomputed for history.
SOURCE_CALENDAR_COLS = [
    "day_of_week", "day_of_month", "calendar_month", "month_name", "calendar_year",
    "calendar_quarter", "calendar_week_of_year", "day_name", "day_type",
    "is_uk_bank_holiday",
]

# Bank holidays for dates AFTER the extract's last date come from the `holidays` package.
#
# Historical holidays are NOT touched by this - they are read from the warehouse's own
# is_uk_bank_holiday column via build_calendar_lookup(), and this function is only ever
# consulted for dates the warehouse does not cover.  It has to exist because a forecast
# horizon has no rows in the warehouse at all, and because bank holidays are not derivable
# from a date: Good Friday and Easter Monday move every year with the lunar calendar, and
# the substitute-day rule shifts Christmas/Boxing/New Year whenever they land on a weekend.
# A hardcoded list would be correct only until the year it stops at.
#
# `--selftest` validates the package against the warehouse over the range they overlap
# (2024-01-01..2026-07-27) and asserts set equality on all 21 flagged dates, so the
# convention is checked rather than assumed before it is relied on for the future.
UK_HOLIDAY_SUBDIVISION = "England"  # England & Wales share the same bank holidays


def uk_bank_holidays_between(start, end) -> pd.DatetimeIndex:
    """England & Wales bank holidays in [start, end], from the `holidays` package."""
    try:
        import holidays as holidays_pkg
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "the `holidays` package is needed to label bank holidays beyond the extract's "
            "last date - install it with `python3 -m pip install --user holidays`. "
            "Refusing to fall back to 'no holidays': an unlabelled bank holiday inside the "
            "forecast horizon would be silently predicted as an ordinary working day, and "
            "bank holidays land on Mondays, the highest-volume weekday."
        ) from exc

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    calendar = holidays_pkg.country_holidays(
        "GB", subdiv=UK_HOLIDAY_SUBDIVISION, years=list(range(start.year, end.year + 1))
    )
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in calendar)).normalize()
    # The warehouse flags the substitute weekday and never the weekend date itself - checked,
    # 0 of its 21 flagged dates fall on a Saturday or Sunday.  `holidays` lists both, e.g.
    # for 2026 it returns Boxing Day 2026-12-26 (a Saturday) AND 2026-12-28 "(observed)".
    # Dropping the weekend entries is what makes the two conventions identical.
    dates = dates[dates.dayofweek < 5]
    return dates[(dates >= start) & (dates <= end)]

# Populated by prepare_data() so make_future_frame() can reuse the same calendar.
CALENDAR_LOOKUP: pd.DataFrame | None = None


def build_calendar_lookup(df: pd.DataFrame, date_col: str = "request_date") -> pd.DataFrame:
    cols = [c for c in SOURCE_CALENDAR_COLS if c in df.columns]
    lookup = (df[[date_col] + cols]
              .drop_duplicates(subset=[date_col])
              .sort_values(date_col)
              .reset_index(drop=True))
    lookup[date_col] = pd.to_datetime(lookup[date_col]).dt.normalize()
    if "is_uk_bank_holiday" in lookup.columns:
        lookup["is_uk_bank_holiday"] = (
            lookup["is_uk_bank_holiday"].astype("string").str.lower()
            .isin(["true", "1"]).to_numpy()
        )
    print(f"[calendar] lookup built from the extract: {len(lookup):,} dates "
          f"{lookup[date_col].min().date()}..{lookup[date_col].max().date()}, "
          f"columns {cols}")
    return lookup


def extrapolate_calendar(dates: pd.DatetimeIndex,
                         holidays: set | None = None) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if holidays is None:
        holidays = set(uk_bank_holidays_between(dates.min(), dates.max()))
    is_holiday = np.array([d in holidays for d in dates], dtype=bool)
    is_weekend = dates.dayofweek.to_numpy() >= 5

    # BigQuery EXTRACT(WEEK): count whole weeks from the first Sunday of the year; anything
    # before it is week 0.  jan1.dayofweek is Mon=0..Sun=6, so (6 - dow) % 7 is the number
    # of days from 1 January to that year's first Sunday.
    jan1 = pd.to_datetime(dict(year=dates.year, month=1, day=1))
    days_to_first_sunday = (6 - jan1.dt.dayofweek.to_numpy()) % 7
    first_sunday = jan1.to_numpy() + days_to_first_sunday * np.timedelta64(1, "D")
    days_since_first_sunday = (dates.to_numpy() - first_sunday) / np.timedelta64(1, "D")
    week_of_year = np.where(days_since_first_sunday < 0, 0,
                            days_since_first_sunday // 7 + 1).astype("int16")

    return pd.DataFrame({
        "request_date": dates,
        "day_of_week": (dates.dayofweek.to_numpy() + 1).astype("int16"),
        "day_of_month": dates.day.to_numpy().astype("int16"),
        "calendar_month": dates.month.to_numpy().astype("int16"),
        "month_name": [_MONTHS[m - 1] for m in dates.month],
        "calendar_year": dates.year.to_numpy().astype("int16"),
        "calendar_quarter": dates.quarter.to_numpy().astype("int16"),
        "calendar_week_of_year": week_of_year,
        "day_name": [_WEEKDAYS[d] for d in dates.dayofweek],
        "day_type": np.where(is_holiday, "Holiday",
                             np.where(is_weekend, "Weekend", "Working Day")),
        "is_uk_bank_holiday": is_holiday,
    })


def add_calendar_features(df: pd.DataFrame, date_col: str = "request_date",
    calendar: pd.DataFrame | None = None) -> pd.DataFrame: # Attach the warehouse's calendar, then the derived seasonality terms on top
    
    df = df.copy()
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    df[date_col] = dates

    calendar = CALENDAR_LOOKUP if calendar is None else calendar
    if calendar is None:
        raise RuntimeError(
            "No calendar lookup available - call build_calendar_lookup() (prepare_data "
            "does this) before add_calendar_features()"
        )

    # Warehouse values first, for every date it covers.
    df = df.drop(columns=[c for c in SOURCE_CALENDAR_COLS if c in df.columns])
    df = df.merge(calendar.rename(columns={calendar.columns[0]: date_col}),
                  on=date_col, how="left")

    # Every bank holiday this frame can know about.  PAST holidays come from the warehouse's
    # own is_uk_bank_holiday column and are never recomputed.  The `holidays` package is
    # consulted only for the range AFTER the warehouse's coverage ends, which is the one
    # range the warehouse cannot answer for.  (The window is stretched 20 days past the
    # frame so holiday_offset can still see the next holiday from the final rows.)
    warehouse_covers_to = pd.Timestamp(calendar[calendar.columns[0]].max()).normalize()
    warehouse_holidays = set(
        pd.to_datetime(calendar.loc[calendar["is_uk_bank_holiday"].astype(bool),
                                    calendar.columns[0]]).dt.normalize()
    )
    beyond_warehouse = uk_bank_holidays_between(
        warehouse_covers_to + pd.Timedelta(days=1),
        max(dates.max(), warehouse_covers_to) + pd.Timedelta(days=20),
    )
    known_holidays = pd.DatetimeIndex(sorted(warehouse_holidays.union(beyond_warehouse)))

    # Anything left over is beyond the extract, i.e. the forecast horizon.
    uncovered = df.loc[df["day_of_week"].isna(), date_col].unique()
    if len(uncovered):
        extra = extrapolate_calendar(
            pd.DatetimeIndex(uncovered), holidays=set(known_holidays)
        ).rename(columns={"request_date": date_col}).set_index(date_col)
        mask = df[date_col].isin(extra.index)
        for col in extra.columns:
            df.loc[mask, col] = df.loc[mask, date_col].map(extra[col]).to_numpy()
        holiday_dates = list(extra.index[extra["is_uk_bank_holiday"].astype(bool)])
        print(f"[calendar] {len(uncovered)} date(s) beyond the extract's coverage "
              f"({pd.Timestamp(min(uncovered)).date()}..{pd.Timestamp(max(uncovered)).date()}) "
              f"filled from the verified date formulas; bank holidays from the `holidays` "
              f"package (GB/{UK_HOLIDAY_SUBDIVISION}): "
              + (", ".join(str(pd.Timestamp(d).date()) for d in holiday_dates)
                 if holiday_dates else "none in this range")) 

    df["is_uk_bank_holiday"] = (
        df["is_uk_bank_holiday"].astype("string").str.lower().isin(["true", "1"]).to_numpy()
    )

    # --- derived seasonality: the part the warehouse does not provide ----------------
    day_of_year = dates.dt.dayofyear.to_numpy().astype(np.float64)
    for harmonic in (1, 2, 3):
        angle = 2.0 * np.pi * harmonic * day_of_year / 365.25
        df[f"year_sin{harmonic}"] = np.sin(angle)
        df[f"year_cos{harmonic}"] = np.cos(angle)

    # Same idea within the month, so 31 January and 1 February are adjacent rather than
    # opposite ends of a 31-level embedding.  month_length varies (28-31), so the position
    # is expressed as a fraction of the month rather than a raw day number.
    month_length = dates.dt.days_in_month.to_numpy().astype(np.float64)
    day_number = dates.dt.day.to_numpy()
    position_in_month = (day_number.astype(np.float64) - 1.0) / month_length
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * position_in_month
        df[f"month_sin{harmonic}"] = np.sin(angle)
        df[f"month_cos{harmonic}"] = np.cos(angle)

    # --- pay-cycle / month-boundary flags -------------------------------------------
    df["is_month_start"] = (day_number <= 2).astype(np.float32)
    df["is_month_end"] = (day_number >= month_length - 1).astype(np.float32)
    # Salaries in the UK are overwhelmingly paid on the last working day of the month or
    # on the 25th-28th, and consumer card spending, subscription rebills and direct-debit
    # collections all cluster in the days that follow.  25th-to-3rd covers both sides of
    # that boundary in one flag.
    df["is_payday_window"] = ((day_number >= 25) | (day_number <= 3)).astype(np.float32)
    df["is_quarter_end"] = dates.dt.is_quarter_end.to_numpy().astype(np.float32)
    # A monotone "days remaining" ramp.  Derivable from day_of_month and month_length, but
    # only as a non-linear function of a 31-level embedding; as a real it is one weight.
    df["days_to_month_end"] = np.clip(month_length - day_number, 0, 15).astype(np.float32)

    # --- signed distance to the nearest bank holiday, clipped to +/-10 days ----------
    # Revenue does not only move ON a bank holiday: it is pulled forward into the days
    # before and suppressed in the days after.  A binary flag cannot express that, so this
    # gives the network a ramp: negative = that many days after the most recent holiday,
    # positive = that many days before the next one, 0 = the holiday itself.
    #
    # Implementation: take every holiday date known to this frame (warehouse + forward
    # list) as sorted integer nanoseconds. np.searchsorted finds, for each row's date,
    # where it would slot into that sorted array - so the entry before that position is the
    # previous holiday and the entry at it is the next one.  Convert both gaps to days,
    # keep whichever is closer, sign it, and clip to +/-10 so a date in the middle of a
    # holiday-free stretch does not become a huge outlier.
    if len(known_holidays):
        holiday_ns = known_holidays.asi8.astype(np.int64)
        date_ns = dates.astype("int64").to_numpy()
        slot = np.searchsorted(holiday_ns, date_ns)
        previous = holiday_ns[np.clip(slot - 1, 0, len(holiday_ns) - 1)]
        following = holiday_ns[np.clip(slot, 0, len(holiday_ns) - 1)]
        NS_PER_DAY = 86_400_000_000_000
        days_since = (date_ns - previous) / NS_PER_DAY
        days_until = (following - date_ns) / NS_PER_DAY
        signed = np.where(days_since <= days_until, -days_since, days_until)
        df["holiday_offset"] = np.clip(signed, -10, 10).astype(np.float32)
    else:  # pragma: no cover
        df["holiday_offset"] = np.float32(0.0)

    return df

# ======================================================================================
# 3. Loading and dtype coercion
# ======================================================================================

ZERO_ON_INACTIVE_DAY_COLS = [
    # Kept in this list even though log_txn_count was dropped as a model input: the column
    # is still needed to zero-fill inserted grid rows and to derive had_no_attempts.
    # Note every "*_amount" column in this extract is a COUNT, not a currency amount.
    "daily_transaction_amount",
    "daily_successful_transaction_count",
    "daily_successful_amount_gbp",
    "daily_unsuccessful_amount_gbp",
    "daily_fraud_amount",
    "daily_dispute_amount",
]
ACQUIRER_COLS = [
    "migration_amount", "acquired_amount", "trust_amount", "unknown_amount",
    "cashflows_amount", "barclaycard_amount", "lloyds_amount", "shift4_amount",
]
RESPONSE_COLS = [
    "daily_successful_transaction_count", "declined_amount", "expired_amount",
    "cancelled_amount", "pending_amount", "error_amount", "quarantined_amount",
    "unknown_response_amount", "blocked_amount",
]
PAYMENT_COLS = [
    "card_amount", "dd_amount", "pbb_amount", "payouts_amount", "bacs_amount", "vrp_amount",
]

# ``rest``/``core`` are not columns in this extract - ``integration_api`` is a single
# string column whose values are 'core' (89,316 rows) and 'rest' (82,949).  Naming them
# here as if they were per-day counts silently produced nothing, because every ratio
# group is filtered by ``if col in df.columns``.  These are the names the SQL below now
# creates, so the ratios switch on automatically once you re-run the query; until then
# ``build_ratio_features`` prints that it skipped the group.
INTEGRATION_COLS = ["core_amount", "rest_amount"]

# Card-scheme columns dropped: they are absent from this extract, and in the raw
# transaction data ``card_scheme`` is 68% 'unknown', so they would be mostly noise.

# Verified 100% constant per merchant_id in the extract.
STATIC_ATTR_COLS = [
    "merchant_category", "sector_group", "iso_name",
    "merchant_name", "merchant_created_datetime",
]

def load_frame(path: str) -> pd.DataFrame:
    print(f"[data] reading {path}")
    df = pd.read_parquet(path)
    print(f"[data] {len(df):,} rows x {df.shape[1]} columns")
    return df


def coerce_dtypes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Turn BigQuery extension dtypes into plain numpy dtypes.

    ``daily_successful_amount_gbp`` arrives as ``object`` holding
    ``decimal.Decimal`` (BigQuery NUMERIC), and 35 of the 52 columns as nullable
    ``Int64``.  These silently survive a lot of pandas work and then either blow
    up or quietly become NaN on the way to a torch tensor - ``.astype(float)`` on
    the Decimal target turns 25 rows into NaN without raising.
    """
    # Copy so the caller's frame is never mutated: prepare_data() chains six
    # transformations, and in-place edits would make the stage order matter.
    df = df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col]).dt.normalize()
    df[cfg.group_col] = df[cfg.group_col].astype("string").astype(str).str.strip()

    # Every column that must end up as a plain numpy float.  dict.fromkeys() de-duplicates
    # while preserving order - daily_successful_transaction_count is in two of these groups.
    # This is not the ratio calculation; that is build further down in engineer_features.
    numeric_cols = (
        ZERO_ON_INACTIVE_DAY_COLS + ACQUIRER_COLS + RESPONSE_COLS + PAYMENT_COLS
        + INTEGRATION_COLS
    )

    for col in dict.fromkeys(numeric_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # BigQuery SUM() returns NULL when every summand is NULL, so a merchant-day on which
    # all transactions succeeded but all of them had a NULL amount_gbp (7.4% of raw rows
    # have one) comes back with a NULL target *and a positive success count*.  There are
    # 25 such rows across 7 merchants.
    tpv = df[cfg.raw_target]
    success_count = df.get("daily_successful_transaction_count")
    if success_count is not None:
        # target_imputed marks rows whose revenue we had to invent.  It is NOT the daily
        # grid fill (that is to_daily_grid); it is only these NULL-revenue-but-transactions
        # rows.  It goes into the encoder so the model can learn to discount them.
        is_missing_tpv = tpv.isna() & (success_count.fillna(0) > 0)
        df["target_imputed"] = is_missing_tpv.astype(np.float32)
        if int(is_missing_tpv.sum()):
            merchant_median_tpv = (
                df.loc[tpv > 0]
                .groupby(cfg.group_col, observed=True)[cfg.raw_target]
                .median()
            )
            imputed = (df.loc[is_missing_tpv, cfg.group_col]
                       .map(merchant_median_tpv).astype("float64").fillna(0.0))
            df.loc[is_missing_tpv, cfg.raw_target] = imputed.to_numpy()
            print(f"[data] imputed {int(is_missing_tpv.sum())} NULL-revenue "
                  f"merchant-days that DID have successful transactions "
                  f"(per-merchant median); flagged via 'target_imputed'")
    else:  # pragma: no cover
        df["target_imputed"] = np.float32(0.0)

    for col in STATIC_ATTR_COLS:
        if col not in df.columns:
            continue
        if col == "merchant_created_datetime":
            # tz-aware UTC in the extract while request_date is a naive date.  Strip the
            # zone *and* the time-of-day: keeping 14:05 on the creation timestamp makes
            # a same-day transaction floor to -1 days (14 rows in the extract do this).
            s = pd.to_datetime(df[col], utc=True, errors="coerce")
            df[col] = s.dt.tz_localize(None).dt.normalize()
        else:
            df[col] = df[col].astype("string").fillna("unknown").astype(str)

    return df


# ======================================================================================
# 4. Daily grid + feature engineering
# ======================================================================================
def to_daily_grid(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:

# Reindex every merchant onto a complete daily calendar.

# A day on which a merchant processed nothing is a day of GBP 0 revenue, not a
# missing observation.  The original script instead passed
# ``allow_missing_timesteps=True``, which does not fill anything - it merely
# lets adjacent encoder positions be far apart in real time.  With only 71%
# grid completeness (median largest gap 12 days, p95 138 days) a nominal
# "150-day encoder" could span well over a year of wall-clock, which destroys
# the weekly and annual structure the model is supposed to learn.
    
    data = df.sort_values([cfg.group_col, cfg.date_col]).reset_index(drop=True)

    # One (first_date, last_date) pair per merchant.  The grid runs between them rather
    # than over the global 2024-01-01..2026-07-27 window, because inventing rows before a
    # merchant existed would tell the model it earned GBP 0 for months it was not trading.
    active_span = data.groupby(cfg.group_col, observed=True)[cfg.date_col].agg(["min", "max"])
    per_merchant_grids = []
    for merchant, (first_date, last_date) in active_span.iterrows():
        per_merchant_grids.append(
            pd.DataFrame({
                cfg.group_col: merchant,
                cfg.date_col: pd.date_range(first_date, last_date, freq="D"),
            })
        )
    full_grid = pd.concat(per_merchant_grids, ignore_index=True)

    # Left-join the observed rows onto the complete grid: every calendar day survives, and
    # days the extract never had become all-NaN rows, which the fills below turn into
    # explicit zeros.  The printed counts are observed rows -> grid rows.
    rows_before_fill = len(data)
    data = full_grid.merge(data, on=[cfg.group_col, cfg.date_col], how="left")
    print(f"[grid] {rows_before_fill:,} observed rows -> {len(data):,} grid rows "
          f"(+{len(data) / rows_before_fill - 1:.1%}); the added rows are days with no "
          f"transactions, now GBP 0")

    # dict.fromkeys () de-duplicates while preserving order (the response and flow groups
    # share daily_successful_transaction_count), and the `if col in data.columns` guard
    # skips groups this extract does not carry, e.g. INTEGRATION_COLS today.
    zero_fill_cols = [
        col for col in dict.fromkeys(
            ZERO_ON_INACTIVE_DAY_COLS + ACQUIRER_COLS + RESPONSE_COLS + PAYMENT_COLS
            + INTEGRATION_COLS
        )
        if col in data.columns
    ]
    data[zero_fill_cols] = data[zero_fill_cols].fillna(0.0)
    if "target_imputed" in data.columns:
        data["target_imputed"] = data["target_imputed"].fillna(0.0).astype(np.float32)

    # Merchant attributes ("attr") are properties of the merchant, not per-day flows, so
    # they are carried across the inserted rows instead of zeroed.  ffill propagates the
    # last known value forward; the following bfill covers a merchant whose very first grid
    # row was inserted, which ffill alone would leave as NaN.
    attr_cols = [c for c in STATIC_ATTR_COLS if c in data.columns]
    data[attr_cols] = data.groupby(cfg.group_col, observed=True)[attr_cols].ffill()
    data[attr_cols] = data.groupby(cfg.group_col, observed=True)[attr_cols].bfill()

    if cfg.trim_leading_zeros:
        has_tpv = data[cfg.raw_target].to_numpy() > 0
        data["_has_traded_yet"] = has_tpv
        traded_before = data.groupby(cfg.group_col, observed=True)["_has_traded_yet"].transform("cummax")
        data = data.loc[traded_before].drop(columns="_has_traded_yet").reset_index(drop=True)
        print(f"[grid] {len(data):,} rows after trimming each merchant's "
              f"pre-first-sale zeros (cfg.trim_leading_zeros=True)")
    else:
        print(f"[grid] {len(data):,} rows; pre-first-sale zeros KEPT "
              f"(cfg.trim_leading_zeros=False), so ramp-up from zero is learnable")
    return data


def safe_divide(numerator: pd.Series | np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that returns 0 instead of NaN/inf when the denominator is 0.

    Needed because after the daily grid fill a merchant can have a day with zero
    transactions, so every mix-ratio denominator is legitimately 0 on that day.  Plain
    division would emit NaN, and a single NaN anywhere in a continuous column makes
    TimeSeriesDataSet raise.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    data = df.copy()

    # --- target ---------------------------------------------------------------------
    tpv = pd.to_numeric(data[cfg.raw_target], errors="coerce").fillna(0.0).clip(lower=0.0)
    data[cfg.raw_target] = tpv.astype(np.float64)
    data[cfg.target] = np.log1p(data[cfg.raw_target].to_numpy()).astype(np.float32)

    # --- time index -----------------------------------------------------------------
    # time_idx counts days from the GLOBAL earliest date in the frame (2024-01-01 -> 0), not
    # from each merchant's own first day, so a given time_idx is the same calendar day for
    # every merchant.  That is what lets train/val/holdout be cut at shared indices.
    data["time_idx"] = ((data[cfg.date_col] - data[cfg.date_col].min()).dt.days).astype(np.int64)
    min_date = data[cfg.date_col].min()
    # ...and series_start_idx records where each merchant's own series begins on that shared
    # axis (merchant 171 -> 871).  It is a cohort marker: nothing is renumbered to 0, and an
    # encoder window starting at day 778 keeps time_idx 778.
    data["series_start_idx"] = (
        data.groupby(cfg.group_col, observed=True)["time_idx"].transform("min")
    )

    # --- calendar: the warehouse's own columns, extrapolated only past its last date --
    data = add_calendar_features(data, cfg.date_col)

    # --- merchant age ---------------------------------------------------------------
    # log1p here is not about skew (age is roughly uniform, 0..3843 days).  It is about
    # what the feature is FOR: the difference between day 10 and day 100 of a merchant's
    # life is a real behavioural change - onboarding, ramp-up - while the difference
    # between day 3000 and day 3090 is nothing at all.  On a linear scale those two gaps
    # are identical to the network; on a log scale the first is 2.3 units and the second
    # 0.03.  Both raw and log forms are kept: since_merchant_creation feeds the
    # per-merchant static, log_merchant_age is the known real the decoder reads.
    if "merchant_created_datetime" in data.columns:
        age_days = (data[cfg.date_col] - data["merchant_created_datetime"]).dt.days
        age_days = age_days.fillna(0).clip(lower=0)
    else:  # pragma: no cover
        age_days = pd.Series(0, index=data.index)
    data["since_merchant_creation"] = age_days.astype(np.float32)
    data["log_merchant_age"] = np.log1p(age_days.to_numpy()).astype(np.float32)
    age_at_first_sale = (
        data.groupby(cfg.group_col, observed=True)["since_merchant_creation"].transform("min")
    )
    data["log_tenure_at_series_start"] = np.log1p(
        age_at_first_sale.to_numpy()
    ).astype(np.float32)

    # --- heavy-tailed observed reals -> log1p ---------------------------------------
    # These six columns are scaled by ONE GLOBAL sklearn StandardScaler, because
    # TimeSeriesDataSet does `self._scalers[name] = StandardScaler().fit(data[[name]])`
    # per column with no notion of groups (_timeseries.py:1256-1262) and build_datasets
    # never overrides it.  Measured on the prepared frame, that is ruinous on the raw
    # scale: 94-98% of all rows land inside the BOTTOM 1% of each column's z range,
    # interquartile range over full range is 3.4e-4 to 1.2e-3, and the 4.67% of rows
    # belonging to 14 whale merchants carry 77-97% of the total sum of squares.  For a
    # merchant averaging GBP 482/day, a GBP 20 day and a GBP 900 day - a 45x move - differ
    # by z = 0.0065, which is 0.0058% of the input range and 1/8300th of the gap between
    # two ordinary days of merchant 171.
    #
    # After log1p the same 45x move is z = 1.02, statistically the same size (0.97x) as a
    # whale's 38x move; every size band gets 0.55-0.80 sd of internal resolution instead of
    # 0.0014/0.0063/0.045 for the three smallest; the whale share of sum-of-squares falls to
    # 7-17%; and Spearman(scaled, raw) = 1.000000, so no ordering information is lost.
    # `--selftest` re-measures this and asserts the key thresholds, so the claim is checked
    # rather than trusted.  Caveat: fraud and dispute counts are 91.6%/90.3% zeros, so they
    # stay near-indicators either way (z(max) still +10.7/+10.3); log1p is doing the real
    # work on the dense columns.  note this argument does NOT cover the target - that is
    # scaled per merchant by FlooredGroupNormalizer, and its log1p is justified separately
    # by the loss geometry (MODEL_NOTES.md 2.1).
    txn_count = data["daily_transaction_amount"].to_numpy()
    success_count = data["daily_successful_transaction_count"].to_numpy()
    # log_txn_count dropped as a model input: it scored 0.60% encoder importance in v1 and
    # is near-collinear with log_success_count.  The column is still read here because
    # had_no_attempts needs it, and that flag is the part that carries signal.
    data["log_success_count"] = np.log1p(success_count).astype(np.float32)
    data["log_unsuccessful_gbp"] = np.log1p(
        data["daily_unsuccessful_amount_gbp"].to_numpy()
    ).astype(np.float32)
    data["log_fraud_count"] = np.log1p(data["daily_fraud_amount"].to_numpy()).astype(np.float32)
    data["log_dispute_count"] = np.log1p(
        data["daily_dispute_amount"].to_numpy()
    ).astype(np.float32)
    # Revenue = transactions x average ticket, and the two move for different reasons: a
    # merchant can hold volume while its basket shrinks.  Giving the encoder the
    # decomposition saves it from having to infer the ratio.
    data["log_avg_ticket"] = np.log1p(
        safe_divide(data[cfg.raw_target].to_numpy(), np.maximum(success_count, 0))
    ).astype(np.float32)
    # is_zero_day: earned nothing.  had_no_attempts: nobody even tried.  Different events -
    # the first can mean every transaction was declined, the second means the merchant was
    # dormant - and with 20% of grid days at zero revenue the distinction matters.
    data["is_zero_day"] = (data[cfg.raw_target].to_numpy() <= 0).astype(np.float32)
    data["had_no_attempts"] = (txn_count <= 0).astype(np.float32)

    # --- mix ratios (bounded in [0, 1], so safe for a global standard scaler) --------
    # Every "*_amount" column is a COUNT, and within each group they sum exactly to
    # daily_transaction_amount, so a share-of-the-day ratio is scale-free across merchants.
    ratio_names: list[str] = []
    for source_cols, group_name in ((ACQUIRER_COLS, "acquirer"),
                                    (PAYMENT_COLS, "payment type"),
                                    (RESPONSE_COLS, "response status"),
                                    (INTEGRATION_COLS, "integration")):
        present = [c for c in source_cols if c in data.columns]
        if not present:
            print(f"[features] no {group_name} columns in this extract - "
                  f"skipping that ratio group (expected {source_cols})")
            continue
        daily_total = data[present].to_numpy(dtype=np.float64).sum(axis=1)
        for col in present:
            ratio_col = f"{col}_ratio"
            data[ratio_col] = safe_divide(data[col].to_numpy(), daily_total).astype(np.float32)
            ratio_names.append(ratio_col)
    data.attrs["ratio_names"] = ratio_names

    # --- categoricals as strings ----------------------------------------------------
    # NaNLabelEncoder maps by value, so plain strings are all it needs; a pandas
    # `category` dtype would additionally carry per-frame category codes that differ
    # between the training frame and a single-merchant forecast frame.
    categorical_cols = [
        "day_of_week", "day_of_month", "calendar_month", "day_type",
        "is_uk_bank_holiday", "merchant_category", "sector_group",
        "iso_name"
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)

    return data


def filter_series(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
# Drop merchants that cannot support a training window or normalisation.

# Three independent reasons a merchant is dropped:

# 1. Too short --> A training window needs min_encoder_length + horizon = 58 grid days.
#     Below that the merchant produces no window at all and only inflates the
#     merchant_id embedding table.
# 2. Near-constant --> The normaliser divides by the group's own standard deviation.
#     GroupNormalizer.fit floors that at ``np.finfo(np.float16).eps`` = 9.77e-4, so a
#     merchant whose log revenue never moves is divided by ~1e-3 and its normalised
#     target lands in the thousands.  Because the pinball (quantile) loss is summed over the batch,
#     one such merchant contributes ~1000x the gradient of a normal one and the optimiser
#     spends the step fitting it.  FlooredGroupNormalizer raises the floor to 0.25, and
#     this filter removes the merchants that would sit on the floor anyway.
# 3. Barely active --> Fewer than min_active_days non-zero days means the series is
#     almost all zeros: the model can score well on it by predicting zero forever, which
#     drags the loss down without teaching anything.

# note: ``log_std`` is the standard deviation OF the log-transformed target, not the log of a
# standard deviation.  It is measured in log space because that is the space the model
# optimises in, so the threshold means "this merchant's revenue varies by less than
# e^0.20 = 22% around its own level".
    
    by_merchant = df.groupby(cfg.group_col, observed=True)
    stats = pd.DataFrame(
        {
            "days": by_merchant.size(),
            "log_std": by_merchant[cfg.target].std(),
            "active_days": by_merchant[cfg.raw_target].apply(lambda s: int((s > 0).sum())),
        }
    )
    keep = (
        (stats["days"] >= cfg.min_series_days)
        & (stats["log_std"].fillna(0.0) >= cfg.min_log_std)
        & (stats["active_days"] >= cfg.min_active_days)
    )
    # ~keep inverts the boolean mask elementwise, i.e. "the merchants we are NOT keeping".
    dropped = stats.loc[~keep]
    print(
        f"[filter] keeping {int(keep.sum())}/{len(stats)} merchants "
        f"(dropped {int((stats['days'] < cfg.min_series_days).sum())} too short, "
        f"{int((stats['log_std'].fillna(0) < cfg.min_log_std).sum())} near-constant, "
        f"{int((stats['active_days'] < cfg.min_active_days).sum())} barely active)"
    )
    if len(dropped):
        print(f"[filter] dropped ids (first 20): {list(dropped.index[:20])}")
    out = df[df[cfg.group_col].isin(keep[keep].index)].reset_index(drop=True)
    print(f"[filter] {len(out):,} rows retained")
    return out


def prune_uninformative(df: pd.DataFrame, cols: list[str], cutoff: int, cfg: Config,
                        tol: float = 1e-6) -> list[str]:
    """Drop reals that are (near) constant on the training block.

    Constant inputs still consume a slot in the variable-selection softmax, which is part
    of why the original model's encoder importance chart was a long tail of
    indistinguishable near-zero bars (top variable 6.87%, top three 17.9%).

    ``tol`` is an absolute threshold on the standard deviation, in each column's own
    units.  Every column reaching here is either a [0,1] ratio or a log1p value, so 1e-6
    means "this column is the same number in every training row" rather than any
    scale-dependent judgement.  Measured on the training block only, so a feature that
    only starts varying inside the validation window is correctly treated as unusable.
    """
    train = df.loc[df["time_idx"] <= cutoff, cols]
    # numeric_only=True makes std() skip any non-numeric column instead of raising; it is
    # defensive only, since everything in `cols` should already be a float.
    std = train.std(numeric_only=True)
    keep = [c for c in cols if float(std.get(c, 0.0)) > tol]
    dropped = sorted(set(cols) - set(keep))
    if dropped:
        print(f"[prune] dropping {len(dropped)} constant/near-constant reals: {dropped}")
    return keep


# ======================================================================================
# 5. Normaliser with a scale floor
# ======================================================================================
class FrozenGroupIdEncoder(NaNLabelEncoder):
    """A NaNLabelEncoder that stops learning new classes after its first fit.

    Used for the ``__group_id__<group_col>`` column - the series identifier that ends up in
    ``data["groups"]`` and is the key ``GroupNormalizer.get_parameters`` looks the target
    centre and scale up with.  Two things have to be true of that encoding and neither is
    true of the default:

    1. It must agree with the encoding of ``merchant_id`` itself, because that is the column
       the normaliser was fitted against and therefore what ``norm_`` is indexed by.  A
       plain ``NaNLabelEncoder()`` defaults to add_nan=False and numbers from 0, while the
       merchant_id encoder used here reserves 0 for the unknown token and numbers from 1 -
       an off-by-one that silently hands every series its neighbour's scale.

    2. It must not gain classes when the dataset is rebuilt on a wider frame.
       ``TimeSeriesDataSet`` re-fits this one encoder on every construction with
       ``overwrite=False``, and ``NaNLabelEncoder.fit`` computes its offset as
       ``len(self.classes_)`` - which already counts the nan slot - so the first added class
       lands one code past the end.  ``classes_vector_`` is then shorter than the largest
       code it contains, which both corrupts ``inverse_transform`` and pushes codes past the
       end of the merchant embedding table sized from the training dataset.

    Freezing solves both: validation, holdout and forecast frames reuse the training codes
    verbatim, and a merchant absent from the training block transforms to 0 - the unknown
    token the merchant embedding is trained to handle via ``group_id_dropout``, and a code
    that is deliberately missing from ``norm_`` so the median fallback applies.
    """

    def fit(self, y, overwrite: bool = False):
        if hasattr(self, "classes_") and not overwrite:
            return self
        return super().fit(y, overwrite=True)


class FlooredGroupNormalizer(GroupNormalizer):
    """GroupNormalizer whose per-group scale cannot collapse.

    What the parent does: for each merchant it stores ``center`` = mean of that merchant's
    log revenue and ``scale`` = its standard deviation, then feeds the network
    ``(y - center) / scale`` and inverts that on the way out.  The problem is the floor it
    puts under ``scale``: ``np.finfo(np.float16).eps``, which is 9.77e-4.  A merchant
    billing the same amount every day has a real standard deviation of ~0, so it is divided
    by ~1e-3 and its normalised target becomes ~1000x too large.  The pinball loss is
    summed over the batch, so that merchant alone then dictates the gradient of every step
    it appears in - and after ``gradient_clip_val`` truncates the step, the update is
    almost entirely that one merchant's.  Raising the floor to 0.25 bounds the worst case:
    no group can contribute more than 4x a well-behaved one.

    ``fit`` is overridden rather than the whole class rewritten, so the parent still does
    all the grouping work and this only clamps the result and recomputes the median
    fallback used for merchants absent at fit time.  ``get_parameters`` clamps again on the
    read path, which is what a cold-start merchant hits.
    """

    def __init__(self, *args, scale_floor: float = 0.25, **kwargs):
        # scale_floor is set before super().__init__ because sklearn's BaseEstimator
        # introspects __init__ arguments; *args/**kwargs forward everything else
        # (method, groups, center, ...) to GroupNormalizer unchanged.
        self.scale_floor = scale_floor
        super().__init__(*args, **kwargs)

    def fit(self, y: pd.Series, X: pd.DataFrame):
        super().fit(y, X)
        floor = float(self.scale_floor)
        if isinstance(self.norm_, dict):
            for k, v in self.norm_.items():
                if isinstance(v, pd.DataFrame):
                    v["scale"] = v["scale"].clip(lower=floor)
                    self.norm_[k] = v
            if isinstance(getattr(self, "missing_", None), dict):
                self.missing_ = {
                    g: (s.median().to_dict() if isinstance(s, pd.DataFrame) else s)
                    for g, s in self.norm_.items()
                }
        elif isinstance(self.norm_, pd.DataFrame):
            self.norm_["scale"] = self.norm_["scale"].clip(lower=floor)
            self.missing_ = self.norm_.median().to_dict()
        # sklearn's fit() contract is to return self, so calls can be chained.
        return self

    def get_parameters(self, *args, **kwargs):
        params = np.asarray(super().get_parameters(*args, **kwargs), dtype=np.float64)
        params[..., 1] = np.maximum(params[..., 1], self.scale_floor)
        return params


# ======================================================================================
# 6. The model
# ======================================================================================
class RevenueTFT(TemporalFusionTransformer):
    """TFT with two task-specific additions.

    ``group_id_dropout``
        During training the ``merchant_id`` column of the categorical inputs is
        replaced by the encoder's NaN token with this probability.  164 of 510
        merchants got *zero* training windows under the original configuration
        (merchant 171 among them), so their identity embedding stayed at its
        random initialisation - and a random embedding is exactly how you get a
        confident forecast of ~0 for a merchant that is plainly not at 0.  Making
        the NaN token a well-trained "generic merchant" gives every unseen or
        barely-seen merchant a sane fallback, and forces the model to keep using
        sector / scale / recent-history evidence rather than leaning entirely on
        per-merchant memorisation.

    GBP-space logging
        ``val_loss`` is quantile loss on ``log1p(revenue)``, which is the right thing to
        *optimise* but impossible to sanity-check by eye.  ``step`` also logs MAE and RMSE
        in GBP so the numbers in TensorBoard mean something.

        MAE and RMSE are not the same measurement.  RMSE squares the errors before
        averaging, so it is dominated by the largest ones; MAE is not.  On a target where
        one merchant does GBP 15.2M/day that gap is the diagnostic: MAE falling while RMSE
        stays flat means the typical merchant improved and the whales did not, and RMSE
        falling while MAE stays flat means the opposite.  With only one of them you cannot
        tell those apart - and the previous model's failure was exactly a whale/typical
        split.
    """

    GROUP_COLUMN = "merchant_id"

    def __init__(self, *args, group_id_dropout: float = 0.0, log_gbp_metrics: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_id_dropout = float(group_id_dropout)
        self.log_gbp_metrics = bool(log_gbp_metrics)
        # Persist them so a reloaded checkpoint reports how it was trained.
        try:
            self.hparams["group_id_dropout"] = float(group_id_dropout)
            self.hparams["log_gbp_metrics"] = bool(log_gbp_metrics)
        except Exception:  # pragma: no cover
            pass

    # -- lazily resolve which column of x_cat holds the group id ---------------------
    @property
    def group_cat_index(self) -> int | None:
        """Position of merchant_id inside the categorical input tensor, or None.

        ``-> int | None`` is a type hint: this returns an int when merchant_id is among the
        model's categorical inputs and None when it is not (which build_model turns into a
        hard error).  The answer cannot be computed in __init__ because
        ``hparams.x_categoricals`` is set by from_dataset, so it is resolved on first use
        and cached in ``_group_cat_index``.  The sentinel is the string "unset" rather than
        None, because None is itself a valid cached answer.
        """
        cached = getattr(self, "_group_cat_index", "unset")
        if cached == "unset":
            categorical_inputs = list(getattr(self.hparams, "x_categoricals", []) or [])
            cached = (categorical_inputs.index(self.GROUP_COLUMN)
                      if self.GROUP_COLUMN in categorical_inputs else None)
            self._group_cat_index = cached
        return cached

    def forward(self, x: dict[str, torch.Tensor], **kwargs):
        """One forward pass.  Overridden only to apply merchant-id dropout in training.
        Dropout is when certain nodes randomly switch off during training process in order to reduce risk of overfitting and minimize val loss
        """
        if self.training and self.group_id_dropout > 0.0:
            j = self.group_cat_index
            if j is not None:
                x = dict(x)
                enc, dec = x["encoder_cat"], x["decoder_cat"]
                mask = (
                    torch.rand(enc.size(0), 1, device=enc.device) < self.group_id_dropout
                )
                enc = enc.clone()
                dec = dec.clone()
                # index 0 is NaNLabelEncoder(add_nan=True)'s unknown/NaN token
                enc[..., j] = torch.where(mask, torch.zeros_like(enc[..., j]), enc[..., j])
                dec[..., j] = torch.where(mask, torch.zeros_like(dec[..., j]), dec[..., j])
                x["encoder_cat"], x["decoder_cat"] = enc, dec
        return super().forward(x, **kwargs)

    def step(self, x, y, batch_idx, **kwargs):
        log, out = super().step(x, y, batch_idx, **kwargs)
        if self.log_gbp_metrics and not self.predicting:
            with torch.no_grad():
                try:
                    pred_log = self.to_prediction(out)
                    target_log = y[0] if isinstance(y, tuple | list) else y
                    if isinstance(target_log, torch.nn.utils.rnn.PackedSequence):
                        target_log, _ = torch.nn.utils.rnn.pad_packed_sequence(
                            target_log, batch_first=True
                        )
                    predicted = torch.expm1(pred_log.float()).clamp_min(0.0)
                    actual = torch.expm1(target_log.float()).clamp_min(0.0)
                    n = min(predicted.size(-1), actual.size(-1))
                    predicted, actual = predicted[..., :n], actual[..., :n]
                    error = (predicted - actual).abs()

                    # MAE_log is the metric to watch epoch to epoch.  It is in the space the
                    # model optimises, so every merchant contributes comparably.
                    self.log(f"{self.current_stage}_MAE_log",
                             (pred_log - target_log).abs().mean(),
                             on_step=False, on_epoch=True, batch_size=actual.size(0))
                    # MedAE_gbp is the robust money-space companion: what the typical
                    # forecast day is worth being wrong by.
                    self.log(f"{self.current_stage}_MedAE_gbp", error.flatten().median(),
                             on_step=False, on_epoch=True, batch_size=actual.size(0))
                    # MAE_gbp and RMSE_gbp are kept because they answer the "is it right in
                    # money" question, but read them with care and NOT as an epoch-to-epoch
                    # signal.  expm1 turns a log-space error into a MULTIPLICATIVE one, so a
                    # single overshoot on a merchant doing GBP 15.2M/day dominates the mean
                    # over ~175k validation points - which is why MAE_gbp can rise while
                    # MAE_log falls.  That is the same arithmetic that made a GBP-space LOSS
                    # untrainable (MODEL_NOTES.md Part 1, D2); it applies just as much to a
                    # GBP-space metric.  Judge progress on MAE_log and MedAE_gbp, and use
                    # MAE_gbp/RMSE_gbp only on the final holdout report, where the per-size-
                    # band table shows where the error actually sits.
                    self.log(f"{self.current_stage}_MAE_gbp", error.mean(),
                             on_step=False, on_epoch=True, batch_size=actual.size(0))
                    self.log(f"{self.current_stage}_RMSE_gbp",
                             torch.sqrt(((predicted - actual) ** 2).mean()),
                             on_step=False, on_epoch=True, batch_size=actual.size(0))
                except Exception:  # pragma: no cover - never let logging break a run
                    pass
        return log, out


# ======================================================================================
# 7. Dataset construction
# ======================================================================================
def stride_windows(ds: TimeSeriesDataSet, stride: int) -> TimeSeriesDataSet:
# Thin out overlapping forecast origins, in place.

    if stride <= 1 or len(ds.index) == 0:
        return ds
    idx = ds.index
    if "sequence_id" in idx.columns and not idx.columns.duplicated().any():
        keep = idx.groupby("sequence_id", sort=False).cumcount() % stride == 0
    else:  # pragma: no cover - duplicated columns can appear with predict=True
        keep = np.arange(len(idx)) % stride == 0
    ds.index = idx.loc[keep].reset_index(drop=True)
    return ds


@dataclass
class Split:
    training: TimeSeriesDataSet
    validation: TimeSeriesDataSet
    holdout: TimeSeriesDataSet
    train_cutoff: int
    val_cutoff: int
    max_time_idx: int
    feature_spec: dict = field(default_factory=dict)


def build_datasets(df: pd.DataFrame, cfg: Config, full: bool = False) -> Split:

    max_t = int(df["time_idx"].max())
    val_cutoff = max_t - cfg.horizon
    sel_cutoff = val_cutoff - cfg.val_days
    # ``full=True`` trains on every day for the production forecast; the validation and
    # holdout blocks keep the *same* definitions so the curves stay comparable, but they
    # are then in-sample and must not be used for model selection.
    train_cutoff = max_t if full else sel_cutoff

    known_reals = (
        [f"year_{f}{k}" for k in (1, 2, 3) for f in ("sin", "cos")]
        + [f"month_{f}{k}" for k in (1, 2) for f in ("sin", "cos")]
        + ["is_month_start", "is_month_end", "is_payday_window", "is_quarter_end",
           "days_to_month_end", "holiday_offset", "log_merchant_age"]
    )
    # NOTE: raw ``time_idx`` is deliberately *not* a known real.  The original model
    # had it, which lets the network fit a level against absolute time and then
    # extrapolate off the end of the training range - every forecast day sits beyond
    # anything the scaler ever saw.  ``add_relative_time_idx`` gives position within
    # the window, which is what the decoder actually needs.

    # The target MUST be listed here explicitly.  TimeSeriesDataSet does not add it
    # for you: ``reals`` is exactly static_reals + known_reals + unknown_reals, and
    # TemporalFusionTransformer.forward reads only ``encoder_cont``/``encoder_cat``
    # (``encoder_target`` is used only by MASE-style losses).  Omitting it - as both
    # the original script and the notebook did - builds a TFT whose encoder never
    # sees a single past revenue value.  That is why the published attention plot is
    # a smooth recency ramp with no bump at lag 7/14/21/28: there was no periodic
    # signal in the encoder to attend to.  pytorch_forecasting keeps it out of
    # ``decoder_variables``, so there is no leakage into the horizon.
    unknown_reals = [
        cfg.target,
        "log_success_count", "log_unsuccessful_gbp",
        "log_fraud_count", "log_dispute_count", "log_avg_ticket",
        "is_zero_day", "had_no_attempts", "target_imputed",
    ]
    unknown_reals += list(df.attrs.get("ratio_names", []))
    unknown_reals = [c for c in dict.fromkeys(unknown_reals) if c in df.columns]
    unknown_reals = prune_uninformative(df, unknown_reals, train_cutoff, cfg)
    # A bare `assert` raises AssertionError with this message if the condition is false.
    # It is here because this exact omission is what broke the previous model, silently:
    # without the target in unknown_reals the script still trains, still reports a falling
    # loss, and still plots the observed history (plot_prediction reads encoder_target,
    # which the network never sees) - so nothing looks wrong.
    assert cfg.target in unknown_reals, (
        f"{cfg.target!r} must stay in time_varying_unknown_reals or the encoder cannot "
        "see revenue history"
    )

    # merchant_id must be listed explicitly.  TimeSeriesDataSet does *not* fold
    # group_ids into static_categoricals (``categoricals`` is exactly
    # static + known + unknown categoricals), so the original model - which also
    # omitted it - had no merchant-identity embedding whatsoever: with a global
    # target normaliser on top, nothing in the network could tell one merchant from
    # another except its sector labels and its own encoder history.
    static_cats = [cfg.group_col, "merchant_category", "sector_group",
                   "iso_name"]
    static_cats = [c for c in static_cats if c in df.columns]
    static_reals = ["series_start_idx", "log_tenure_at_series_start"]

    # day_type is deliberately absent: it is a deterministic function of day_of_week and
    # is_uk_bank_holiday (verified on all 939 dates - 'Holiday' iff is_uk_bank_holiday,
    # 'Weekend' iff dayofweek >= 5), so including all three feeds the variable-selection
    # softmax a perfectly collinear triple and splits its attention across duplicates.
    # day_of_week is kept over day_type because day_type collapses Saturday and Sunday,
    # which behave differently.  acquiring_bank_name is absent because the per-day acquirer
    # mix ratios already carry it - see MODEL_NOTES.md for what that costs.
    known_cats = ["day_of_week", "day_of_month", "calendar_month",
                  "is_uk_bank_holiday"]
    known_cats = [c for c in known_cats if c in df.columns]

    all_cats = list(dict.fromkeys([cfg.group_col] + static_cats + known_cats))
    categorical_encoders = {c: NaNLabelEncoder(add_nan=True) for c in all_cats}

    # ------------------------------------------------------------------------------------
    # THE OFF-BY-ONE THAT MADE EVERY ORIGINAL FORECAST WRONG.  TimeSeriesDataSet keeps TWO separate
    # encodings of the group column:
    #
    #   1. ``merchant_id`` itself - a model input, encoded with the encoder supplied above.
    #      Because the target normaliser is a GroupNormalizer, _preprocess_data() encodes
    #      this column BEFORE fitting the normaliser, so ``norm_`` ends up indexed by these
    #      codes.  add_nan=True reserves 0 for the unknown token, so real merchants are
    #      1..N.
    #   2. ``__group_id__merchant_id`` - the series identifier that lands in
    #      ``data["groups"]``.  _preprocess_data() builds it from
    #      ``categorical_encoders.get("__group_id__merchant_id", NaNLabelEncoder())`` - a
    #      DEFAULT NaNLabelEncoder, i.e. add_nan=False, i.e. 0..N-1.
    #
    # __getitem__ then does ``target_scale = target_normalizer.get_parameters(groups)``,
    # looking encoding 2 up in a table keyed by encoding 1.  Every series was therefore
    # centred and scaled with the PREVIOUS merchant's parameters, and the merchant at code 0
    # fell through the KeyError branch onto the global median.  Because the target is
    # log1p(revenue), a wrong centre is a multiplicative error once expm1 undoes it: lending
    # merchant 991 its neighbour's centre (8.96 instead of 6.72) multiplies its forecast by
    # e^2.24 = 9.4x, and a wrong scale stretches the quantile spread by the same mechanism.
    # That is where the GBP 30m p98 came from - not from log1p amplifying an honest error.
    #
    # The fix is to register the group-id encoder explicitly and to pre-fit BOTH encodings on
    # the same id list, so they are the same function by construction.  Two details matter:
    #
    #   * The id list is every merchant in ``df``, not just the training block.  A series
    #     identifier has to be unique per series - if the merchants missing from the training
    #     block all collapsed onto the unknown token, TimeSeriesDataSet would treat them as
    #     ONE series and reject the frame for having non-consecutive time steps.
    #   * FrozenGroupIdEncoder rather than NaNLabelEncoder, because TimeSeriesDataSet re-fits
    #     this particular encoder on every construction - see its docstring.
    #
    # ``norm_`` is still fitted on the training block alone, so it simply has no row for a
    # merchant that never appears there, and the median fallback applies to it.  That is the
    # intended cold-start path.  assert_target_scales_aligned() below fails the run if these
    # two encodings ever drift apart again.
    every_merchant = pd.Series(sorted(df[cfg.group_col].unique()))
    categorical_encoders[cfg.group_col] = NaNLabelEncoder(add_nan=True).fit(every_merchant)
    categorical_encoders[f"__group_id__{cfg.group_col}"] = (
        FrozenGroupIdEncoder(add_nan=True).fit(every_merchant)
    )

    train_df = df[df["time_idx"] <= train_cutoff].reset_index(drop=True)
    if full:
        # Printed differently because "(939, 909]" is an empty interval and reads as a bug.
        # In full mode training covers everything, so the val and holdout blocks keep their
        # definitions only so the loss curves stay comparable - they are now in-sample.
        print(
            f"[split] FULL REFIT: max_time_idx={max_t}  train<= {train_cutoff} (everything)  "
            f"val ({sel_cutoff}, {val_cutoff}] and holdout ({val_cutoff}, {max_t}] are now "
            f"INSIDE the training data and must not be read as scores"
        )
    else:
        print(
            f"[split] max_time_idx={max_t}  train<= {train_cutoff}  "
            f"val decoder in ({train_cutoff}, {val_cutoff}]  holdout in ({val_cutoff}, {max_t}]"
        )
    print(f"[split] train rows={len(train_df):,} merchants={train_df[cfg.group_col].nunique()}")

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=cfg.target,
        group_ids=[cfg.group_col],
        categorical_encoders=categorical_encoders,
        min_encoder_length=cfg.min_encoder_length,
        max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=cfg.horizon,
        max_prediction_length=cfg.horizon,
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_categoricals=known_cats,
        time_varying_known_reals=known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=FlooredGroupNormalizer(
            method="standard", groups=[cfg.group_col], center=True, scale_floor=0.25
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
        randomize_length=None,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= val_cutoff].reset_index(drop=True),
        min_prediction_idx=sel_cutoff + 1,
        stop_randomization=True,
        predict=False,
    )
    n_val_full = len(validation)
    stride_windows(validation, cfg.val_stride)
    holdout = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=val_cutoff + 1,
        stop_randomization=True,
        predict=True,
    )
    print(
        f"[split] windows: train={len(training):,} "
        f"val={len(validation):,} (of {n_val_full:,}, stride {cfg.val_stride}) "
        f"holdout={len(holdout):,}"
    )

    for name, ds in (("train", training), ("val", validation), ("holdout", holdout)):
        assert_target_scales_aligned(ds, df, cfg, train_cutoff, name)

    spec = dict(
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_categoricals=known_cats,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
    )
    # Split is a plain container (a dataclass): the three TimeSeriesDataSets plus the
    # cutoff indices and the feature lists, so callers do not have to recompute them.
    # No percentages - `spec` is written to config.json as a record of what went in.
    return Split(training, validation, holdout, train_cutoff, val_cutoff, max_t, spec)


def assert_target_scales_aligned(dataset: TimeSeriesDataSet, df: pd.DataFrame, cfg: Config,
                                 train_cutoff: int, label: str, n_check: int = 24) -> None:
    """Fail loudly if a window's target_scale is not that window's own merchant's.

    The bug this guards against is invisible in every metric: a series normalised with its
    neighbour's centre still trains, still shows a falling loss, and still plots a sensible
    log-space curve.  It only surfaces after expm1, as a forecast that is a constant
    multiple out.  So the check has to be explicit and it has to compare against numbers
    recomputed from the dataframe rather than from anything the dataset derived.

    ``groups`` is decoded back to a merchant id through the dataset's own group-id encoder,
    then that merchant's centre and scale are recomputed from the training block and matched
    against the ``target_scale`` the dataset actually hands the model.  Cold-start merchants
    (absent from the training block) legitimately carry the median fallback, so they are
    counted and reported rather than treated as failures.
    """
    if len(dataset) == 0:
        return
    normaliser = dataset.target_normalizer
    per_merchant = (df[df["time_idx"] <= train_cutoff]
                    .groupby(cfg.group_col, observed=True)[cfg.target]
                    .agg(["mean", "std"]))

    step = max(1, len(dataset) // n_check)
    mismatched, cold, checked = [], 0, 0
    for w in range(0, len(dataset), step):
        x, _ = dataset[w]
        code = int(x["groups"][dataset.group_ids.index(cfg.group_col)])
        merchant = dataset.transform_values(cfg.group_col, pd.Series([code]),
                                           inverse=True, group_id=True)[0]
        centre, scale = float(x["target_scale"][0]), float(x["target_scale"][1])
        if code not in normaliser.norm_.index or str(merchant) not in per_merchant.index:
            cold += 1
            continue
        checked += 1
        want_centre = float(per_merchant.loc[str(merchant), "mean"])
        want_scale = max(float(per_merchant.loc[str(merchant), "std"]),
                         normaliser.scale_floor)
        if abs(centre - want_centre) > 1e-2 or abs(scale - want_scale) > 1e-2:
            mismatched.append((merchant, code, centre, want_centre, scale, want_scale))

    if mismatched:
        lines = "\n".join(
            f"    merchant {m} (code {c}): dataset gave centre {gc:.4f} scale {gs:.4f}, "
            f"but its own training block is centre {wc:.4f} scale {ws:.4f}"
            for m, c, gc, wc, gs, ws in mismatched[:5]
        )
        raise SystemExit(
            f"[{label}] target_scale is not aligned with the target normaliser for "
            f"{len(mismatched)}/{checked} sampled windows:\n{lines}\n"
            f"  Every forecast would be a multiplicative factor out after expm1.  This is "
            f"the __group_id__ encoding drifting away from the codes GroupNormalizer was "
            f"fitted on - see the comment in build_datasets()."
        )
    print(f"[check] {label}: target_scale matches each window's own merchant for all "
          f"{checked} sampled windows"
          + (f" ({cold} cold-start window(s) on the median fallback)" if cold else ""))


# ======================================================================================
# 8. Metrics in GBP
# ======================================================================================
def expected_from_quantiles(q_values: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """E[Y] from a predicted quantile function, by integrating F^-1 over (0, 1).

    The median of ``log1p(revenue)`` maps to the median of revenue exactly, but the
    *mean* does not - and if you want to add merchant forecasts up into a company
    total you need the mean.  Trapezoidal integration of the predicted quantile
    function gives it without assuming lognormality.
    """
    levels = np.asarray(quantiles, dtype=np.float64)          # e.g. 0.02 .. 0.98
    predicted = np.asarray(q_values, dtype=np.float64)         # revenue at those levels
    # Extend to the full (0, 1) interval by holding the outermost predictions flat, which
    # makes the tails conservative rather than extrapolated.
    levels_padded = np.concatenate([[0.0], levels, [1.0]])
    values_padded = np.concatenate(
        [predicted[..., :1], predicted, predicted[..., -1:]], axis=-1
    )
    segment_width = np.diff(levels_padded)                     # probability mass per slice
    segment_mean = 0.5 * (values_padded[..., 1:] + values_padded[..., :-1])
    return (segment_mean * segment_width).sum(axis=-1)


def gbp_metrics(actual_gbp: np.ndarray, q_gbp: np.ndarray, quantiles: list[float]) -> dict:
    actual = np.asarray(actual_gbp, dtype=np.float64).ravel()
    predicted = np.asarray(q_gbp, dtype=np.float64).reshape(len(actual), -1)
    median_pred = (predicted[:, quantiles.index(0.5)] if 0.5 in quantiles
                   else predicted[:, predicted.shape[1] // 2])
    error = median_pred - actual
    total_actual = np.abs(actual).sum()

    # Pinball (= quantile = check) loss, the thing QuantileLoss minimises, recomputed here
    # in GBP purely to report it.  For quantile level tau it charges tau * error when the
    # prediction is too LOW and (1 - tau) * error when it is too HIGH, so a q90 prediction
    # is penalised 9x more for undershooting than overshooting - which is what makes the
    # model place q90 at the 90th percentile instead of the middle.  The model itself
    # optimises this on log1p(revenue); this copy is on GBP, so the two are not comparable
    # and only the GBP one is divided through below to make it scale-free.
    pinball_total = 0.0
    for i, tau in enumerate(quantiles):
        shortfall = actual - predicted[:, i]
        pinball_total += np.maximum(tau * shortfall, (tau - 1.0) * shortfall).sum()

    smape_terms = np.abs(error) / np.maximum(
        (np.abs(actual) + np.abs(median_pred)) / 2.0, 1e-9
    )
    # Coverage: how often the actual really landed inside a predicted band.  The outer band
    # is the first and last quantile supplied; the p10-p90 one is only defined if those two
    # levels were actually requested, hence the NaN default rather than a wrong number.
    # Index by the smallest and largest LEVEL rather than by position, so the outer band is
    # still the outer band if cfg.quantiles is ever passed unsorted.
    lowest = int(np.argmin(quantiles))
    highest = int(np.argmax(quantiles))
    coverage_outer = (
        (actual >= predicted[:, lowest]) & (actual <= predicted[:, highest])
    ).mean()
    coverage_p10_p90 = np.nan
    if 0.1 in quantiles and 0.9 in quantiles:
        coverage_p10_p90 = (
            (actual >= predicted[:, quantiles.index(0.1)])
            & (actual <= predicted[:, quantiles.index(0.9)])
        ).mean()
    return dict(
        n=int(len(actual)),
        MAE_gbp=float(np.abs(error).mean()),
        RMSE_gbp=float(np.sqrt((error ** 2).mean())),
        MedAE_gbp=float(np.median(np.abs(error))),
        MAE_log=float(np.abs(np.log1p(np.maximum(median_pred, 0)) - np.log1p(actual)).mean()),
        wQL=float(pinball_total / total_actual) if total_actual > 0 else float("nan"),
        sMAPE=float(smape_terms.mean()),
        bias_gbp=float(error.mean()),
        total_actual_gbp=float(actual.sum()),
        total_pred_median_gbp=float(median_pred.sum()),
        coverage_p10_p90=float(coverage_p10_p90),
        coverage_outer=float(coverage_outer),
    )


def predictions_to_frame(pred, dataset: TimeSeriesDataSet, cfg: Config) -> pd.DataFrame:
    """Flatten a quantile Prediction into a tidy (merchant, time_idx, actual, q...) frame."""
    q_log = pred.output.detach().cpu().numpy()  # (n_series, horizon, n_quantiles)
    index = pred.index.reset_index(drop=True)
    dec_t = pred.x["decoder_time_idx"].detach().cpu().numpy()
    y = pred.y[0] if isinstance(pred.y, tuple | list) else pred.y
    y_log = y.detach().cpu().numpy() if y is not None else np.full(dec_t.shape, np.nan)

    n_series, horizon, n_q = q_log.shape
    rows = {
        cfg.group_col: np.repeat(index[cfg.group_col].to_numpy(), horizon),
        "origin_time_idx": np.repeat(index["time_idx"].to_numpy(), horizon),
        "time_idx": dec_t.ravel(),
        "step": np.tile(np.arange(1, horizon + 1), n_series),
        "actual_gbp": np.expm1(y_log.astype(np.float64)).clip(min=0).ravel(),
    }
    q_gbp = np.expm1(q_log.astype(np.float64)).clip(min=0)
    for i, tau in enumerate(cfg.quantiles):
        rows[f"q{int(round(tau * 100)):02d}"] = q_gbp[:, :, i].ravel()
    out = pd.DataFrame(rows)
    out["expected_gbp"] = expected_from_quantiles(q_gbp, list(cfg.quantiles)).ravel()
    return out


def report_metrics(frame: pd.DataFrame, cfg: Config, label: str) -> dict:
    qcols = [f"q{int(round(t * 100)):02d}" for t in cfg.quantiles]
    m = gbp_metrics(frame["actual_gbp"].to_numpy(), frame[qcols].to_numpy(), list(cfg.quantiles))
    print(f"\n[{label}] n={m['n']:,}  MAE={m['MAE_gbp']:,.0f} GBP  RMSE={m['RMSE_gbp']:,.0f} GBP")
    print(f"[{label}] MedAE={m['MedAE_gbp']:,.1f} GBP  MAE(log1p)={m['MAE_log']:.3f}  "
          f"wQL={m['wQL']:.4f}  sMAPE={m['sMAPE']:.3f}")
    print(f"[{label}] total actual={m['total_actual_gbp']:,.0f}  "
          f"total median forecast={m['total_pred_median_gbp']:,.0f}  "
          f"bias={m['bias_gbp']:,.1f} GBP/day")
    outer_low, outer_high = min(cfg.quantiles), max(cfg.quantiles)
    print(f"[{label}] p10-p90 coverage={m['coverage_p10_p90']:.3f} (target 0.80)  "
          f"p{outer_low * 100:.0f}-p{outer_high * 100:.0f} coverage={m['coverage_outer']:.3f} "
          f"(target {outer_high - outer_low:.2f})")
    return m


def naive_baselines(df: pd.DataFrame, frame: pd.DataFrame, cfg: Config) -> dict:
    """Last-value and seasonal-naive baselines on exactly the same rows, in GBP.

    The seasonal one averages the four most recent *same-weekday* days at or before the
    forecast origin.  Getting that alignment right matters, because this is the number the
    model has to beat: a misaligned "seasonal" baseline is really a smear of nearby
    weekdays, which is weaker and would flatter the model.

    Alignment: the day predicted at ``step`` has ``time_idx = origin + step``, where
    ``origin`` is the last OBSERVED index.  We want a key ``origin - back`` whose weekday
    matches, i.e. ``(origin + step) - (origin - back) = step + back`` divisible by 7, so
    ``back = (-step) % 7``.  Worked through: step 1 -> back 6 (gap 7), step 7 -> back 0
    (gap 7), step 8 -> back 6 (gap 14).  Asserted below.
    """
    history = df[[cfg.group_col, "time_idx", cfg.raw_target]].copy()
    lookup = history.set_index([cfg.group_col, "time_idx"])[cfg.raw_target]

    merchant_ids = frame[cfg.group_col].to_numpy()
    origin = frame["origin_time_idx"].to_numpy() - 1  # last observed index
    step = frame["step"].to_numpy()
    actual = frame["actual_gbp"].to_numpy()

    days_back_to_same_weekday = (-step) % 7
    gap = step + days_back_to_same_weekday
    assert np.all(gap % 7 == 0), (
        f"seasonal baseline is not weekday-aligned; offending steps="
        f"{np.unique(step[gap % 7 != 0])[:10]}"
    )

    results = {}
    for name, weeks_back in (("naive_last", [None]), ("seasonal_naive_dow", [0, 1, 2, 3])):
        predicted = np.zeros(len(frame), dtype=np.float64)
        for week in weeks_back:
            if name == "naive_last":
                keys = origin
            else:
                keys = origin - days_back_to_same_weekday - 7 * week
            values = lookup.reindex(
                pd.MultiIndex.from_arrays([merchant_ids, keys])
            ).to_numpy(dtype=np.float64)
            # A key before the merchant's series starts has no row; 0 is the right fill,
            # since the grid already represents "no trading" as 0.
            predicted += np.nan_to_num(values, nan=0.0)
        predicted /= len(weeks_back)
        results[name] = dict(
            MAE_gbp=float(np.abs(predicted - actual).mean()),
            RMSE_gbp=float(np.sqrt(((predicted - actual) ** 2).mean())),
            MAE_log=float(np.abs(np.log1p(predicted) - np.log1p(actual)).mean()),
        )
        print(f"[baseline:{name}] MAE={results[name]['MAE_gbp']:,.0f} GBP  "
              f"RMSE={results[name]['RMSE_gbp']:,.0f} GBP  "
              f"MAE(log1p)={results[name]['MAE_log']:.3f}")
    return results


# ======================================================================================
# 9. Future frame for a real forecast
# ======================================================================================
def append_future_rows(history: pd.DataFrame, cfg: Config, horizon: int | None = None,
                       ratio_names: list[str] | None = None) -> pd.DataFrame:
    """History + ``horizon`` future rows, for ONE merchant or for every merchant at once.

    Only the encoder needs observed values; the decoder needs the *known*
    covariates to be right.  So the future rows recompute every calendar feature
    from the date via ``add_calendar_features`` - the same function the training
    frame used - and forward-fill only the things that are genuinely known in
    advance (sector, ISO, current acquirer, integration).  Unknown reals and the
    target are set to 0 in the decoder; pytorch_forecasting never reads them.

    There is one function rather than a single-merchant and a batch version because the
    horizon rows ARE the decoder input: if the two paths ever disagreed about how a future
    row is built, ``forecast_merchant`` and ``forecast_all_merchants`` would return different
    numbers for the same merchant and nothing would say so.  ``--selftest`` asserts they
    agree row-for-row.

    Each merchant's block is anchored on that merchant's OWN last observed day, which is what
    ``predict_mode`` does for a window and therefore the only anchor that keeps the encoder
    contiguous with the decoder.  For a merchant that stopped trading a year ago that means
    the "forecast" dates are in the past; forecast_all_merchants reports the staleness per
    merchant rather than quietly dropping or re-anchoring them.
    """
    # `horizon if horizon is not None else cfg.horizon`: let a caller override the horizon
    # for one call without changing the config.  Written explicitly rather than as
    # `horizon or cfg.horizon`, because `or` also replaces a deliberate 0.
    horizon = cfg.horizon if horizon is None else horizon
    # Passed in rather than read off ``history.attrs`` because .attrs does not reliably
    # survive a boolean filter or a concat, and a silently empty ratio list would leave the
    # mix ratios of the last observed day copied across the whole horizon.
    ratio_names = list(history.attrs.get("ratio_names", []) if ratio_names is None
                       else ratio_names)
    history = history.sort_values([cfg.group_col, "time_idx"]).reset_index(drop=True)

    # index.repeat() lays the copies out contiguously per merchant (row0 x horizon, then
    # row1 x horizon, ...), which is exactly the layout np.tile(arange(1, horizon + 1))
    # numbers - so the day offsets line up with the repeated rows without a merge.
    last_rows = history.groupby(cfg.group_col, observed=True, sort=False).tail(1)
    future = last_rows.loc[last_rows.index.repeat(horizon)].reset_index(drop=True)
    offset = np.tile(np.arange(1, horizon + 1), len(last_rows))
    future["time_idx"] = future["time_idx"].to_numpy() + offset
    future[cfg.date_col] = future[cfg.date_col] + pd.to_timedelta(offset, unit="D")

    # Everything observed rather than known is zeroed, so a copied value can never be
    # mistaken for a real one.  These are all time_varying_unknown_reals, which
    # pytorch_forecasting reads in the encoder only, so the zeros are never used - the
    # point is that if a future refactor ever did read them, it would read zeros and not
    # 30 copies of the last observed day.
    # log_merchant_age and log_tenure_at_series_start are excluded because they are KNOWN
    # reals, not observations: the merchant's age on a future date is simply arithmetic.
    KNOWN_LOG_FEATURES = ("log_merchant_age", "log_tenure_at_series_start")
    observed_cols = [c for c in future.columns
                     if c.startswith(("log_", "is_zero_day", "had_no_attempts",
                                      "target_imputed"))]
    observed_cols += ratio_names
    observed_cols += [cfg.raw_target, cfg.target, "daily_transaction_amount",
                      "daily_successful_transaction_count", "daily_unsuccessful_amount_gbp",
                      "daily_fraud_amount", "daily_dispute_amount"]
    for col in dict.fromkeys(observed_cols):
        if col in future.columns and col not in KNOWN_LOG_FEATURES:
            future[col] = 0.0

    future = add_calendar_features(future, cfg.date_col)
    if "merchant_created_datetime" in future.columns:
        age_days = ((future[cfg.date_col] - future["merchant_created_datetime"])
                    .dt.days.fillna(0).clip(lower=0))
        future["since_merchant_creation"] = age_days.astype(np.float32)
        future["log_merchant_age"] = np.log1p(age_days.to_numpy()).astype(np.float32)
    for col in ("day_of_week", "day_of_month", "calendar_month", "day_type",
                "is_uk_bank_holiday"):
        if col in future.columns:
            future[col] = future[col].astype(str)

    combined = pd.concat([history, future], ignore_index=True)
    combined[cfg.target] = combined[cfg.target].astype(np.float32)
    combined.attrs["ratio_names"] = ratio_names
    # Sorted so each merchant's history and its horizon are one contiguous ascending block.
    # For a single merchant this is a no-op (history and future are each already ascending);
    # for many merchants the raw concat interleaves them, and _construct_index would then see
    # a frame whose consecutive rows jump between merchants and time steps.
    return combined.sort_values([cfg.group_col, "time_idx"]).reset_index(drop=True)


def make_future_frame(df: pd.DataFrame, merchant_id: str, cfg: Config,
                      horizon: int | None = None) -> pd.DataFrame:
    """``append_future_rows`` restricted to one merchant - the single-forecast entry point."""
    history = df[df[cfg.group_col] == str(merchant_id)]
    if history.empty:
        raise SystemExit(f"merchant {merchant_id!r} not found in the prepared frame")
    return append_future_rows(history, cfg, horizon,
                              ratio_names=df.attrs.get("ratio_names", []))


# ======================================================================================
# 10. Plumbing
# ======================================================================================
def pick_device(requested: str | None = None) -> tuple[str, str]:
    """The original script hard-coded ``accelerator="cuda"``; this machine is macOS."""
    if requested and requested != "auto":
        acc = requested
    elif torch.cuda.is_available():
        acc = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        acc = "mps"
    else:
        acc = "cpu"
    precision = {"cuda": "bf16-mixed", "mps": "32-true", "cpu": "32-true"}[acc]
    print(f"[device] accelerator={acc} precision={precision}")
    return acc, precision


def trainer_kwargs(acc: str) -> dict:
    """A FRESH dict per predict() call.

    ``BaseModel.predict`` does ``trainer_kwargs.setdefault("callbacks", ... + [cb])``,
    which mutates the dict it is handed.  Reuse the same dict for a second call and
    ``setdefault`` finds the first call's callback list already present, so the new
    PredictCallback is never attached and ``.result`` comes back as the empty list it
    was initialised to.
    """
    # logger/enable_checkpointing are False *for prediction*, and that does not affect what
    # gets saved from training.  Trained checkpoints come from the ModelCheckpoint callback
    # in train(), which already writes the two best epochs plus last.ckpt to
    # <outdir>/checkpoints_<tag>/, and metrics come from the TensorBoardLogger and CSVLogger
    # also configured there.  Setting them True here only makes every predict() call spin up
    # its own logger directory and checkpoint callback for a pass that computes no loss and
    # updates no weights - empty version_N folders, no extra information.
    return dict(accelerator=acc, devices=1, logger=False, enable_checkpointing=False)


def default_workers(acc: str, override: int | None) -> int:
    if override is not None:
        return override
    if platform.system() == "Darwin":
        return 0  # spawn-based workers on macOS cost more than they save here
    return max(0, min(8, (os.cpu_count() or 2) - 2))


def load_checkpoint(path: str) -> RevenueTFT:
    """Load without globally disabling torch's unpickler.

    The original script permanently replaced ``torch.load`` with a version that
    forces ``weights_only=False``.  Scope it to this one call instead.
    """
    try:
        return RevenueTFT.load_from_checkpoint(path, weights_only=False)
    except TypeError:
        real = torch.load
        try:
            torch.load = lambda *a, **k: real(*a, **{**k, "weights_only": False})
            return RevenueTFT.load_from_checkpoint(path)
        finally:
            torch.load = real


class EpochTimer(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._t = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(self, "_t", None) is None:
            return
        m = trainer.callback_metrics
        parts = [f"epoch {trainer.current_epoch}", f"{time.time() - self._t:.0f}s"]
        for k in ("train_loss_epoch", "val_loss", "val_MAE_log", "val_MedAE_gbp",
                  "val_MAE_gbp"):
            if k in m:
                parts.append(f"{k}={float(m[k]):,.4f}")
        # leading newline so the line survives the progress bar's carriage returns
        print("\n[fit] " + "  ".join(parts), flush=True)


# ======================================================================================
# 11. Pipeline stages
# ======================================================================================
def prepare_data(cfg: Config) -> pd.DataFrame:
    global CALENDAR_LOOKUP
    df = load_frame(cfg.data_path)
    df = coerce_dtypes(df, cfg)
    df = df.drop_duplicates(subset=[cfg.group_col, cfg.date_col], keep="last")
    # Captured BEFORE the daily grid fill, while every date still has its warehouse row.
    # Module-level so make_future_frame() reaches the same calendar without threading it
    # through every call site.
    CALENDAR_LOOKUP = build_calendar_lookup(df, cfg.date_col)
    df = to_daily_grid(df, cfg)
    df = engineer_features(df, cfg)
    df = filter_series(df, cfg)

    y = df[cfg.raw_target]
    print(f"[data] target GBP: mean={y.mean():,.0f} median={y.median():,.0f} "
          f"max={y.max():,.0f} zero-days={100 * (y == 0).mean():.1f}%")
    print(f"[data] log1p target: mean={df[cfg.target].mean():.3f} "
          f"std={df[cfg.target].std():.3f} skew={df[cfg.target].skew():.3f} "
          f"(raw skew={y.skew():.1f})")
    assert df[[cfg.target, "time_idx"]].notna().all().all(), "NaNs in target/time_idx"
    return df


def build_model(split: Split, cfg: Config) -> RevenueTFT:
    model = RevenueTFT.from_dataset(
        split.training,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_continuous_size,
        loss=QuantileLoss(quantiles=list(cfg.quantiles)),
        optimizer=cfg.optimizer,
        weight_decay=cfg.weight_decay,
        reduce_on_plateau_patience=cfg.reduce_on_plateau_patience,
        log_interval=-1,
        group_id_dropout=cfg.group_id_dropout,
    )
    model.group_id_dropout = cfg.group_id_dropout
    print(f"[model] {model.size() / 1e3:,.1f}k parameters, "
          f"loss={type(model.loss).__name__}{list(cfg.quantiles)}")
    print(f"[model] group_id_dropout={model.group_id_dropout} "
          f"(merchant_id at x_cat index {model.group_cat_index})")
    if cfg.group_id_dropout > 0 and model.group_cat_index is None:
        raise RuntimeError(
            f"{RevenueTFT.GROUP_COLUMN!r} is not among the model's categorical inputs, so "
            "group_id_dropout would silently do nothing. Add it to static_categoricals."
        )
    # The two invariants that quietly destroyed the previous model.
    assert cfg.target in model.encoder_variables, (
        f"{cfg.target!r} missing from encoder_variables -> the encoder cannot see the "
        f"target's own history. encoder_variables={model.encoder_variables}"
    )
    assert cfg.target not in model.decoder_variables, (
        f"{cfg.target!r} leaked into decoder_variables"
    )
    # These two asserts are the guard rails on the fix that mattered most.  The first fails
    # if the encoder cannot see revenue history (the defect that made the previous model
    # forecast a flat line); the second fails if it can see it in the DECODER, which would
    # be reading the future and would make the validation scores meaningless.  Both are
    # cheap and run once per build, and both failure modes are otherwise invisible.

    print(f"[model] encoder sees {len(model.encoder_variables)} variables "
          f"(target included), decoder sees {len(model.decoder_variables)}, "
          f"statics {len(model.static_variables)}")
    return model


def train(model: RevenueTFT, split: Split, cfg: Config, acc: str, precision: str,
          outdir: Path, tag: str, max_epochs: int | None = None,
          early_stop: bool = True) -> tuple[pl.Trainer, str]:
    workers = default_workers(acc, cfg.num_workers)
    kw = dict(num_workers=workers, pin_memory=acc == "cuda")
    if workers > 0:
        kw["persistent_workers"] = True
    train_dl = split.training.to_dataloader(train=True, batch_size=cfg.batch_size, **kw)
    val_dl = split.validation.to_dataloader(train=False, batch_size=cfg.batch_size * 2, **kw)

    ckpt = ModelCheckpoint(
        dirpath=str(outdir / f"checkpoints_{tag}"),
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss" if early_stop else None,
        mode="min", save_top_k=2 if early_stop else 1, save_last=True,
    )
    callbacks = [
        ckpt,
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=20),
        EpochTimer(),
    ]
    if early_stop:
        callbacks.insert(1, EarlyStopping(monitor="val_loss", min_delta=1e-4,
                                          patience=cfg.patience, mode="min", verbose=True))
    trainer = pl.Trainer(
        # the per-call override if one was passed, otherwise the config value
        max_epochs=cfg.max_epochs if max_epochs is None else max_epochs,
        accelerator=acc,
        devices=1,
        precision=precision,
        gradient_clip_val=cfg.gradient_clip_val,
        limit_train_batches=cfg.train_batches_per_epoch,
        enable_model_summary=True,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(str(outdir / "tb"), name=tag),
            CSVLogger(str(outdir / "csv"), name=tag),
        ],
        log_every_n_steps=25,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best = (ckpt.best_model_path if early_stop else "") or ckpt.last_model_path
    print(f"[fit] checkpoint: {best} (val_loss={ckpt.best_model_score})")
    return trainer, best


def evaluate(model: RevenueTFT, split: Split, df: pd.DataFrame, cfg: Config,
             acc: str, outdir: Path) -> dict:
    results = {}
    for label, ds in (("validation", split.validation), ("holdout", split.holdout)):
        if len(ds) == 0:
            print(f"[{label}] empty, skipping")
            continue
        pred = model.predict(ds.to_dataloader(train=False, batch_size=cfg.batch_size * 2,
                                              num_workers=0),
                             mode="quantiles", return_x=True, return_y=True,
                             return_index=True, trainer_kwargs=trainer_kwargs(acc))
        frame = predictions_to_frame(pred, ds, cfg)
        frame.to_csv(outdir / f"predictions_{label}.csv", index=False)
        results[label] = report_metrics(frame, cfg, label)
        if label == "holdout":
            results["holdout_baselines"] = naive_baselines(df, frame, cfg)
            _report_by_size(frame, cfg, df)
    return results


def _report_by_size(frame: pd.DataFrame, cfg: Config, df: pd.DataFrame) -> None:
    """Per-size-band errors: proves the loss is not owned by the whales."""
    level = df.groupby(cfg.group_col, observed=True)[cfg.raw_target].mean()
    f = frame.copy()
    f["merchant_mean_gbp"] = f[cfg.group_col].map(level)
    # Mean daily revenue in GBP, in decades: under 100/day, 100-1k, 1k-10k, 10k-100k, and
    # over 100k.  Reported separately because a single average error hides whether the model
    # is good everywhere or just on the whales - which is precisely how the previous
    # GBP-scale loss went wrong.
    bands = [0, 100, 1_000, 10_000, 100_000, np.inf]
    labels = ["<100", "100-1k", "1k-10k", "10k-100k", ">100k"]
    f["band"] = pd.cut(f["merchant_mean_gbp"], bands, labels=labels)
    print("\n[holdout] error by merchant size band (mean daily GBP)")
    print(f"{'band':>10} {'n':>7} {'MAE_gbp':>12} {'MAE_log1p':>10} {'sMAPE':>7}")
    for b in labels:
        s = f[f["band"] == b]
        if s.empty:
            continue
        a, p = s["actual_gbp"].to_numpy(), s["q50"].to_numpy()
        sm = (np.abs(p - a) / np.maximum((np.abs(a) + np.abs(p)) / 2, 1e-9)).mean()
        print(f"{b:>10} {len(s):>7,} {np.abs(p - a).mean():>12,.1f} "
              f"{np.abs(np.log1p(p) - np.log1p(a)).mean():>10.3f} {sm:>7.3f}")


def make_plots(model: RevenueTFT, split: Split, cfg: Config, acc: str, outdir: Path) -> None:
    raw = model.predict(
        split.holdout.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=0),
        mode="raw", return_x=True, return_y=True, trainer_kwargs=trainer_kwargs(acc),
    )
    n = min(cfg.n_example_plots, raw.output["prediction"].shape[0])
    for i in range(n):
        fig = model.plot_prediction(raw.x, raw.output, idx=i, add_loss_to_title=True)
        fig.savefig(outdir / f"holdout_example_{i + 1:02d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"[plots] wrote {n} holdout example plots (y-axis is log1p GBP)")

    interp = model.interpret_output(raw.output, reduction="sum")
    figs = model.plot_interpretation(interp)
    for name, fig in (figs.items() if isinstance(figs, dict) else enumerate(figs)):
        fig.savefig(outdir / f"interpretation_{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("[plots] wrote interpretation plots")


def assert_trained_horizon(model: RevenueTFT, cfg: Config) -> int:
    """Fail if cfg.horizon disagrees with the horizon the checkpoint was trained on.

    predict() rebuilds the dataset from the checkpoint's own dataset_parameters, so the
    decoder length comes from the TRAINED max_prediction_length, not from cfg.horizon.  If
    they disagree, the appended future rows and the decoder are different lengths and every
    forecast date silently shifts.  (Deliberately widening the horizon at prediction time is
    discussed in future_considerations.md; it is not something to do by accident via
    --horizon.)
    """
    trained_horizon = int(model.dataset_parameters.get("max_prediction_length", cfg.horizon))
    if trained_horizon != cfg.horizon:
        raise SystemExit(
            f"--horizon is {cfg.horizon} but this checkpoint was trained with "
            f"max_prediction_length={trained_horizon}. Re-run with --horizon "
            f"{trained_horizon}, or retrain."
        )
    return trained_horizon


def cold_start_codes(model: RevenueTFT, cfg: Config,
                     merchant_ids: list[str]) -> np.ndarray | None:
    """Which of ``merchant_ids`` had no rows at or before the training cutoff.

    Cold start is a supported path, not an error, but it answers a different question -
    "what would a merchant like this one do" rather than "what does merchant 171 do" - and
    the output alone cannot tell you which, so it is reported explicitly.

    The test is whether the merchant's group code has a row in ``target_normalizer.norm_``.
    norm_ is indexed by group CODE, not by merchant id, and it is fitted on the training
    block only - so "code present" is exactly "this merchant had training rows", which is
    the thing that decides whether its own centre and scale are available instead of the
    median fallback.  Encoding first and looking the code up is essential:
    `str(merchant_id) in norm_.index` compares a merchant id against integer codes and is
    always False, which would report cold start for everything.

    Returns None if the checkpoint does not expose what is needed - a diagnostic must never
    be able to stop a forecast.
    """
    try:
        group_encoder = model.dataset_parameters["categorical_encoders"][cfg.group_col]
        codes = np.asarray(group_encoder.transform(pd.Series([str(m) for m in merchant_ids])))
        known = np.asarray(model.dataset_parameters["target_normalizer"].norm_.index)
        return ~np.isin(codes, known)
    except Exception:  # pragma: no cover - never let a diagnostic stop a forecast
        return None


def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, merchant_id: str,
                  cfg: Config, path: Path) -> None:
    """Observed history + median + reported band + expected value, saved to ``path``.

    Shared by the single-merchant and all-merchant paths so the chart cannot drift between
    them.  ``history`` needs cfg.date_col and cfg.raw_target; ``forecast`` needs "date", the
    two outer report quantiles, "q50" and "expected_gbp".
    """
    lo_col = f"q{int(round(min(cfg.report_quantiles) * 100)):02d}"
    hi_col = f"q{int(round(max(cfg.report_quantiles) * 100)):02d}"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(history[cfg.date_col], history[cfg.raw_target], color="#1f77b4", lw=1.6,
            marker="o", ms=3, label="Observed")
    ax.plot(forecast["date"], forecast["q50"], color="#ff7f0e", lw=2, ls="--", marker="o", ms=4,
            label="Forecast (median, p50)")
    ax.fill_between(forecast["date"], forecast[lo_col], forecast[hi_col], color="#ff7f0e",
                    alpha=0.25,
                    label=f"p{min(cfg.report_quantiles) * 100:.0f}-"
                          f"p{max(cfg.report_quantiles) * 100:.0f}")
    ax.plot(forecast["date"], forecast["expected_gbp"], color="#2ca02c", lw=1.4, ls=":",
            label="Forecast (expected value)")
    ax.set_title(f"TPV forecast - merchant {merchant_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily successful amount (GBP)")

    # The axis is scaled from the OBSERVED history and the reported band only.  Left to
    # itself, matplotlib scaled to the p98 line - one exponentiated tail quantile at GBP 30m
    # against a history that never passed GBP 20k - so it collapsed every real value onto the
    # zero line and relabelled the axis 1e7.  Padding the observed range and clipping to it
    # keeps the history legible; an offscreen band is a visible signal that the forecast is
    # over-dispersed, rather than something that silently rescales the whole chart.
    visible = np.concatenate([
        history[cfg.raw_target].to_numpy(dtype=np.float64),
        forecast[lo_col].to_numpy(dtype=np.float64),
        forecast["q50"].to_numpy(dtype=np.float64),
        forecast[hi_col].to_numpy(dtype=np.float64),
        forecast["expected_gbp"].to_numpy(dtype=np.float64),
    ])
    top = float(np.nanmax(visible)) if len(visible) else 1.0
    ax.set_ylim(0.0, max(top * 1.1, 1.0))
    # Plain thousands separators instead of 1e7 offset notation.  A FuncFormatter rather than
    # ticklabel_format(style="plain"), because the latter only works on a ScalarFormatter and
    # would raise once the formatter is replaced.
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def forecast_merchant(model: RevenueTFT, df: pd.DataFrame, cfg: Config, acc: str,
                      outdir: Path, merchant_id: str) -> pd.DataFrame:
    """The long-form forecast for ONE merchant: full table on stdout, CSV and PNG in outdir.

    ``forecast_all_merchants`` covers every merchant including this one and is what a
    production run reads; this exists because the per-merchant sanity checks (level ratio,
    dispersion ratio, cold start, quantile crossings, recent observed level) are worth
    reading in full for one merchant, and unreadable for 384.
    """
    frame = make_future_frame(df, merchant_id, cfg)
    # All observed rows, zeros included.  Filtering to revenue > 0 would (a) make "the last
    # 28 observed days" span more than 28 calendar days for an intermittent merchant and so
    # overstate the recent level, and (b) hide the zero days from the chart, which is
    # exactly the behaviour a reader needs to see.
    history = frame[frame["time_idx"] <= frame["time_idx"].max() - cfg.horizon]

    cold = cold_start_codes(model, cfg, [str(merchant_id)])
    is_cold_start = None if cold is None else bool(cold[0])
    if is_cold_start:
        print(f"[forecast] COLD START: merchant {merchant_id} has no rows at or before the "
              f"training cutoff, so its merchant embedding was never updated and its target "
              f"centre/scale is the across-merchant median rather than its own.  The level of "
              f"this forecast is therefore not trustworthy - re-run with --refit-full, which "
              f"puts every day of its history into training.")
    elif is_cold_start is False:
        print(f"[forecast] merchant {merchant_id} has its own trained embedding and its own "
              f"fitted target scale")

    assert_trained_horizon(model, cfg)

    pred = model.predict(frame, mode="quantiles", return_x=True, return_index=True,
                         trainer_kwargs=trainer_kwargs(acc))

    q_log = pred.output.detach().cpu().numpy()[0]
    q_gbp = np.expm1(q_log.astype(np.float64)).clip(min=0)

    # QuantileLoss gives the head one INDEPENDENT output channel per level, with nothing tying
    # them together, so nothing guarantees q35 <= q50 <= q65.  Adjacent levels cross most
    # often, precisely because they are closest - the first run of this reported p35 above p50
    # on 10 of 30 days, which is indefensible in a delivered CSV.  Sorting each day's
    # quantiles into order is monotone rearrangement (Chernozhukov, Fernandez-Val & Galichon
    # 2010): it is not a cosmetic patch, it provably cannot increase the error of the
    # estimated quantile function, because the true one is monotone by definition and sorting
    # moves the estimate no further from it.  Reported rather than silent, since a high
    # crossing count means the levels are barely distinguishable and the run is under-trained.
    crossings = int((np.diff(q_gbp, axis=1) < 0).sum())
    q_gbp = np.sort(q_gbp, axis=1)
    if crossings:
        print(f"[forecast] repaired {crossings} quantile crossing(s) across "
              f"{q_gbp.shape[0]} days x {q_gbp.shape[1]} levels by monotone rearrangement"
              + ("  <- high; the levels are barely separated, train longer"
                 if crossings > q_gbp.shape[0] else ""))

    t_idx = pred.x["decoder_time_idx"].detach().cpu().numpy()[0]
    last_t = int(frame["time_idx"].max()) - cfg.horizon
    last_date = frame.loc[frame["time_idx"] == last_t, cfg.date_col].iloc[0]
    dates = last_date + pd.to_timedelta(t_idx - last_t, unit="D")

    forecast = pd.DataFrame({"date": dates, "time_idx": t_idx})
    for i, tau in enumerate(cfg.quantiles):
        forecast[f"q{int(round(tau * 100)):02d}"] = q_gbp[:, i]
    forecast["expected_gbp"] = expected_from_quantiles(q_gbp, list(cfg.quantiles))

    # What lands in the CSV.  Only a narrow band around the median is reported, because the
    # width of the outer quantiles is a statement about log space that expm1 turns into a
    # number nobody can act on: the same +-2 sigma that reads as a sane band on the log-scale
    # holdout plots becomes a four-orders-of-magnitude range in GBP.  p35/p50/p65 is the part
    # of the distribution that survives the inverse transform as a usable daily figure.  The
    # full set stays in the printed table and in predictions_holdout.csv for diagnostics.
    report_cols = ["time_idx", "date"]
    for tau in cfg.report_quantiles:
        report_cols.append(f"q{int(round(tau * 100)):02d}")
    report_cols.append("expected_gbp")

    print(f"\n--- {cfg.horizon}-day forecast for merchant {merchant_id} ---")
    print(forecast[report_cols].to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    recent = history[cfg.raw_target].tail(28)
    print(f"\n[forecast] last 28 observed days: mean={recent.mean():,.0f} "
          f"median={recent.median():,.0f} GBP")
    print(f"[forecast] horizon median: mean={forecast['q50'].mean():,.0f} GBP  "
          f"expected: mean={forecast['expected_gbp'].mean():,.0f} GBP")
    print(f"[forecast] horizon total (median)={forecast['q50'].sum():,.0f} GBP  "
          f"(expected)={forecast['expected_gbp'].sum():,.0f} GBP")
    level_ratio = forecast["q50"].mean() / max(recent.mean(), 1e-9)
    print(f"[forecast] level check: horizon median / recent mean = {level_ratio:.2f} "
          f"({'plausible' if 0.2 <= level_ratio <= 5 else 'SUSPICIOUS - investigate'})")
    # E[Y] from a lognormal-ish predictive distribution is median * exp(sigma^2/2), so a
    # huge expected/median ratio means the predictive spread in log space is enormous -
    # i.e. an under-trained or over-dispersed model, not a useful mean forecast.
    dispersion_ratio = forecast["expected_gbp"].mean() / max(forecast["q50"].mean(), 1e-9)
    print(f"[forecast] dispersion check: expected / median = {dispersion_ratio:.2f} "
          f"({'ok' if dispersion_ratio <= 6 else 'TOO WIDE - the log-space quantile spread is large, '
             'so the expected-value column is unreliable; trust q50 and train longer'})")
    forecast[report_cols].to_csv(outdir / f"forecast_merchant_{merchant_id}.csv", index=False)

    plot_forecast(history, forecast, merchant_id, cfg,
                  outdir / f"forecast_merchant_{merchant_id}.png")
    print(f"[forecast] wrote {outdir / f'forecast_merchant_{merchant_id}.png'}")
    return forecast


def forecast_all_merchants(model: RevenueTFT, df: pd.DataFrame, cfg: Config, acc: str,
                           outdir: Path, plots: int = 0,
                           train_cutoff: int | None = None) -> pd.DataFrame:
    """One ``horizon``-day forecast per merchant, in ONE batched forward pass.

    Writes ``<outdir>/<cfg.forecast_dir>/forecast_merchant_<id>.csv`` for every merchant in
    the prepared frame - same columns, same units and same monotone-rearrangement repair as
    the single-merchant CSV - plus ``<outdir>/forecast_summary.csv``, one row per merchant,
    which is where you look to decide which of those forecasts to believe.

    Why one pass rather than a loop over ``forecast_merchant``: each call to
    ``BaseModel.predict`` builds a TimeSeriesDataSet and spins up a fresh Lightning Trainer,
    so 384 calls pay that fixed cost 384 times for 384 single-window batches.  Appending the
    horizon to every merchant at once and predicting the resulting frame gives exactly the
    same windows - ``predict_mode`` keeps one window per series, anchored at that series'
    last row, which is that merchant's last horizon row - in one Trainer and a handful of
    batches.  ``--selftest`` checks the equality rather than assuming it.

    Nothing here re-fits anything: ``from_parameters`` reuses the checkpoint's own fitted
    scalers, categorical encoders and target normaliser, so a merchant's numbers do not
    depend on which other merchants happen to be in the frame.

    ``plots``: 0 = none, -1 = every merchant, N > 0 = the N largest by mean daily revenue.
    """
    assert_trained_horizon(model, cfg)
    forecast_dir = outdir / cfg.forecast_dir
    forecast_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ the forward pass
    combined = append_future_rows(df, cfg, cfg.horizon,
                                  ratio_names=df.attrs.get("ratio_names", []))
    n_merchants = df[cfg.group_col].nunique()
    print(f"\n[forecast-all] {n_merchants} merchants x {cfg.horizon} days: "
          f"{len(combined):,} frame rows (history + horizon)")
    started = time.time()
    pred = model.predict(combined, mode="quantiles", return_x=True, return_index=True,
                         batch_size=cfg.batch_size, num_workers=0,
                         trainer_kwargs=trainer_kwargs(acc))
    ids = pred.index[cfg.group_col].astype(str).to_numpy()
    q_log = pred.output.detach().cpu().numpy()               # (n_series, horizon, n_levels)
    t_idx = pred.x["decoder_time_idx"].detach().cpu().numpy()  # (n_series, horizon)
    print(f"[forecast-all] forward pass: {len(ids)} windows in {time.time() - started:.1f}s")

    # A merchant with fewer than min_encoder_length observed days produces no window at all,
    # and pytorch_forecasting only warns about it.  Say which ones, loudly: a silently absent
    # merchant is a missing file that looks like a merchant that simply was not requested.
    missing = sorted(set(df[cfg.group_col].astype(str)) - set(ids))
    if missing:
        days = df.groupby(cfg.group_col, observed=True).size()
        print(f"[forecast-all] WARNING: {len(missing)} merchant(s) produced no forecast "
              f"window and have NO csv - they have fewer than min_encoder_length="
              f"{cfg.min_encoder_length} observed days: "
              + ", ".join(f"{m} ({int(days.get(m, 0))}d)" for m in missing[:15])
              + (" ..." if len(missing) > 15 else ""))

    # ------------------------------------------------------------------ log space -> GBP
    q_gbp = np.expm1(q_log.astype(np.float64)).clip(min=0)
    # Same monotone rearrangement as the single-merchant path, per merchant per day, over the
    # quantile axis.  See forecast_merchant for why this is a correction and not a cosmetic
    # patch.  Counted per merchant so forecast_summary.csv can show which forecasts had
    # barely-separated levels.
    crossings_per_merchant = (np.diff(q_gbp, axis=2) < 0).sum(axis=(1, 2)).astype(int)
    q_gbp = np.sort(q_gbp, axis=2)
    expected = np.asarray(expected_from_quantiles(q_gbp, list(cfg.quantiles)))
    total_crossings = int(crossings_per_merchant.sum())
    n_affected = int((crossings_per_merchant > 0).sum())
    print(f"[forecast-all] repaired {total_crossings} quantile crossing(s) by monotone "
          f"rearrangement, affecting {n_affected}/{len(ids)} merchants "
          f"({total_crossings / max(len(ids) * cfg.horizon, 1):.2f} per forecast day)")

    # ------------------------------------------------------------------ dates and history
    # Looked up per (merchant, time_idx) out of the frame that produced the decoder rather
    # than recomputed from an offset, so the date on a row is by construction the date the
    # model was given known covariates for.
    date_lookup = combined.set_index([cfg.group_col, "time_idx"])[cfg.date_col]
    flat = pd.MultiIndex.from_arrays([np.repeat(ids, cfg.horizon), t_idx.ravel()])
    dates = date_lookup.reindex(flat).to_numpy().reshape(t_idx.shape)
    assert not pd.isna(dates).any(), "a decoder time_idx has no date in the forecast frame"

    by_merchant = df.sort_values([cfg.group_col, "time_idx"]).groupby(cfg.group_col,
                                                                     observed=True)
    last_t = by_merchant["time_idx"].max()
    last_date = by_merchant[cfg.date_col].max()
    history_days = by_merchant.size()
    # Days at or before the training cutoff, which is what decides whether a merchant could
    # produce a training WINDOW - see had_training_windows below.
    train_days = (df.loc[df["time_idx"] <= train_cutoff]
                  .groupby(cfg.group_col, observed=True).size()
                  if train_cutoff is not None else None)
    # Same 28-day window as the single-merchant level check, zeros included.
    recent = (df.sort_values([cfg.group_col, "time_idx"])
              .groupby(cfg.group_col, observed=True).tail(28)
              .groupby(cfg.group_col, observed=True)[cfg.raw_target].mean())
    names = (by_merchant["merchant_name"].last() if "merchant_name" in df.columns
             else pd.Series("", index=history_days.index))
    max_t = int(df["time_idx"].max())
    cold = cold_start_codes(model, cfg, list(ids))
    report_cols = (["time_idx", "date"]
                   + [f"q{int(round(t * 100)):02d}" for t in cfg.report_quantiles]
                   + ["expected_gbp"])
    level_positions = {f"q{int(round(t * 100)):02d}": list(cfg.quantiles).index(t)
                       for t in cfg.report_quantiles}
    median_pos = (list(cfg.quantiles).index(0.5) if 0.5 in cfg.quantiles
                  else len(cfg.quantiles) // 2)

    # ------------------------------------------------------------------ write the CSVs
    rows = []
    written = 0
    for i, merchant in enumerate(ids):
        forecast = pd.DataFrame({"time_idx": t_idx[i], "date": dates[i]})
        for col, pos in level_positions.items():
            forecast[col] = q_gbp[i, :, pos]
        forecast["expected_gbp"] = expected[i]
        # q50 is needed by the chart and by the level check even if report_quantiles were
        # ever changed to exclude it.
        if "q50" not in forecast.columns:
            forecast["q50"] = q_gbp[i, :, median_pos]
        forecast[report_cols].to_csv(
            forecast_dir / f"forecast_merchant_{merchant}.csv", index=False
        )
        written += 1

        recent_mean = float(recent.get(merchant, np.nan))
        median_total = float(forecast["q50"].sum())
        expected_total = float(forecast["expected_gbp"].sum())
        rows.append({
            cfg.group_col: merchant,
            "merchant_name": names.get(merchant, ""),
            "history_days": int(history_days.get(merchant, 0)),
            "last_observed_date": pd.Timestamp(last_date.get(merchant)).date(),
            "days_since_last_observation": int(max_t - int(last_t.get(merchant))),
            "forecast_start_date": pd.Timestamp(dates[i][0]).date(),
            "forecast_end_date": pd.Timestamp(dates[i][-1]).date(),
            "recent28_mean_gbp": recent_mean,
            "horizon_total_median_gbp": median_total,
            "horizon_total_expected_gbp": expected_total,
            # The two checks forecast_merchant prints, kept per merchant so 384 forecasts can
            # be triaged by sorting a column instead of by reading 384 terminal reports.
            # NaN rather than a huge number when the merchant earned literally nothing in its
            # last 28 observed days: there is no recent level to be a multiple of, and
            # 1.4e12 sorts to the top of the file and buries the forecasts that are genuinely
            # wrong.  Those merchants are counted separately below.  A merchant on GBP 1/day
            # does keep its ratio - a huge multiple of a tiny but real level is exactly the
            # kind of forecast that should sort to the top.
            "level_ratio": (median_total / cfg.horizon / recent_mean
                            if recent_mean > 0 else np.nan),
            "dispersion_ratio": (expected_total / median_total if median_total > 0
                                 else np.nan),
            "cold_start": None if cold is None else bool(cold[i]),
            # Distinct from cold_start, and the sharper test.  norm_ has a row for any
            # merchant with rows at or before the cutoff, but a TRAINING WINDOW needs
            # min_encoder_length + horizon consecutive days - so a merchant just above
            # min_series_days gets its own centre and scale (cold_start False) while its
            # identity embedding was never once updated by a gradient step.
            "had_training_windows": (
                None if train_days is None
                else bool(int(train_days.get(merchant, 0))
                          >= cfg.min_encoder_length + cfg.horizon)
            ),
            "quantile_crossings": int(crossings_per_merchant[i]),
        })

    summary = pd.DataFrame(rows).sort_values("horizon_total_expected_gbp", ascending=False)
    summary.to_csv(outdir / "forecast_summary.csv", index=False)
    print(f"[forecast-all] wrote {written} csv files to {forecast_dir.resolve()}")
    print(f"[forecast-all] wrote {(outdir / 'forecast_summary.csv').resolve()}")

    # ------------------------------------------------------------------ what to trust
    stale = summary["days_since_last_observation"]
    print(f"\n[forecast-all] horizon totals: median={summary['horizon_total_median_gbp'].sum():,.0f} "
          f"GBP  expected={summary['horizon_total_expected_gbp'].sum():,.0f} GBP "
          f"across {written} merchants")
    print(f"[forecast-all] each merchant's horizon starts the day after ITS OWN last "
          f"observed day.  {int((stale <= 1).sum())}/{written} are current (<=1 day behind "
          f"the extract); {int((stale > 30).sum())} are more than 30 days behind and their "
          f"'forecast' dates are therefore in the past - see days_since_last_observation "
          f"and forecast_start_date in forecast_summary.csv")
    if summary["cold_start"].notna().any():
        print(f"[forecast-all] {int(summary['cold_start'].fillna(False).sum())} cold-start "
              f"merchant(s) on the median target scale rather than their own")
    if summary["had_training_windows"].notna().any():
        untrained = summary.loc[summary["had_training_windows"] == False]  # noqa: E712
        print(f"[forecast-all] {len(untrained)} merchant(s) had NO training window "
              f"(fewer than min_encoder_length + horizon = "
              f"{cfg.min_encoder_length + cfg.horizon} days at or before the training "
              f"cutoff), so their merchant embedding was never updated"
              + (f": {list(untrained[cfg.group_col])[:10]}" if len(untrained) else ""))
    dormant = int((summary["recent28_mean_gbp"] <= 0).sum())
    if dormant:
        print(f"[forecast-all] {dormant} merchant(s) earned GBP 0 across their last 28 "
              f"observed days, so level_ratio is undefined (NaN) for them - the level of "
              f"those forecasts cannot be checked this way at all")
    suspicious = summary[(summary["level_ratio"] < 0.2) | (summary["level_ratio"] > 5)
                         | (summary["dispersion_ratio"] > 6)]
    print(f"[forecast-all] {len(suspicious)}/{written} forecasts fail a sanity check "
          f"(level ratio outside [0.2, 5] or dispersion ratio > 6)")
    if len(suspicious):
        cols = [cfg.group_col, "history_days", "days_since_last_observation",
                "recent28_mean_gbp", "horizon_total_median_gbp", "level_ratio",
                "dispersion_ratio"]
        worst = suspicious.reindex(
            suspicious["horizon_total_expected_gbp"].abs().sort_values(ascending=False).index
        ).head(15)
        print("[forecast-all] the 15 largest of them, by forecast size - inspect with "
              "--forecast-merchant <id>:")
        print(worst[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ------------------------------------------------------------------ optional charts
    if plots:
        level = df.groupby(cfg.group_col, observed=True)[cfg.raw_target].mean()
        order = [m for m in level.sort_values(ascending=False).index if m in set(ids)]
        chosen = order if plots < 0 else order[:plots]
        print(f"[forecast-all] rendering {len(chosen)} forecast PNG(s)"
              + ("" if plots < 0 else f" (the {len(chosen)} largest by mean daily revenue)"))
        started = time.time()
        position = {m: i for i, m in enumerate(ids)}
        for merchant in chosen:
            i = position[merchant]
            forecast = pd.DataFrame({"date": dates[i]})
            for col, pos in level_positions.items():
                forecast[col] = q_gbp[i, :, pos]
            forecast["expected_gbp"] = expected[i]
            if "q50" not in forecast.columns:
                forecast["q50"] = q_gbp[i, :, median_pos]
            history = df.loc[df[cfg.group_col] == merchant].sort_values("time_idx")
            plot_forecast(history, forecast, merchant, cfg,
                          forecast_dir / f"forecast_merchant_{merchant}.png")
        elapsed = time.time() - started
        print(f"[forecast-all] {len(chosen)} PNG(s) in {elapsed:.0f}s "
              f"({elapsed / max(len(chosen), 1):.2f}s each)")
    else:
        print("[forecast-all] no PNGs (--forecast-plots -1 for all, N for the N largest "
              "by mean daily revenue)")
    return summary


# ======================================================================================
# 12. Self-test
# ======================================================================================
def selftest(cfg: Config) -> None:
    """Check the things that failed silently last time.  Every claim in MODEL_NOTES.md
    that could be checked in under two minutes is checked here rather than asserted."""
    print("=" * 88, "\nSELF-TEST\n", "=" * 88, sep="")
    global CALENDAR_LOOKUP

    source = coerce_dtypes(load_frame(cfg.data_path), cfg)
    CALENDAR_LOOKUP = build_calendar_lookup(source, cfg.date_col)

    # 1. The extract's calendar is used verbatim, and the formulas that extend it past its
    #    last date agree with it on every date it does cover.  (Bank holidays themselves are
    #    taken straight from is_uk_bank_holiday, so there is nothing to re-verify there.)
    known_dates = source[[cfg.date_col]].drop_duplicates().copy()
    extract_holidays = set(pd.to_datetime(
        CALENDAR_LOOKUP.loc[CALENDAR_LOOKUP["is_uk_bank_holiday"].astype(bool), cfg.date_col]
    ).dt.normalize())
    extrapolated = extrapolate_calendar(
        pd.DatetimeIndex(known_dates[cfg.date_col]), holidays=extract_holidays
    ).rename(columns={"request_date": cfg.date_col})
    warehouse = CALENDAR_LOOKUP.set_index(cfg.date_col)
    formula = extrapolated.set_index(cfg.date_col)
    for col in ["day_of_week", "day_of_month", "calendar_month", "calendar_week_of_year",
                "day_type", "day_name", "month_name"]:
        if col not in warehouse.columns:
            continue
        left = warehouse[col].astype(str)
        right = formula.loc[left.index, col].astype(str)
        agree = float((left == right).mean())
        assert agree == 1.0, (
            f"{col}: extrapolation formula agrees with the extract on only {agree:.1%} of "
            f"dates - the forecast horizon would be labelled differently from history. "
            f"Examples: {left[left != right].head(5).to_dict()}"
        )
    print(f"  [pass] the extrapolation formulas reproduce the extract's own "
          f"day_of_week / day_of_month / calendar_month / calendar_week_of_year / "
          f"day_type / day_name / month_name on all {len(warehouse):,} dates it covers "
          f"(day_type given the extract's own {len(extract_holidays)} bank holidays, since "
          f"that list is the one input a date cannot supply)")

    # 1b. The `holidays` package agrees with the warehouse over the range they overlap.
    #     Past holidays are always read from the warehouse; this only establishes that the
    #     package can be trusted for the future dates the warehouse cannot cover.  Bank
    #     holidays are not derivable from a date - Good Friday and Easter Monday move with
    #     the lunar calendar, and substitute days shift whenever a fixed holiday lands on a
    #     weekend - so a hardcoded list would be right only until the year it stops at.
    covered_from = pd.Timestamp(CALENDAR_LOOKUP[cfg.date_col].min())
    covered_to = pd.Timestamp(CALENDAR_LOOKUP[cfg.date_col].max())
    from_package = set(uk_bank_holidays_between(covered_from, covered_to))
    assert from_package == extract_holidays, (
        f"the `holidays` package disagrees with the warehouse over "
        f"{covered_from.date()}..{covered_to.date()}: "
        f"package-only={sorted(str(pd.Timestamp(d).date()) for d in from_package - extract_holidays)} "
        f"warehouse-only={sorted(str(pd.Timestamp(d).date()) for d in extract_holidays - from_package)}. "
        f"Do not trust it for future dates until this matches."
    )
    import holidays as _holidays_pkg
    print(f"  [pass] holidays=={_holidays_pkg.__version__} GB/{UK_HOLIDAY_SUBDIVISION} "
          f"reproduces all {len(extract_holidays)} bank holidays the warehouse flags over "
          f"{covered_from.date()}..{covered_to.date()}, exactly - so it is safe to use for "
          f"dates past {covered_to.date()}, which is the only range it is used for")
    horizon_start = covered_to + pd.Timedelta(days=1)
    horizon_end = covered_to + pd.Timedelta(days=cfg.horizon)
    ahead = uk_bank_holidays_between(horizon_start, horizon_end)
    following = uk_bank_holidays_between(horizon_end + pd.Timedelta(days=1),
                                         horizon_end + pd.Timedelta(days=200))
    print(f"  [info] next {cfg.horizon}-day horizon {horizon_start.date()}"
          f"..{horizon_end.date()} contains {len(ahead)} bank holiday(s)"
          + (f": {[str(d.date()) for d in ahead]}" if len(ahead) else "")
          + (f"; next after that: {[str(d.date()) for d in following[:3]]}"
             if len(following) else ""))

    # 2. QuantileLoss really is evaluated in the target's own units.
    #    TemporalFusionTransformer.forward calls transform_output(...) -> the loss's
    #    rescale_parameters(...) -> TorchNormalizer.__call__, which un-scales AND
    #    inverse-transforms before BaseModel.step evaluates self.loss(prediction, y).
    #    Reproduce that with the exact custom transformation the original model used.
    custom_log1p = dict(forward=torch.log1p, reverse=torch.expm1,
                        inverse_torch=Expm1Transform())
    normaliser = TorchNormalizer(method="identity", center=True, transformation=custom_log1p)
    round_tripped = normaliser(dict(
        prediction=torch.tensor([[float(np.log1p(100.0))]]),
        target_scale=torch.tensor([[0.0, 1.0]]),
    ))
    assert abs(float(round_tripped) - 100.0) < 1e-2, float(round_tripped)
    print("  [pass] a normaliser 'transformation' is inverted before the loss sees the "
          "prediction -> QuantileLoss on a log1p GroupNormalizer is pinball loss in GBP")
    # TRANSFORMATIONS is TorchNormalizer's registry of named transforms: pass
    # transformation="log1p" and it looks the name up here, in preprocess() on the way in and
    # inverse_preprocess() on the way out.  Worth knowing that the built-in entry's reverse
    # is exp, not expm1, so it returns revenue + 1 - which is why this script log1p-s the
    # target column itself instead of naming a built-in transform.
    builtin_log1p = TorchNormalizer.TRANSFORMATIONS["log1p"]
    assert builtin_log1p["reverse"] is torch.exp, builtin_log1p["reverse"]
    print("  [pass] note: the built-in transformation=\"log1p\" reverses with exp(), not "
          "expm1() - it returns revenue + 1")

    # 3. The scale floor bites.  Two synthetic groups: "a" is constant (std 0, the
    #    degenerate case), "b" varies normally.  In the real pipeline the only group is
    #    merchant_id; "g" here is just a stand-in column name.
    floored = FlooredGroupNormalizer(method="standard", groups=["g"], center=True,
                                     scale_floor=0.25)
    group_frame = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10})
    values = pd.Series([5.0] * 10 + list(np.linspace(0, 10, 10)))
    floored.fit(values, group_frame)
    assert float(floored.norm_.loc["a", "scale"]) >= 0.25 - 1e-9, floored.norm_
    unfloored = GroupNormalizer(method="standard", groups=["g"], center=True).fit(
        values, group_frame
    )
    print(f"  [pass] scale floor: constant group's scale "
          f"{float(unfloored.norm_.loc['a', 'scale']):.2e} -> "
          f"{float(floored.norm_.loc['a', 'scale']):.2f}")

    # 4. Quantile-function integration recovers a known mean.
    levels = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    uniform_quantiles = (np.asarray(levels, dtype=np.float64) * 100.0).reshape(1, -1)
    estimated_mean = float(
        np.asarray(expected_from_quantiles(uniform_quantiles, levels)).ravel()[0]
    )
    assert abs(estimated_mean - 50.0) < 6.0, estimated_mean
    print(f"  [pass] expected_from_quantiles(uniform 0..100) = {estimated_mean:.1f} "
          f"(true mean 50)")
    # ...and on a lognormal, which is the shape the predictive distribution actually has once
    # log-space quantiles are exponentiated.  Reported rather than asserted tightly, because
    # holding the tails flat below q02 and above q98 makes the estimate deliberately
    # CONSERVATIVE - it cannot see the part of the mean that lives in the top 2%, and for a
    # heavy right tail that is a real fraction of it.  The size of that gap is the number to
    # know before adding these forecasts into a company total.
    from scipy.stats import norm as _norm
    for sigma in (0.6, 1.2, 2.0):
        true_mean = float(np.exp(0.5 * sigma ** 2))          # lognormal(mu=0, sigma)
        quantile_values = np.exp(sigma * _norm.ppf(levels)).reshape(1, -1)
        estimate = float(np.asarray(
            expected_from_quantiles(quantile_values, levels)
        ).ravel()[0])
        print(f"  [info] expected_from_quantiles(lognormal sigma={sigma}) = "
              f"{estimate:.3f} vs true {true_mean:.3f} "
              f"({estimate / true_mean - 1:+.1%}, conservative by construction)")

    df = prepare_data(cfg)

    # 5. log1p really does stop the whales owning the encoder inputs.  This is the claim in
    #    engineer_features, measured rather than asserted: for each globally-scaled column,
    #    what share of the total sum-of-squares belongs to the 14 largest merchants, and how
    #    much resolution is left inside the smallest size band.
    from sklearn.preprocessing import StandardScaler

    merchant_level = df.groupby(cfg.group_col, observed=True)[cfg.raw_target].mean()
    band = pd.cut(df[cfg.group_col].map(merchant_level),
                  [0, 100, 1_000, 10_000, 100_000, np.inf])
    smallest_band = band == band.cat.categories[0]
    whale_band = band == band.cat.categories[-1]
    print(f"  [info] encoder-input scaling check "
          f"({int(whale_band.sum()):,} whale rows = {whale_band.mean():.2%} of the frame)")
    print(f"    {'column':<26}{'whale share of SS':>19}{'sd within <100/day band':>26}")
    for col, log_col in [("daily_successful_transaction_count", "log_success_count"),
                         ("daily_unsuccessful_amount_gbp", "log_unsuccessful_gbp"),
                         (cfg.raw_target, cfg.target)]:
        row = []
        for name in (col, log_col):
            z = StandardScaler().fit_transform(
                df[[name]].to_numpy(dtype=np.float64)
            ).ravel()
            row.append((float((z[whale_band] ** 2).sum() / (z ** 2).sum()),
                        float(z[smallest_band].std())))
        (raw_share, raw_sd), (log_share, log_sd) = row
        assert log_share < raw_share, f"{col}: log1p did not reduce the whale share"
        assert log_sd > raw_sd * 5, f"{col}: log1p did not restore small-merchant resolution"
        print(f"    {col[:26]:<26}{raw_share:>8.1%} -> {log_share:<8.1%}"
              f"{raw_sd:>12.4f} -> {log_sd:<.4f}")
    print("    (all pass: whale share of signal falls, small-merchant resolution rises >5x)")

    # 6. Future frame carries correct, non-forward-filled calendar covariates.  This is the
    #    test that would have caught the previous model's biggest prediction-path bug.
    merchant = cfg.forecast_merchant
    if merchant not in set(df[cfg.group_col]):
        merchant = df[cfg.group_col].iloc[-1]
        print(f"  [note] merchant {cfg.forecast_merchant} filtered out; using {merchant}")
    horizon_rows = make_future_frame(df, merchant, cfg).tail(cfg.horizon)
    derived_weekday = (horizon_rows["day_of_week"].astype(int)
                       .map(dict(enumerate(_WEEKDAYS, start=1))).to_numpy())
    true_weekday = pd.to_datetime(horizon_rows[cfg.date_col]).dt.day_name().to_numpy()
    assert (derived_weekday == true_weekday).all(), list(zip(derived_weekday, true_weekday))
    assert horizon_rows["day_of_week"].nunique() == 7, "horizon must cycle all 7 weekdays"
    assert horizon_rows["day_type"].nunique() >= 2, "horizon must contain weekdays and weekends"
    print(f"  [pass] future horizon for merchant {merchant}: every day's weekday label "
          f"matches the real calendar, {horizon_rows['day_type'].nunique()} distinct "
          f"day_types, "
          f"{int(horizon_rows['is_uk_bank_holiday'].astype(str).eq('True').sum())} "
          f"bank holidays")

    # 6b. The all-merchant horizon is row-for-row the single-merchant horizon.  The horizon
    #     rows ARE the decoder input, so if the batched path built them even slightly
    #     differently - a forward-filled ratio, a date off by one, a categorical rendered
    #     "5.0" instead of "5" and therefore mapped to the unknown token - forecast_merchant
    #     and forecast_all_merchants would disagree for the same merchant and nothing else in
    #     the run would notice.  Sampled at the three ends of the population: the merchant
    #     reported on stdout, the stalest, and the shortest-history one.
    every_horizon = append_future_rows(df, cfg, cfg.horizon,
                                       ratio_names=df.attrs.get("ratio_names", []))
    last_t = df.groupby(cfg.group_col, observed=True)["time_idx"].max()
    grid_days = df.groupby(cfg.group_col, observed=True).size()
    sampled = [m for m in dict.fromkeys([str(merchant), str(last_t.idxmin()),
                                         str(grid_days.idxmin())])
               if m in set(df[cfg.group_col])]
    for m in sampled:
        one = make_future_frame(df, m, cfg).reset_index(drop=True)
        many = every_horizon[every_horizon[cfg.group_col] == m].reset_index(drop=True)
        pd.testing.assert_frame_equal(one, many)
    print(f"  [pass] append_future_rows over all {df[cfg.group_col].nunique()} merchants is "
          f"row-for-row identical to make_future_frame for {sampled} (staleness "
          f"{[int(int(df['time_idx'].max()) - int(last_t[m])) for m in sampled]} days, "
          f"history {[int(grid_days[m]) for m in sampled]} days)")

    # ...and every merchant can actually be forecast.  With `horizon` rows appended, a
    # predict-mode window needs min_prediction_length + min_encoder_length rows and
    # min_prediction_length IS the horizon, so the horizon cancels and the requirement on
    # observed history is min_encoder_length - NOT min_encoder_length + horizon, which is
    # only what a TRAINING window needs.  The two are different numbers and the gap between
    # them is a real population: merchants that get a forecast but never contributed a
    # gradient step.  forecast_summary.csv reports them per merchant as had_training_windows.
    too_short = grid_days[grid_days < cfg.min_encoder_length]
    assert too_short.empty, (
        f"{len(too_short)} merchant(s) survived filter_series with fewer than "
        f"min_encoder_length={cfg.min_encoder_length} days and would silently get no "
        f"forecast csv: {dict(too_short.head())}"
    )
    untrainable = grid_days[grid_days < cfg.min_encoder_length + cfg.horizon]
    print(f"  [pass] all {len(grid_days)} merchants have >= min_encoder_length="
          f"{cfg.min_encoder_length} days, so all of them get a forecast; "
          f"{len(untrainable)} of them have < {cfg.min_encoder_length + cfg.horizon} days and "
          f"so produce no TRAINING window at any cutoff"
          + (f" ({list(untrainable.index[:8])})" if len(untrainable) else ""))

    # 7. Datasets build, and the group normaliser is genuinely per merchant.
    split = build_datasets(df, cfg)
    target_normaliser = split.training.target_normalizer
    assert getattr(target_normaliser, "_groups", []) == [cfg.group_col], (
        f"normaliser groups={target_normaliser._groups!r} - an empty list means it is "
        f"global, not per merchant"
    )
    assert isinstance(target_normaliser.norm_, pd.DataFrame) and len(target_normaliser.norm_) > 1
    # norm_ is indexed by NaNLabelEncoder code, not by merchant id, because the normaliser is
    # fitted on the already-encoded group column.
    print(f"  [pass] target normaliser fitted per merchant: "
          f"{len(target_normaliser.norm_)} groups, centre range "
          f"[{target_normaliser.norm_['center'].min():.2f}, "
          f"{target_normaliser.norm_['center'].max():.2f}], scale range "
          f"[{target_normaliser.norm_['scale'].min():.2f}, "
          f"{target_normaliser.norm_['scale'].max():.2f}]")
    # ...and the property that actually matters: a real batch must carry a DIFFERENT
    # (centre, scale) per merchant.  A global normaliser would give every row the same pair,
    # which is what the previous model had and what no summary statistic above would reveal.
    batch_x, _ = next(iter(split.training.to_dataloader(train=True, batch_size=256,
                                                        num_workers=0)))
    batch_scale = batch_x["target_scale"].numpy()
    n_merchants = len(set(batch_x["groups"][:, 0].tolist()))
    n_scales = len({tuple(row) for row in batch_scale})
    assert n_scales > 1, (
        "every row in the batch shares one (centre, scale) - the normaliser is GLOBAL, not "
        "per merchant"
    )
    print(f"  [pass] a real batch of {len(batch_scale)} windows carries {n_scales} distinct "
          f"(centre, scale) pairs across {n_merchants} distinct merchants "
          f"(centre {batch_scale[:, 0].min():.2f}..{batch_scale[:, 0].max():.2f}, "
          f"scale {batch_scale[:, 1].min():.2f}..{batch_scale[:, 1].max():.2f})")
    assert len(split.validation) > 0 and len(split.holdout) > 0
    print("\nSELF-TEST PASSED")


# ======================================================================================
# 13. main
# ======================================================================================
def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options.

    ``parser`` is argparse's builder object: each add_argument() registers one option, and
    parse_args() then reads sys.argv and returns them as attributes.  The ``--`` prefix is
    argparse's convention for an OPTIONAL, named flag - ``--hidden-size 96`` can appear in
    any order or be left out entirely, whereas a name without dashes would be a positional
    argument that must always be supplied in a fixed slot.  Every option below has a
    default taken from Config, so plain `python3 tft_revenue_v2.py` runs the shipped
    configuration and each flag overrides exactly one field of it.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=Config.data_path)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--train-batches-per-epoch", type=int, default=Config.train_batches_per_epoch)
    parser.add_argument("--hidden-size", type=int, default=Config.hidden_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--max-encoder-length", type=int, default=Config.max_encoder_length)
    parser.add_argument("--forecast-merchant", default=Config.forecast_merchant,
                        help="the one merchant reported in full on stdout, with a duplicate "
                             "csv and png at the top level of outdir. 'none' skips that "
                             "report.  It does not decide which merchants are forecast - "
                             "every merchant is, see --no-forecast-all")
    parser.add_argument("--no-forecast-all", action="store_true",
                        help=f"skip the all-merchant pass, i.e. do not write "
                             f"{Config.outdir}/{Config.forecast_dir}/ or forecast_summary.csv")
    parser.add_argument("--forecast-plots", type=int, default=Config.forecast_plots,
                        metavar="N",
                        help="pngs written by the all-merchant pass: -1 (default) every "
                             "merchant, 0 none, N the N largest by mean daily revenue. "
                             "Measured here: 384 charts = 50s and 58 MB, against 4s for the "
                             "forecasts themselves")
    parser.add_argument("--checkpoint", default=None, help="reuse an existing checkpoint")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--refit-full", action="store_true",
                   help="after model selection, refit on every day of data for the "
                        "production forecast (this is what brings short-history "
                        "merchants such as 171 into the training set)")
    parser.add_argument("--find-lr", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    parser.add_argument("--show", action="store_true",
                        help="display figures as well as saving them")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")

    cfg = Config(
        data_path=args.data,
        outdir=args.outdir,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_batches_per_epoch=args.train_batches_per_epoch,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        horizon=args.horizon,
        max_encoder_length=args.max_encoder_length,
        forecast_merchant=str(args.forecast_merchant),
        forecast_plots=args.forecast_plots,
    )
    if args.smoke:
        cfg.max_epochs = 2
        cfg.train_batches_per_epoch = 30
        cfg.hidden_size = 32
        cfg.hidden_continuous_size = 16
        cfg.lstm_layers = 1
        cfg.n_example_plots = 2
        cfg.patience = 99
        print("[smoke] tiny configuration - not a trained model")

    if args.selftest:
        selftest(cfg)
        return

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.seed, workers=True)
    acc, precision = pick_device(args.accelerator)

    df = prepare_data(cfg)
    split = build_datasets(df, cfg)
    # Yes - one small JSON file recording every hyperparameter and the exact feature lists
    # this run used.  It exists because the previous project has 80 hparams.yaml files that
    # cannot be told apart, so "which settings produced this checkpoint" was unanswerable
    # after the fact.  Written before training so it survives a crash.
    (outdir / "config.json").write_text(
        json.dumps({"config": cfg.as_dict(), "features": split.feature_spec}, indent=2)
    )

    # Baseline in the model's own loss geometry, for context.
    val_dl = split.validation.to_dataloader(train=False, batch_size=cfg.batch_size * 2,
                                            num_workers=0)
    bl = Baseline().predict(val_dl, return_y=True, trainer_kwargs=trainer_kwargs(acc))
    y_true = bl.y[0] if isinstance(bl.y, tuple | list) else bl.y
    bl_mae_log = float((bl.output.detach().cpu() - y_true.detach().cpu()).abs().mean())
    print(f"[baseline] last-value MAE on log1p(revenue) = {bl_mae_log:.4f}")

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
        best = args.checkpoint
        print(f"[model] loaded {best}")
    else:
        model = build_model(split, cfg)
        best = None

    # Nothing to do with the TFT's internal skip connections.  --skip-train is a CLI
    # convenience: combined with --checkpoint it loads an already-trained model and jumps
    # straight to evaluation, plots and the forecast, so you can re-plot or re-forecast
    # without paying for training again.
    if not args.skip_train:
        if args.find_lr:
            from lightning.pytorch.tuner import Tuner
            tuner = Tuner(pl.Trainer(accelerator=acc, devices=1, precision=precision,
                                     gradient_clip_val=cfg.gradient_clip_val, logger=False))
            res = tuner.lr_find(
                model,
                train_dataloaders=split.training.to_dataloader(
                    train=True, batch_size=cfg.batch_size, num_workers=0),
                val_dataloaders=val_dl, min_lr=1e-5, max_lr=1e-1,
            )
            # res.suggestion() is Lightning's pick from the lr_find sweep: it runs a short
            # burst at exponentially increasing learning rates and returns the point of
            # STEEPEST LOSS DESCENT (roughly the minimum of dloss/dlog(lr)).  It does not
            # protect against local minima or plateaus, and it cannot - it is a single
            # forward sweep over ~100 steps of one batch order, so it can land on a
            # coincidental cliff or, if the loss is flat early, return an absurdly large
            # value that diverges a few epochs in.  Two guards:
            #   1. clamp it to a range that is known to be trainable for this loss, so a
            #      bad sweep degrades to a sane number instead of wrecking the run;
            #   2. plateau escape is the SCHEDULER's job, not lr_find's - hence
            #      ReduceLROnPlateau (reduce_on_plateau_patience=3), which cuts the rate
            #      when val_loss stops improving.  That is exactly what every run in
            #      lightning_logs lacked, with patience left at 1000.
            LR_MIN, LR_MAX = 1e-4, 1e-2
            suggested = float(res.suggestion())
            chosen = float(np.clip(suggested, LR_MIN, LR_MAX))
            if chosen != suggested:
                print(f"[lr_find] suggestion {suggested:.5g} is outside "
                      f"[{LR_MIN:g}, {LR_MAX:g}] - clamped to {chosen:.5g}")
            else:
                print(f"[lr_find] suggestion = {chosen:.5g} (within the trainable range)")
            model.hparams.learning_rate = chosen
            cfg.learning_rate = chosen

        trainer, best = train(model, split, cfg, acc, precision, outdir, "select")
        model = load_checkpoint(best)
        best_epoch = trainer.early_stopping_callback.stopped_epoch or trainer.current_epoch
    else:
        best_epoch = cfg.max_epochs

    metrics = evaluate(model, split, df, cfg, acc, outdir)
    try:
        make_plots(model, split, cfg, acc, outdir)
    except Exception as exc:  # plotting must never lose a trained model
        print(f"[plots] skipped: {type(exc).__name__}: {exc}")

    if args.refit_full:
        print("\n" + "=" * 88)
        # The epoch count is TAKEN FROM the early-stopped selection run and not re-derived,
        # which is what keeps this from being a licence to overfit.  Epoch count is the only
        # hyperparameter that could still be tuned against the validation block once that
        # block is inside the training data, so it is fixed first, on data the refit model
        # never sees the answer to, and then simply reused.  Every other hyperparameter was
        # already fixed before either run started.  Nothing after this point may consult a
        # metric to decide when to stop - hence early_stop=False rather than a larger
        # patience.
        n = max(3, int(best_epoch) + 1)
        if args.skip_train:
            print(f"[refit] WARNING: --skip-train means no selection run happened, so there "
                  f"is no early-stopped epoch count to reuse; falling back to "
                  f"--max-epochs={cfg.max_epochs}.  Pass --max-epochs explicitly to match "
                  f"the epoch the loaded checkpoint actually stopped at, or the refit will "
                  f"train for a different length than the model you evaluated.")
        print(f"[refit] retraining from scratch on all {len(df):,} rows for {n} epochs "
              f"(the selection run's stopping epoch; no early stopping, because validation "
              f"is now inside the training data)")
        print(f"[refit] the holdout metrics above stand - they belong to the SELECTION model "
              f"and are the only out-of-sample numbers in this run.  The refit model that "
              f"produces the forecast below has seen every day of data, so it cannot be "
              f"scored on any of it.")
        print("=" * 88)
        full = build_datasets(df, cfg, full=True)
        model = build_model(full, cfg)
        _, best = train(model, full, cfg, acc, precision, outdir, "full",
                        max_epochs=n, early_stop=False)
        model = load_checkpoint(best)
        split = full
        metrics["refit_full"] = dict(
            epochs=n,
            note="forecast comes from a model refit on all data; holdout metrics in this "
                 "file belong to the selection model, which never saw the holdout block",
        )

    # The all-merchant pass runs FIRST because it is the deliverable: a bad
    # --forecast-merchant (an id that filter_series dropped) raises SystemExit, and it must
    # not be able to take 384 forecasts down with it.  The single-merchant report is then
    # wrapped for the same reason in reverse - same rule as make_plots above.
    if not args.no_forecast_all:
        # split is `full` here if --refit-full ran, so train_cutoff is the cutoff the model
        # that produces these forecasts was actually trained to.
        summary = forecast_all_merchants(model, df, cfg, acc, outdir,
                                         plots=cfg.forecast_plots,
                                         train_cutoff=split.train_cutoff)
        metrics["forecast_all"] = dict(
            merchants=int(len(summary)),
            horizon=cfg.horizon,
            directory=str((outdir / cfg.forecast_dir).resolve()),
            horizon_total_median_gbp=float(summary["horizon_total_median_gbp"].sum()),
            horizon_total_expected_gbp=float(summary["horizon_total_expected_gbp"].sum()),
            current_merchants=int((summary["days_since_last_observation"] <= 1).sum()),
            stale_over_30_days=int((summary["days_since_last_observation"] > 30).sum()),
            cold_start=int(summary["cold_start"].fillna(False).sum()),
            failed_sanity_check=int(
                ((summary["level_ratio"] < 0.2) | (summary["level_ratio"] > 5)
                 | (summary["dispersion_ratio"] > 6)).sum()
            ),
        )

    if cfg.forecast_merchant.lower() in ("none", ""):
        print("[forecast] no single-merchant report (--forecast-merchant none)")
    else:
        try:
            fc = forecast_merchant(model, df, cfg, acc, outdir, cfg.forecast_merchant)
            metrics["forecast_summary"] = dict(
                merchant=cfg.forecast_merchant,
                horizon_total_median_gbp=float(fc["q50"].sum()),
                horizon_total_expected_gbp=float(fc["expected_gbp"].sum()),
            )
        except SystemExit as exc:  # e.g. filter_series dropped this merchant
            print(f"[forecast] single-merchant report for {cfg.forecast_merchant!r} skipped: "
                  f"{exc}" + ("" if args.no_forecast_all else
                              f"  (its forecast, if it has one, is still in "
                              f"{outdir / cfg.forecast_dir}/)"))
            metrics["forecast_summary"] = dict(merchant=cfg.forecast_merchant, error=str(exc))

    metrics["baseline_last_value_MAE_log"] = bl_mae_log
    metrics["best_checkpoint"] = best
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\n[done] artefacts in {outdir.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    data = df.copy()

    # --- target ---------------------------------------------------------------------
    tpv = pd.to_numeric(data[cfg.raw_target], errors="coerce").fillna(0.0).clip(lower=0.0)
    data[cfg.raw_target] = tpv.astype(np.float64)
    data[cfg.target] = np.log1p(data[cfg.raw_target].to_numpy()).astype(np.float32)

    # --- time index -----------------------------------------------------------------
    # time_idx counts days from the GLOBAL earliest date in the frame (2024-01-01 -> 0), not
    # from each merchant's own first day, so a given time_idx is the same calendar day for
    # every merchant.  That is what lets train/val/holdout be cut at shared indices.
    data["time_idx"] = ((data[cfg.date_col] - data[cfg.date_col].min()).dt.days).astype(np.int64)
    min_date = data[cfg.date_col].min()
    # ...and series_start_idx records where each merchant's own series begins on that shared
    # axis (merchant 171 -> 871).  It is a cohort marker: nothing is renumbered to 0, and an
    # encoder window starting at day 778 keeps time_idx 778.
    data["series_start_idx"] = (
        data.groupby(cfg.group_col, observed=True)["time_idx"].transform("min")
    )

    # --- calendar: the warehouse's own columns, extrapolated only past its last date --
    data = add_calendar_features(data, cfg.date_col)

    # --- merchant age ---------------------------------------------------------------
    # log1p here is not about skew (age is roughly uniform, 0..3843 days).  It is about
    # what the feature is FOR: the difference between day 10 and day 100 of a merchant's
    # life is a real behavioural change - onboarding, ramp-up - while the difference
    # between day 3000 and day 3090 is nothing at all.  On a linear scale those two gaps
    # are identical to the network; on a log scale the first is 2.3 units and the second
    # 0.03.  Both raw and log forms are kept: since_merchant_creation feeds the
    # per-merchant static, log_merchant_age is the known real the decoder reads.
    if "merchant_created_datetime" in data.columns:
        age_days = (data[cfg.date_col] - data["merchant_created_datetime"]).dt.days
        age_days = age_days.fillna(0).clip(lower=0)
    else:  # pragma: no cover
        age_days = pd.Series(0, index=data.index)
    data["since_merchant_creation"] = age_days.astype(np.float32)
    data["log_merchant_age"] = np.log1p(age_days.to_numpy()).astype(np.float32)
    age_at_first_sale = (
        data.groupby(cfg.group_col, observed=True)["since_merchant_creation"].transform("min")
    )
    data["log_tenure_at_series_start"] = np.log1p(
        age_at_first_sale.to_numpy()
    ).astype(np.float32)

    # --- heavy-tailed observed reals -> log1p ---------------------------------------
    # These six columns are scaled by ONE GLOBAL sklearn StandardScaler, because
    # TimeSeriesDataSet does `self._scalers[name] = StandardScaler().fit(data[[name]])`
    # per column with no notion of groups (_timeseries.py:1256-1262) and build_datasets
    # never overrides it.  Measured on the prepared frame, that is ruinous on the raw
    # scale: 94-98% of all rows land inside the BOTTOM 1% of each column's z range,
    # interquartile range over full range is 3.4e-4 to 1.2e-3, and the 4.67% of rows
    # belonging to 14 whale merchants carry 77-97% of the total sum of squares.  For a
    # merchant averaging GBP 482/day, a GBP 20 day and a GBP 900 day - a 45x move - differ
    # by z = 0.0065, which is 0.0058% of the input range and 1/8300th of the gap between
    # two ordinary days of merchant 171.
    #
    # After log1p the same 45x move is z = 1.02, statistically the same size (0.97x) as a
    # whale's 38x move; every size band gets 0.55-0.80 sd of internal resolution instead of
    # 0.0014/0.0063/0.045 for the three smallest; the whale share of sum-of-squares falls to
    # 7-17%; and Spearman(scaled, raw) = 1.000000, so no ordering information is lost.
    # `--selftest` re-measures this and asserts the key thresholds, so the claim is checked
    # rather than trusted.  Caveat: fraud and dispute counts are 91.6%/90.3% zeros, so they
    # stay near-indicators either way (z(max) still +10.7/+10.3); log1p is doing the real
    # work on the dense columns.  NOTE this argument does NOT cover the target - that is
    # scaled per merchant by FlooredGroupNormalizer, and its log1p is justified separately
    # by the loss geometry (MODEL_NOTES.md 2.1).
    txn_count = data["daily_transaction_amount"].to_numpy()
    success_count = data["daily_successful_transaction_count"].to_numpy()
    # log_txn_count dropped as a model input: it scored 0.60% encoder importance in v1 and
    # is near-collinear with log_success_count.  The column is still read here because
    # had_no_attempts needs it, and that flag is the part that carries signal.
    data["log_success_count"] = np.log1p(success_count).astype(np.float32)
    data["log_unsuccessful_gbp"] = np.log1p(
        data["daily_unsuccessful_amount_gbp"].to_numpy()
    ).astype(np.float32)
    data["log_fraud_count"] = np.log1p(data["daily_fraud_amount"].to_numpy()).astype(np.float32)
    data["log_dispute_count"] = np.log1p(
        data["daily_dispute_amount"].to_numpy()
    ).astype(np.float32)
    # Revenue = transactions x average ticket, and the two move for different reasons: a
    # merchant can hold volume while its basket shrinks.  Giving the encoder the
    # decomposition saves it from having to infer the ratio.
    data["log_avg_ticket"] = np.log1p(
        safe_divide(data[cfg.raw_target].to_numpy(), np.maximum(success_count, 0))
    ).astype(np.float32)
    # is_zero_day: earned nothing.  had_no_attempts: nobody even tried.  Different events -
    # the first can mean every transaction was declined, the second means the merchant was
    # dormant - and with 20% of grid days at zero revenue the distinction matters.
    data["is_zero_day"] = (data[cfg.raw_target].to_numpy() <= 0).astype(np.float32)
    data["had_no_attempts"] = (txn_count <= 0).astype(np.float32)

    # --- mix ratios (bounded in [0, 1], so safe for a global standard scaler) --------
    # Every "*_amount" column is a COUNT, and within each group they sum exactly to
    # daily_transaction_amount, so a share-of-the-day ratio is scale-free across merchants.
    ratio_names: list[str] = []
    for source_cols, group_name in ((ACQUIRER_COLS, "acquirer"),
                                    (PAYMENT_COLS, "payment type"),
                                    (RESPONSE_COLS, "response status"),
                                    (INTEGRATION_COLS, "integration")):
        present = [c for c in source_cols if c in data.columns]
        if not present:
            print(f"[features] no {group_name} columns in this extract - "
                  f"skipping that ratio group (expected {source_cols})")
            continue
        daily_total = data[present].to_numpy(dtype=np.float64).sum(axis=1)
        for col in present:
            ratio_col = f"{col}_ratio"
            data[ratio_col] = safe_divide(data[col].to_numpy(), daily_total).astype(np.float32)
            ratio_names.append(ratio_col)
    data.attrs["ratio_names"] = ratio_names

    # --- categoricals as strings ----------------------------------------------------
    # NaNLabelEncoder maps by value, so plain strings are all it needs; a pandas
    # `category` dtype would additionally carry per-frame category codes that differ
    # between the training frame and a single-merchant forecast frame.
    categorical_cols = [
        "day_of_week", "day_of_month", "calendar_month", "day_type",
        "is_uk_bank_holiday", "merchant_category", "sector_group",
        "iso_name"
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)

    return data
