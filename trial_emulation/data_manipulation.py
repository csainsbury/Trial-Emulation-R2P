"""
Data manipulation function.

Pre-processes data for weight calculation and extension.
Maps R's data_manipulation() function.
"""

from typing import Optional
import warnings

import pandas as pd
import numpy as np

try:
    from ._censor_func import censor_func as _censor_func_cpp

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

    # Placeholder function - actual fallback is _censor_func_python() below
    def _censor_func_cpp(first, eligible, treatment, switch_flag):
        raise NotImplementedError("C++ module not available, using Python fallback")


def data_manipulation(data: pd.DataFrame, use_censor: bool = True) -> pd.DataFrame:
    """
    Data manipulation function.

    Takes the data and performs preprocessing for weight calculation and extension.

    Args:
        data: DataFrame to pre-process
        use_censor: Apply censoring due to treatment switch?

    Returns:
        Preprocessed DataFrame
    """
    # Input validation
    if data.empty:
        raise ValueError("Input data is empty")

    required_cols = ["id", "period", "treatment", "outcome", "eligible"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = data.copy()
    ids = data["id"]

    # Remove observations before the first eligible period (R: period >=
    # .SD[eligible == 1, min(period, Inf)] by id). Patients never eligible are
    # removed entirely.
    first_eligible = data["period"].where(data["eligible"] == 1).groupby(ids).transform("min")
    after_eligibility = data["period"] >= first_eligible.fillna(np.inf)
    if not after_eligibility.all():
        warnings.warn("Observations before trial eligibility were removed", UserWarning)
        data = data[after_eligibility].copy()

    # Remove observations after the first outcome event.
    first_event = data["period"].where(data["outcome"] == 1).groupby(data["id"]).transform("min")
    after_event = data["period"] > first_event.fillna(np.inf)
    if after_event.any():
        warnings.warn("Observations after the outcome occurred were removed", UserWarning)
        data = data[~after_event].copy()

    # Event time from each patient's last row (R: .SD[.N, list(period, outcome)]).
    last_rows = data.groupby("id", sort=False).tail(1)
    event_data = pd.DataFrame(
        {
            "id": last_rows["id"].to_numpy(),
            "time_of_event": np.where(
                last_rows["outcome"].notna() & (last_rows["outcome"] == 1),
                last_rows["period"].astype(float),
                9999.0,
            ),
        }
    )

    sw_data = data.merge(event_data, on="id", how="left")

    by_id = sw_data.groupby("id", sort=False)
    sw_data["first"] = ~sw_data["id"].duplicated()

    # Lagged treatment within patient; 0 on the first row.
    sw_data["am_1"] = by_id["treatment"].shift(1)
    sw_data.loc[sw_data["first"], "am_1"] = 0

    # Treatment switch indicator.
    sw_data["switch"] = np.where(
        sw_data["first"], 0, (sw_data["am_1"] != sw_data["treatment"]).astype(int)
    )

    # Regime start and time on the current regime.
    regime_start = sw_data["period"].where(sw_data["first"] | (sw_data["switch"] == 1))
    sw_data["regime_start"] = regime_start.groupby(sw_data["id"], sort=False).ffill()
    regime_start_shift = sw_data.groupby("id", sort=False)["regime_start"].shift(1)
    sw_data["time_on_regime"] = np.where(
        sw_data["first"], 0.0, sw_data["period"] - regime_start_shift
    )

    # Cumulative treatment received up to and including this period.
    sw_data["cumA"] = sw_data.groupby("id", sort=False)["treatment"].cumsum()

    # Handle censoring state machine
    if use_censor:
        sw_data["started0"] = np.nan
        sw_data["started1"] = np.nan
        sw_data["stop0"] = np.nan
        sw_data["stop1"] = np.nan
        sw_data["eligible0_sw"] = np.nan
        sw_data["eligible1_sw"] = np.nan
        sw_data["delete"] = np.nan

        # Use C++ function if available, otherwise Python fallback
        if HAS_CPP:
            try:
                result = _censor_func_cpp(
                    sw_data["first"].astype(int).values,
                    sw_data["eligible"].astype(int).values,
                    sw_data["treatment"].astype(int).values,
                    sw_data["switch"].astype(int).values,
                )
                sw_data["started0"] = result["started0"]
                sw_data["started1"] = result["started1"]
                sw_data["stop0"] = result["stop0"]
                sw_data["stop1"] = result["stop1"]
                sw_data["eligible0_sw"] = result["eligible0_sw"]
                sw_data["eligible1_sw"] = result["eligible1_sw"]
                sw_data["delete"] = result["delete"]
            except Exception as e:
                warnings.warn(
                    f"C++ censoring function failed, using Python fallback: {e}", RuntimeWarning
                )
                sw_data = _censor_func_python(sw_data)
        else:
            sw_data = _censor_func_python(sw_data)

        # Filter rows with delete == False
        sw_data = sw_data[sw_data["delete"] == False].copy()

        # Drop helper columns
        sw_data = sw_data.drop(
            columns=[
                "delete",
                "eligible0_sw",
                "eligible1_sw",
                "started0",
                "started1",
                "stop0",
                "stop1",
            ]
        )

    # Compute eligible0 and eligible1
    sw_data["eligible0"] = 0
    sw_data["eligible1"] = 0
    sw_data.loc[sw_data["am_1"] == 0, "eligible0"] = 1
    sw_data.loc[sw_data["am_1"] == 1, "eligible1"] = 1

    # Key by id (R: setkeyv(sw_data, "id"), a stable sort).
    sw_data = sw_data.sort_values("id", kind="stable").reset_index(drop=True)

    return sw_data


def _censor_func_python(sw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Python fallback implementation of censoring function.

    Implements the same logic as the C++ version (see cpp/censor_func.cpp).
    This function is used when the C++ extension is not available or fails.
    It performs artificial censoring based on treatment switches for per-protocol analysis.

    The implementation mirrors the C++ state machine logic exactly:
    - Tracks started0/started1 flags for treatment regimes
    - Tracks stop0/stop1 flags when switching while ineligible
    - Sets eligible0_sw/eligible1_sw based on current regime
    - Marks rows for deletion when neither eligible flag is set
    """
    sw_data = sw_data.copy()
    n = len(sw_data)

    started0_ = 0
    started1_ = 0
    stop0_ = 0
    stop1_ = 0
    eligible0_sw_ = 0
    eligible1_sw_ = 0

    started0 = np.zeros(n, dtype=int)
    started1 = np.zeros(n, dtype=int)
    stop0 = np.zeros(n, dtype=int)
    stop1 = np.zeros(n, dtype=int)
    eligible0_sw = np.zeros(n, dtype=int)
    eligible1_sw = np.zeros(n, dtype=int)
    delete_flag = np.zeros(n, dtype=bool)

    first_v = sw_data["first"].to_numpy(dtype=bool)
    eligible_v = sw_data["eligible"].to_numpy()
    treatment_v = sw_data["treatment"].to_numpy()
    switch_v = sw_data["switch"].to_numpy()

    for i in range(n):
        if first_v[i]:
            started0_ = 0
            started1_ = 0
            stop0_ = 0
            stop1_ = 0
            eligible0_sw_ = 0
            eligible1_sw_ = 0

        if stop0_ == 1 or stop1_ == 1:
            started0_ = 0
            started1_ = 0
            stop0_ = 0
            stop1_ = 0
            eligible0_sw_ = 0
            eligible1_sw_ = 0

        if started0_ == 0 and started1_ == 0 and eligible_v[i] == 1:
            if treatment_v[i] == 0:
                started0_ = 1
            elif treatment_v[i] == 1:
                started1_ = 1

        if started0_ == 1 and stop0_ == 0:
            eligible0_sw_ = 1
            eligible1_sw_ = 0
        elif started1_ == 1 and stop1_ == 0:
            eligible0_sw_ = 0
            eligible1_sw_ = 1
        else:
            eligible0_sw_ = 0
            eligible1_sw_ = 0

        if switch_v[i] == 1:
            if eligible_v[i] == 1:
                if treatment_v[i] == 1:
                    started1_ = 1
                    stop1_ = 0
                    started0_ = 0
                    stop0_ = 0
                    eligible1_sw_ = 1
                elif treatment_v[i] == 0:
                    started0_ = 1
                    stop0_ = 0
                    started1_ = 0
                    stop1_ = 0
                    eligible0_sw_ = 1
            else:
                stop0_ = started0_
                stop1_ = started1_

        if eligible0_sw_ == 0 and eligible1_sw_ == 0:
            delete_flag[i] = True
        else:
            started0[i] = started0_
            started1[i] = started1_
            stop0[i] = stop0_
            stop1[i] = stop1_
            eligible1_sw[i] = eligible1_sw_
            eligible0_sw[i] = eligible0_sw_
            delete_flag[i] = False

    sw_data["started0"] = started0
    sw_data["started1"] = started1
    sw_data["stop0"] = stop0
    sw_data["stop1"] = stop1
    sw_data["eligible0_sw"] = eligible0_sw
    sw_data["eligible1_sw"] = eligible1_sw
    sw_data["delete"] = delete_flag

    return sw_data
