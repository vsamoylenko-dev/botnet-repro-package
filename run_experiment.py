import os
import sys
import json
import time
import random
import hashlib
import platform
import subprocess
import shutil
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CONFIG = {
    "seed": 42,
    "n_trials": 30,
    "n_blocks": 3,
    "forward_embargo_minutes": 1,
    "bench_batch_size": 100000,
    "bench_repeats": 30,
    "session_1_start": "10:11:00",
    "session_1_end": "10:50:00",
    "session_2_start": "11:00:00",
    "session_2_end": "11:34:00",
    "purge_minutes": 1,
    "feature_cols": [
        "Proto",
        "Duration",
        "TotPkts",
        "TotBytes",
        "Pkts_per_Sec",
        "Bytes_per_Sec",
        "Bytes_per_Pkt",
    ],
    "bot_label_token": "bot",
    "benign_label_token": "benign",
    "max_thu_fpr": 0.01,
    "max_oof_fpr": 0.01,
    "n_oof_blocks": 6,
    "thresh_grid_size": 400,
}

ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiment_results"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PAPER_FIG_DIR = OUTPUT_DIR / "paper_figures"

IDS_FRI_PATH = DATA_DIR / "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
IDS_THU_PATH = DATA_DIR / "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = CONFIG["feature_cols"]
BOT_LABEL_TOKEN = CONFIG["bot_label_token"]
BENIGN_LABEL_TOKEN = CONFIG["benign_label_token"]


def get_file_hash(filepath: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                "wmic cpu get name", shell=True
            ).decode(errors="ignore").split("\n")
            return out[1].strip() if len(out) > 1 else platform.processor()

        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()

        return platform.processor()
    except Exception:
        return platform.processor()


def get_ram_and_cores():
    ram_gb, phys_cores = None, None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 4)
        phys_cores = psutil.cpu_count(logical=False)
    except Exception:
        ram_gb = None
        phys_cores = None
    return ram_gb, phys_cores


def save_run_metadata():
    ram_gb, phys_cores = get_ram_and_cores()
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Python_Version": sys.version.split()[0],
        "XGBoost_Version": xgb.__version__,
        "Optuna_Version": optuna.__version__,
        "Sklearn_Version": sklearn.__version__,
        "CPU_Model": get_cpu_name(),
        "Architecture": platform.machine(),
        "Seed": CONFIG["seed"],
        "Friday_File": str(IDS_FRI_PATH),
        "Thursday_File": str(IDS_THU_PATH),
        "Friday_File_SHA256": get_file_hash(IDS_FRI_PATH),
        "Thursday_File_SHA256": get_file_hash(IDS_THU_PATH),
        "RAM_GB": ram_gb,
        "Physical_Cores": phys_cores,
    }
    pd.DataFrame([info]).to_csv(DIAG_DIR / "experimental_setup.csv", index=False)
    with open(DIAG_DIR / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)


def parse_timestamp_series(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")


def process_ids_botnet(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    if "Timestamp" not in df.columns:
        raise ValueError("No 'Timestamp' column found.")
    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found.")

    df["Timestamp_dt"] = parse_timestamp_series(df["Timestamp"])
    df = df[df["Timestamp_dt"].notna()].copy()

    df["Label_str"] = df["Label"].astype(str)
    df["Label_str_l"] = df["Label_str"].str.lower()
    df["Label"] = (df["Label_str_l"] == BOT_LABEL_TOKEN).astype(int)

    required_cols = [
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Duration",
        "Protocol",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    def to_num(colname: str) -> pd.Series:
        return pd.to_numeric(df[colname], errors="coerce")

    df["TotPkts"] = to_num("Tot Fwd Pkts") + to_num("Tot Bwd Pkts")
    df["TotBytes"] = to_num("TotLen Fwd Pkts") + to_num("TotLen Bwd Pkts")
    df["Duration"] = to_num("Flow Duration") / 1e6

    df = df.rename(columns={"Protocol": "Proto"})
    df["Proto"] = pd.to_numeric(df["Proto"], errors="coerce").fillna(0).astype(int)

    for col in ["TotPkts", "TotBytes", "Duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["TotPkts", "TotBytes", "Duration"]
    ).copy()

    df["Pkts_per_Sec"] = df["TotPkts"] / (df["Duration"] + 1e-6)
    df["Bytes_per_Sec"] = df["TotBytes"] / (df["Duration"] + 1e-6)
    df["Bytes_per_Pkt"] = df["TotBytes"] / (df["TotPkts"] + 1e-6)

    cols = ["Timestamp_dt", "Label", "Label_str", "Label_str_l"] + FEATURE_COLS
    return df[cols].copy()


def bind_session_window(
    df: pd.DataFrame, start_hms: str, end_hms: str, purge_min: int
) -> pd.DataFrame:
    if df.empty:
        return df

    day = df["Timestamp_dt"].dt.normalize().mode()[0]
    start = pd.to_datetime(f"{day.date()} {start_hms}")
    end = pd.to_datetime(f"{day.date()} {end_hms}")
    purge = pd.Timedelta(minutes=int(purge_min))

    return df[
        (df["Timestamp_dt"] >= start + purge)
        & (df["Timestamp_dt"] <= end - purge)
    ].copy()


def time_coverage_report(df_raw_rows: int, df_processed: pd.DataFrame, day_name: str):
    return {
        f"{day_name.lower()}_raw_rows": int(df_raw_rows),
        f"{day_name.lower()}_rows_after_processing": int(len(df_processed)),
        f"{day_name.lower()}_min_ts": str(df_processed["Timestamp_dt"].min())
        if len(df_processed)
        else None,
        f"{day_name.lower()}_max_ts": str(df_processed["Timestamp_dt"].max())
        if len(df_processed)
        else None,
    }


def summarize_split(name: str, df: pd.DataFrame):
    if df is None or df.empty:
        return {
            "name": name,
            "rows": 0,
            "pos": 0,
            "neg": 0,
            "pos_rate": np.nan,
            "min_ts": None,
            "max_ts": None,
        }

    y = df["Label"].values.astype(int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))

    return {
        "name": name,
        "rows": int(len(df)),
        "pos": pos,
        "neg": neg,
        "pos_rate": float(pos / max(len(df), 1)),
        "min_ts": str(df["Timestamp_dt"].min()),
        "max_ts": str(df["Timestamp_dt"].max()),
    }


def build_forward_time_folds(
    df_sorted: pd.DataFrame, n_blocks: int, embargo_minutes: int
):
    if df_sorted.empty:
        return []

    t_min = df_sorted["Timestamp_dt"].min()
    t_max = df_sorted["Timestamp_dt"].max()
    total_seconds = (t_max - t_min).total_seconds()

    if total_seconds <= 0:
        return []

    cut_seconds = np.linspace(0, total_seconds, n_blocks + 1)
    cuts = [t_min + pd.Timedelta(seconds=float(s)) for s in cut_seconds]
    embargo = pd.Timedelta(minutes=int(embargo_minutes))
    folds = []

    for i in range(1, n_blocks):
        train_end = cuts[i] - embargo
        val_start = cuts[i]
        val_end = cuts[i + 1]

        train_idx = df_sorted.index[df_sorted["Timestamp_dt"] <= train_end].to_numpy()
        val_idx = df_sorted.index[
            (df_sorted["Timestamp_dt"] >= val_start)
            & (df_sorted["Timestamp_dt"] < val_end)
        ].to_numpy()

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
        else:
            folds.append((np.array([], dtype=int), np.array([], dtype=int)))

    return folds


def benchmark_detailed(model, X_full, batch_size=50000, n_repeats=30, seed=42):
    rng = np.random.default_rng(seed)
    n = int(len(X_full))

    if n <= 2:
        return {
            "t_e2e_median": np.nan,
            "t_e2e_mean": np.nan,
            "t_e2e_std": np.nan,
            "throughput_e2e_median": np.nan,
            "t_dmatrix_median": np.nan,
            "t_predict_median": np.nan,
        }

    bs = int(batch_size)
    if bs >= n:
        bs = max(1, n // 2)

    t_e2e = []
    t_dm = []
    t_pr = []

    d_w = xgb.DMatrix(X_full[:bs])
    for _ in range(3):
        _ = model.predict(d_w)

    for _ in range(int(n_repeats)):
        idx = int(rng.integers(0, n - bs))
        X_batch = X_full[idx : idx + bs]

        t0 = time.perf_counter()
        dmat = xgb.DMatrix(X_batch)
        _ = model.predict(dmat)
        t1 = time.perf_counter()
        t_e2e.append(t1 - t0)

        t2 = time.perf_counter()
        dmat2 = xgb.DMatrix(X_batch)
        t3 = time.perf_counter()
        t_dm.append(t3 - t2)

        t4 = time.perf_counter()
        _ = model.predict(dmat2)
        t5 = time.perf_counter()
        t_pr.append(t5 - t4)

    arr = np.asarray(t_e2e, dtype=float)
    arr_dm = np.asarray(t_dm, dtype=float)
    arr_pr = np.asarray(t_pr, dtype=float)

    t_e2e_median = float(np.median(arr))
    return {
        "t_e2e_median": t_e2e_median,
        "t_e2e_mean": float(np.mean(arr)),
        "t_e2e_std": float(np.std(arr)),
        "throughput_e2e_median": float(bs / max(t_e2e_median, 1e-12)),
        "t_dmatrix_median": float(np.median(arr_dm)),
        "t_predict_median": float(np.median(arr_pr)),
        "batch_size": int(bs),
        "n_repeats": int(n_repeats),
    }


def safe_prf_at_threshold(y_true, probs, threshold):
    yhat = (probs > threshold).astype(int)
    p = precision_score(y_true, yhat, zero_division=0)
    r = recall_score(y_true, yhat, zero_division=0)
    f1 = f1_score(y_true, yhat, zero_division=0)
    return float(p), float(r), float(f1)


def safe_confusion(y_true, probs, threshold):
    yhat = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


def forward_oof_probs_xgb(
    X, y, trial_params, ratio, seed, n_blocks=6, min_train_frac=0.25
):
    n = len(y)
    probs_oof = np.full(n, np.nan, dtype=float)

    min_train = max(int(n * min_train_frac), 1000)
    block_size = max(int((n - min_train) / n_blocks), 1)

    params = trial_params.copy()
    n_est = int(params.pop("n_estimators"))

    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": ratio,
        "verbosity": 0,
        "n_jobs": 1,
        "nthread": 1,
        "seed": seed,
        **params,
    }

    invalid_blocks = 0

    for i in range(n_blocks):
        start = min_train + i * block_size
        end = min(start + block_size, n)

        if start >= n or end <= start:
            continue

        X_tr, y_tr = X[:start], y[:start]
        X_te = X[start:end]

        if len(np.unique(y_tr)) < 2:
            invalid_blocks += 1
            continue

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dte = xgb.DMatrix(X_te)

        model = xgb.train(full_params, dtr, num_boost_round=n_est)
        probs_oof[start:end] = model.predict(dte)

    valid_mask = np.isfinite(probs_oof)
    return probs_oof, valid_mask, invalid_blocks


def threshold_from_fpr_constraint(probs_neg, max_fpr):
    probs_neg = np.asarray(probs_neg, dtype=float)
    probs_neg = probs_neg[np.isfinite(probs_neg)]

    if probs_neg.size == 0:
        return 0.5

    q = float(np.quantile(probs_neg, 1.0 - max_fpr))
    return min(max(q + 1e-12, 0.0), 1.0)


def pick_threshold_max_f1_with_min_threshold(y_true, probs, min_threshold):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)

    mask = np.isfinite(probs)
    y_true = y_true[mask]
    probs = probs[mask]

    _, _, th_pr = precision_recall_curve(y_true, probs)

    candidates = []
    candidates.extend(list(th_pr))
    candidates.extend(list(np.linspace(0.0, 1.0, CONFIG["thresh_grid_size"])))

    candidates = np.array(sorted(set(candidates)), dtype=float)
    candidates = candidates[candidates >= float(min_threshold)]

    if candidates.size == 0:
        _, _, f1 = safe_prf_at_threshold(y_true, probs, float(min_threshold))
        return float(min_threshold), float(f1)

    best_th = float(min_threshold)
    best_f1 = -1.0

    for t in candidates:
        _, _, f1 = safe_prf_at_threshold(y_true, probs, float(t))
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(t)

    return best_th, best_f1


def train_from_trial(trial_obj, X_s1, y_s1, ratio, seed):
    params = trial_obj.params.copy()
    n_est = int(params.pop("n_estimators"))

    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": ratio,
        "verbosity": 0,
        "n_jobs": 1,
        "nthread": 1,
        "seed": seed,
        **params,
    }

    model = xgb.train(full_params, xgb.DMatrix(X_s1, label=y_s1), num_boost_round=n_est)
    return model, full_params, n_est


def select_representatives(df: pd.DataFrame, ap_floor: float = 0.95, f1_floor: float = 0.90):
    if df.empty:
        raise RuntimeError("Pareto dataframe is empty.")

    eligible = df[
        (df["ap_val"] >= ap_floor) &
        (df["mean_f1_val"] >= f1_floor)
    ].copy()

    if eligible.empty:
        eligible = df[df["ap_val"] >= ap_floor].copy()

    if eligible.empty:
        eligible = df.copy()

    heavy_id = int(eligible.loc[eligible["ap_val"].idxmax(), "trial_id"])
    light_id = int(eligible.loc[eligible["throughput"].idxmax(), "trial_id"])

    tmp = eligible.copy()
    tmp["throughput_norm"] = tmp["throughput"] / (tmp["throughput"].max() + 1e-12)
    tmp["ap_norm"] = tmp["ap_val"] / (tmp["ap_val"].max() + 1e-12)
    tmp["hmean"] = (
        2 * tmp["throughput_norm"] * tmp["ap_norm"] /
        (tmp["throughput_norm"] + tmp["ap_norm"] + 1e-12)
    )
    tmp = tmp.sort_values("hmean", ascending=False)

    picked = {"Light": light_id, "Heavy": heavy_id}
    balanced_id = None

    for tid in tmp["trial_id"].astype(int).tolist():
        if tid not in picked.values():
            balanced_id = int(tid)
            break

    if balanced_id is None:
        balanced_id = int(tmp.iloc[0]["trial_id"])

    return {
        "Light": int(light_id),
        "Balanced": int(balanced_id),
        "Heavy": int(heavy_id),
    }


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def _savefig_fig(fig, path: Path, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _model_order(names):
    preferred = ["Light", "Balanced", "Heavy"]
    ordered = [n for n in preferred if n in names]
    rest = [n for n in names if n not in ordered]
    return ordered + sorted(rest)


def compute_bin_metrics_from_cm(cm):
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def _round4(x):
    try:
        return round(float(x), 4)
    except Exception:
        return x


def _round_int(x):
    try:
        return int(round(float(x)))
    except Exception:
        return x


def main():
    save_run_metadata()

    if not IDS_FRI_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {IDS_FRI_PATH}. "
            "Place the Friday CSV file in the data/ directory."
        )

    print("Loading Friday data...")
    df_fri_raw = pd.read_csv(IDS_FRI_PATH, low_memory=False)
    fri_raw_rows = len(df_fri_raw)
    df_fri = process_ids_botnet(df_fri_raw)

    df_fri_main = df_fri[
        df_fri["Label_str_l"].isin([BOT_LABEL_TOKEN, BENIGN_LABEL_TOKEN])
    ].copy()
    df_fri_main = df_fri_main.sort_values("Timestamp_dt").reset_index(drop=True)

    df_s1 = bind_session_window(
        df_fri_main,
        CONFIG["session_1_start"],
        CONFIG["session_1_end"],
        CONFIG["purge_minutes"],
    )
    df_s2 = bind_session_window(
        df_fri_main,
        CONFIG["session_2_start"],
        CONFIG["session_2_end"],
        CONFIG["purge_minutes"],
    )

    if len(df_s1) == 0 or len(df_s2) == 0:
        raise RuntimeError("S1 or S2 is empty. Check session windows and file coverage.")

    X_s1 = df_s1[FEATURE_COLS].values
    y_s1 = df_s1["Label"].values.astype(int)
    X_s2 = df_s2[FEATURE_COLS].values
    y_s2 = df_s2["Label"].values.astype(int)

    ratio = float(np.sum(y_s1 == 0)) / max(float(np.sum(y_s1 == 1)), 1.0)

    df_thu = None
    thu_raw_rows = None
    if IDS_THU_PATH.exists():
        print("Loading Thursday data...")
        df_thu_raw = pd.read_csv(IDS_THU_PATH, low_memory=False)
        thu_raw_rows = len(df_thu_raw)
        df_thu = process_ids_botnet(df_thu_raw)

    fri_cov = time_coverage_report(fri_raw_rows, df_fri, "Friday")
    with open(DIAG_DIR / "fri_time_coverage.json", "w", encoding="utf-8") as f:
        json.dump(fri_cov, f, indent=2)

    if df_thu is not None:
        thu_cov = time_coverage_report(thu_raw_rows, df_thu, "Thursday")
        with open(DIAG_DIR / "thu_time_coverage.json", "w", encoding="utf-8") as f:
            json.dump(thu_cov, f, indent=2)

    split_rows = [
        summarize_split("Friday_S1_train_window", df_s1),
        summarize_split("Friday_S2_test_window", df_s2),
    ]
    if df_thu is not None:
        split_rows.append(summarize_split("Thursday_all_rows_botnet_as_1", df_thu))

    pd.DataFrame(split_rows).to_csv(DIAG_DIR / "split_summaries.csv", index=False)

    df_s1 = df_s1.sort_values("Timestamp_dt").reset_index(drop=True)
    X_s1 = df_s1[FEATURE_COLS].values
    y_s1 = df_s1["Label"].values.astype(int)

    folds = build_forward_time_folds(
        df_s1,
        CONFIG["n_blocks"],
        CONFIG["forward_embargo_minutes"],
    )

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "scale_pos_weight": ratio,
            "verbosity": 0,
            "n_jobs": 1,
            "nthread": 1,
            "seed": SEED,
            "max_bin": trial.suggest_int("max_bin", 32, 256),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        num_rounds = trial.suggest_int("n_estimators", 10, 500, step=10)

        ap_scores = []
        f1_scores = []
        invalid_blocks = 0

        for train_idx, val_idx in folds:
            if len(train_idx) == 0 or len(val_idx) == 0:
                invalid_blocks += 1
                continue

            y_val = y_s1[val_idx]
            if len(np.unique(y_val)) < 2:
                invalid_blocks += 1
                continue

            dtrain = xgb.DMatrix(X_s1[train_idx], label=y_s1[train_idx])
            dval = xgb.DMatrix(X_s1[val_idx], label=y_val)

            model = xgb.train(params, dtrain, num_boost_round=num_rounds)
            probs = model.predict(dval)

            ap_scores.append(average_precision_score(y_val, probs))
            yhat = (probs > 0.5).astype(int)
            f1_scores.append(f1_score(y_val, yhat, zero_division=0))

        if len(ap_scores) == 0:
            mean_ap = 0.0
            mean_f1 = 0.0
        else:
            mean_ap = float(np.mean(ap_scores))
            mean_f1 = float(np.mean(f1_scores)) if len(f1_scores) else 0.0

        dtrain_full = xgb.DMatrix(X_s1, label=y_s1)
        model_full = xgb.train(params, dtrain_full, num_boost_round=num_rounds)

        bench = benchmark_detailed(
            model_full,
            X_s1,
            batch_size=CONFIG["bench_batch_size"],
            n_repeats=CONFIG["bench_repeats"],
            seed=SEED,
        )

        trial.set_user_attr("n_estimators", int(num_rounds))
        trial.set_user_attr("mean_ap_val", float(mean_ap))
        trial.set_user_attr("mean_f1_val", float(mean_f1))
        trial.set_user_attr("invalid_val_blocks", int(invalid_blocks))
        trial.set_user_attr("bench", bench)

        return float(mean_ap), float(bench["throughput_e2e_median"])

    print("Starting NSGA-II optimization...")
    sampler = optuna.samplers.NSGAIISampler(seed=SEED)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=CONFIG["n_trials"])

    pareto_trials = study.best_trials
    pareto_rows = []

    for t in pareto_trials:
        bench = t.user_attrs.get("bench", {})
        pareto_rows.append({
            "trial_id": int(t.number),
            "ap_val": float(t.values[0]),
            "throughput": float(t.values[1]),
            "n_estimators": int(t.user_attrs.get("n_estimators", t.params.get("n_estimators", -1))),
            "mean_f1_val": float(t.user_attrs.get("mean_f1_val", float("nan"))),
            "invalid_val_blocks": int(t.user_attrs.get("invalid_val_blocks", 0)),
            "bench_t_e2e_median_s": float(bench.get("t_e2e_median", float("nan"))),
            "bench_t_e2e_mean_s": float(bench.get("t_e2e_mean", float("nan"))),
            "bench_t_e2e_std_s": float(bench.get("t_e2e_std", float("nan"))),
            "bench_t_dmatrix_median_s": float(bench.get("t_dmatrix_median", float("nan"))),
            "bench_t_predict_median_s": float(bench.get("t_predict_median", float("nan"))),
            "params_json": json.dumps(t.params),
        })

    df_pareto = pd.DataFrame(pareto_rows)
    df_pareto.to_csv(DIAG_DIR / "pareto_points.csv", index=False)

    selected_ids = select_representatives(df_pareto, ap_floor=0.95, f1_floor=0.90)
    with open(DIAG_DIR / "selected_ids.json", "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, indent=2)

    trial_map = {t.number: t for t in study.trials}

    X_thu_all = None
    X_thu_benign = None
    if df_thu is not None:
        X_thu_all = df_thu[FEATURE_COLS].values
        if "Label_str_l" in df_thu.columns:
            df_thu_ben = df_thu[df_thu["Label_str_l"] == BENIGN_LABEL_TOKEN].copy()
            X_thu_benign = (
                df_thu_ben[FEATURE_COLS].values if not df_thu_ben.empty else X_thu_all
            )
        else:
            X_thu_benign = X_thu_all

    results = []
    rep_json = {}

    for rep_name, tid in selected_ids.items():
        t = trial_map[int(tid)]

        model, full_params, n_est = train_from_trial(t, X_s1, y_s1, ratio, SEED)

        probs_oof, valid_mask, invalid_blocks = forward_oof_probs_xgb(
            X_s1,
            y_s1,
            t.params,
            ratio,
            SEED,
            n_blocks=CONFIG["n_oof_blocks"],
            min_train_frac=0.25,
        )
        y_oof = y_s1[valid_mask]
        p_oof = probs_oof[valid_mask]

        if X_thu_benign is not None:
            probs_thu_b = model.predict(xgb.DMatrix(X_thu_benign))
            min_th = threshold_from_fpr_constraint(probs_thu_b, CONFIG["max_thu_fpr"])
            constraint_mode = f"thursday_fpr<= {CONFIG['max_thu_fpr']}"
        else:
            probs_oof_neg = p_oof[y_oof == 0]
            min_th = threshold_from_fpr_constraint(probs_oof_neg, CONFIG["max_oof_fpr"])
            constraint_mode = f"oof_neg_fpr<= {CONFIG['max_oof_fpr']}"

        best_th, best_f1_oof = pick_threshold_max_f1_with_min_threshold(y_oof, p_oof, min_th)

        _, _, f1_oof_05 = safe_prf_at_threshold(y_oof, p_oof, 0.5)
        _, _, f1_oof_min = safe_prf_at_threshold(y_oof, p_oof, min_th)

        probs_s2 = model.predict(xgb.DMatrix(X_s2))
        p_s2, r_s2, f1_s2 = safe_prf_at_threshold(y_s2, probs_s2, best_th)
        tp, fp, fn, tn = safe_confusion(y_s2, probs_s2, best_th)

        if len(np.unique(y_s2)) > 1:
            auc_s2 = float(roc_auc_score(y_s2, probs_s2))
            pr_s2 = float(average_precision_score(y_s2, probs_s2))
        else:
            auc_s2 = float("nan")
            pr_s2 = float("nan")

        thu_rate_all = np.nan
        thu_fpr_benign = np.nan

        if X_thu_all is not None:
            probs_thu_all = model.predict(xgb.DMatrix(X_thu_all))
            thu_rate_all = float(np.mean(probs_thu_all > best_th))

            if X_thu_benign is not None:
                probs_thu_b = model.predict(xgb.DMatrix(X_thu_benign))
                thu_fpr_benign = float(np.mean(probs_thu_b > best_th))

        bench = benchmark_detailed(
            model,
            X_s1,
            batch_size=CONFIG["bench_batch_size"],
            n_repeats=CONFIG["bench_repeats"],
            seed=SEED,
        )

        ap_val_forward = float(t.values[0])

        results.append({
            "Model": rep_name,
            "trial_id": int(tid),
            "Val_AP_forward": ap_val_forward,
            "Constraint_Mode": constraint_mode,
            "Constraint_MinThreshold": float(min_th),
            "Calibrated_Threshold": float(best_th),
            "Val_OOF_Max_F1": float(best_f1_oof),
            "Val_OOF_F1@0.5": float(f1_oof_05),
            "Val_OOF_F1@MinTh": float(f1_oof_min),
            "invalid_val_blocks": int(invalid_blocks),
            "Test_S2_P": float(p_s2),
            "Test_S2_R": float(r_s2),
            "Test_S2_F1": float(f1_s2),
            "Test_S2_TP": int(tp),
            "Test_S2_FP": int(fp),
            "Test_S2_FN": int(fn),
            "Test_S2_TN": int(tn),
            "Test_S2_ROC_AUC": float(auc_s2),
            "Test_S2_PR_AUC": float(pr_s2),
            "Thu_AlertRate_All": float(thu_rate_all) if np.isfinite(thu_rate_all) else np.nan,
            "Thu_FPR_Benign": float(thu_fpr_benign) if np.isfinite(thu_fpr_benign) else np.nan,
            "Throughput_Median": float(bench["throughput_e2e_median"]),
            "N_Trees": int(n_est),
        })

        rep_json[rep_name] = {
            "trial_id": int(tid),
            "params": full_params,
            "n_trees": int(n_est),
            "constraint_mode": constraint_mode,
            "min_threshold": float(min_th),
            "calibrated_threshold": float(best_th),
            "oof": {
                "max_f1": float(best_f1_oof),
                "f1_at_0_5": float(f1_oof_05),
                "f1_at_min_th": float(f1_oof_min),
                "invalid_blocks": int(invalid_blocks),
            },
            "metrics_s2": {
                "F1": float(f1_s2),
                "P": float(p_s2),
                "R": float(r_s2),
                "PR_AUC": float(pr_s2),
                "ROC_AUC": float(auc_s2),
            },
            "metrics_thursday": {
                "FPR_Benign": float(thu_fpr_benign) if np.isfinite(thu_fpr_benign) else None,
                "AlertRate_All": float(thu_rate_all) if np.isfinite(thu_rate_all) else None,
            },
            "benchmark": bench,
        }

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "final_calibrated_results.csv", index=False)

    with open(DIAG_DIR / "calibrated_models.json", "w", encoding="utf-8") as f:
        json.dump(rep_json, f, indent=2)

    for p in PAPER_FIG_DIR.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass

    df_selected = df_res.copy()
    model_names = (
        df_selected["Model"].tolist()
        if (not df_selected.empty and "Model" in df_selected.columns)
        else list(selected_ids.keys())
    )
    model_names = [m for m in _model_order(model_names) if m in rep_json]

    model_cache = {}

    def train_model_from_rep(model_name: str):
        if model_name not in rep_json:
            raise KeyError(f"Model '{model_name}' not found in rep_json")
        blob = rep_json[model_name]
        params = dict(blob["params"])
        n_trees = int(blob["n_trees"])
        dtrain = xgb.DMatrix(X_s1, label=y_s1)
        model = xgb.train(params, dtrain, num_boost_round=n_trees)
        return model, blob

    def get_model_blob(model_name: str):
        if model_name in model_cache:
            return model_cache[model_name]
        model, blob = train_model_from_rep(model_name)
        model_cache[model_name] = (model, blob)
        return model, blob

    if not df_pareto.empty:
        fig = plt.figure(figsize=(9, 6))
        plt.scatter(df_pareto["throughput"], df_pareto["ap_val"], alpha=0.7)
        plt.xlabel("E2E Throughput (flows/sec) [median]")
        plt.ylabel("Forward-Validation PR-AUC (Average Precision)")
        plt.title("Pareto Frontier: Detection Quality vs Efficiency")
        plt.grid(True, linestyle="--", alpha=0.4)

        for name, tid in selected_ids.items():
            pr = df_pareto[df_pareto["trial_id"] == int(tid)]
            if not pr.empty:
                x = float(pr.iloc[0]["throughput"])
                y = float(pr.iloc[0]["ap_val"])
                plt.scatter([x], [y], marker="x", s=180)
                plt.text(x, y, f"  {name}", fontsize=10, fontweight="bold")

        _savefig_fig(fig, PAPER_FIG_DIR / "fig1_pareto_frontier.png")

    if not df_pareto.empty:
        rows = []
        for model_name, tid in selected_ids.items():
            pr = df_pareto[df_pareto["trial_id"] == int(tid)]
            if pr.empty:
                continue
            rows.append({
                "Model": model_name,
                "t_dmatrix": float(pr.iloc[0].get("bench_t_dmatrix_median_s", np.nan)),
                "t_predict": float(pr.iloc[0].get("bench_t_predict_median_s", np.nan)),
            })

        df_decomp = pd.DataFrame(rows).dropna()

        if not df_decomp.empty:
            df_decomp["__ord"] = df_decomp["Model"].map({
                m: i for i, m in enumerate(_model_order(df_decomp["Model"].tolist()))
            })
            df_decomp = df_decomp.sort_values("__ord").drop(columns=["__ord"])

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            x = np.arange(len(df_decomp))

            ax.bar(x, df_decomp["t_dmatrix"].values, label="DMatrix")
            ax.bar(
                x,
                df_decomp["t_predict"].values,
                bottom=df_decomp["t_dmatrix"].values,
                label="Predict",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(df_decomp["Model"].astype(str).tolist())
            ax.set_ylabel("Median time per batch (s)")
            ax.set_title("E2E Time Decomposition (median)")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            _savefig_fig(fig, PAPER_FIG_DIR / "fig2_e2e_time_decomposition.png")

    if len(model_names) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ax_pr, ax_roc = axes[0], axes[1]

        for model_name in model_names:
            model, _ = get_model_blob(model_name)
            probs = model.predict(xgb.DMatrix(X_s2))

            prec, rec, _ = precision_recall_curve(y_s2, probs)
            pr_auc = average_precision_score(y_s2, probs)
            ax_pr.plot(rec, prec, label=f"{model_name} (AP={pr_auc:.4f})")

            fpr, tpr, _ = roc_curve(y_s2, probs)
            roc_auc = roc_auc_score(y_s2, probs)
            ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})")

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall (S2)")
        ax_pr.grid(True, linestyle="--", alpha=0.35)
        ax_pr.legend(loc="lower left")

        ax_roc.plot([0, 1], [0, 1], linestyle="--", alpha=0.6, label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC (S2)")
        ax_roc.grid(True, linestyle="--", alpha=0.35)
        ax_roc.legend(loc="lower right")

        fig.suptitle("Curves on S2 (Future Window)", y=1.02)
        _savefig_fig(fig, PAPER_FIG_DIR / "fig3_s2_pr_roc_curves.png")

    if len(model_names) > 0:
        n = len(model_names)
        fig_w = 5.3 * n
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.2))

        if n == 1:
            axes = [axes]

        cms = []
        metas = []
        vmax = 0

        for model_name in model_names:
            model, blob = get_model_blob(model_name)
            probs = model.predict(xgb.DMatrix(X_s2))
            th = _safe_float(blob.get("calibrated_threshold", 0.5), 0.5)
            yhat = (probs > th).astype(int)
            cm = confusion_matrix(y_s2, yhat, labels=[0, 1])
            m = compute_bin_metrics_from_cm(cm)

            cms.append(cm)
            metas.append((model_name, th, m))
            vmax = max(vmax, int(cm.max()))

        im = None
        for ax, (cm, (model_name, th, m)) in zip(axes, zip(cms, metas)):
            im = ax.imshow(cm, vmin=0, vmax=vmax)
            ax.set_title(f"{model_name} (th={th:.4f})")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Malicious"])
            ax.set_yticklabels(["Normal", "Malicious"])

            ax.set_xlabel("Predicted")
            if ax is axes[0]:
                ax.set_ylabel("True")

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center")

            metrics_line = (
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
                f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}"
            )
            ax.text(
                0.5,
                -0.18,
                metrics_line,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=10,
            )

        fig.suptitle("S2 Confusion Matrices at Calibrated Thresholds", y=1.02)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.05)
        cbar.set_label("Count")

        _savefig_fig(fig, PAPER_FIG_DIR / "fig6_s2_confmat_panel.png")

    if len(model_names) > 0:
        fig, ax = plt.subplots(figsize=(9, 6))
        bins = 60

        for model_name in model_names:
            model, blob = get_model_blob(model_name)
            probs = model.predict(xgb.DMatrix(X_s2))
            th = _safe_float(blob.get("calibrated_threshold", np.nan))
            label = f"{model_name}" + (f" (th={th:.3f})" if np.isfinite(th) else "")

            ax.hist(probs, bins=bins, alpha=0.30, density=True, label=label)
            if np.isfinite(th):
                ax.axvline(th, linestyle="--", alpha=0.6)

        ax.set_xlabel("Predicted probability score")
        ax.set_ylabel("Density (histogram)")
        ax.set_title("S2 Score Distributions (with calibrated thresholds)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        _savefig_fig(fig, PAPER_FIG_DIR / "fig5_s2_score_distributions.png")

    if X_thu_benign is not None and len(model_names) > 0:
        thresholds = np.linspace(0, 1, 500)
        fig, ax = plt.subplots(figsize=(9, 6))

        for model_name in model_names:
            model, blob = get_model_blob(model_name)
            probs_thu = model.predict(xgb.DMatrix(X_thu_benign))
            fprs = np.array([(probs_thu > t).mean() for t in thresholds], dtype=float)

            line, = ax.plot(thresholds, fprs, label=model_name)
            color = line.get_color()

            min_th = _safe_float(blob.get("min_threshold", np.nan))
            cal_th = _safe_float(blob.get("calibrated_threshold", np.nan))

            if np.isfinite(min_th):
                ax.axvline(min_th, linestyle="--", alpha=0.45, color=color)
            if np.isfinite(cal_th):
                ax.axvline(cal_th, linestyle=":", alpha=0.60, color=color)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Benign alert rate / FPR on Thursday")
        ax.set_title("Negative-Day Reliability: FPR vs Threshold (Thursday)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        _savefig_fig(fig, PAPER_FIG_DIR / "fig4_thursday_fpr_vs_threshold_all.png")

    # ============================================================
    # ARTICLE-READY EXPORTS
    # ============================================================

    # Table 1: Selected Pareto-optimal configurations
    table1_rows = []
    for model_name, tid in selected_ids.items():
        pr = df_pareto[df_pareto["trial_id"] == int(tid)]
        if pr.empty:
            continue

        row = pr.iloc[0]
        params = json.loads(row["params_json"])

        table1_rows.append({
            "Model": model_name,
            "Val AP forward": _round4(row["ap_val"]),
            "Throughput E2E median": _round_int(row["throughput"]),
            "learning rate": _round4(params.get("learning_rate")),
            "max depth": int(params.get("max_depth")),
            "subsample": _round4(params.get("subsample")),
            "colsample bytree": _round4(params.get("colsample_bytree")),
            "n estimators": int(row["n_estimators"]),
        })

    df_table1 = pd.DataFrame(table1_rows)
    table1_order = {"Balanced": 0, "Heavy": 1, "Light": 2}
    if not df_table1.empty:
        df_table1["__ord"] = df_table1["Model"].map(table1_order).fillna(999)
        df_table1 = df_table1.sort_values("__ord").drop(columns="__ord")

    df_table1.to_csv(OUTPUT_DIR / "table1_selected_pareto_configs.csv", index=False)

    # Table 2: Compact article results
    table2_cols = [
        "Model",
        "Constraint_MinThreshold",
        "Calibrated_Threshold",
        "Test_S2_F1",
        "Test_S2_ROC_AUC",
        "Test_S2_PR_AUC",
        "Thu_FPR_Benign",
        "Throughput_Median",
        "N_Trees",
    ]

    df_table2 = df_res[table2_cols].copy()
    df_table2 = df_table2.rename(columns={
        "Constraint_MinThreshold": "Constraint MinThreshold",
        "Calibrated_Threshold": "Calibrated Threshold",
        "Test_S2_F1": "Test S2 F1",
        "Test_S2_ROC_AUC": "Test S2 ROC AUC",
        "Test_S2_PR_AUC": "Test S2 PR AUC",
        "Thu_FPR_Benign": "Thu FPR Benign",
        "Throughput_Median": "Throughput Median",
        "N_Trees": "N Trees",
    })

    for col in [
        "Constraint MinThreshold",
        "Calibrated Threshold",
        "Test S2 F1",
        "Test S2 ROC AUC",
        "Test S2 PR AUC",
        "Thu FPR Benign",
    ]:
        df_table2[col] = df_table2[col].apply(_round4)

    df_table2["Throughput Median"] = df_table2["Throughput Median"].apply(_round_int)

    table2_order = {"Light": 0, "Balanced": 1, "Heavy": 2}
    if not df_table2.empty:
        df_table2["__ord"] = df_table2["Model"].map(table2_order).fillna(999)
        df_table2 = df_table2.sort_values("__ord").drop(columns="__ord")

    df_table2.to_csv(OUTPUT_DIR / "table2_article_results.csv", index=False)

    # Full rounded article table
    df_table2_full = df_res.copy()
    round_cols = [
        "Val_AP_forward",
        "Constraint_MinThreshold",
        "Calibrated_Threshold",
        "Val_OOF_Max_F1",
        "Val_OOF_F1@0.5",
        "Val_OOF_F1@MinTh",
        "Test_S2_P",
        "Test_S2_R",
        "Test_S2_F1",
        "Test_S2_ROC_AUC",
        "Test_S2_PR_AUC",
        "Thu_AlertRate_All",
        "Thu_FPR_Benign",
        "Throughput_Median",
    ]
    for col in round_cols:
        if col in df_table2_full.columns:
            if col == "Throughput_Median":
                df_table2_full[col] = df_table2_full[col].apply(_round_int)
            else:
                df_table2_full[col] = df_table2_full[col].apply(_round4)

    df_table2_full.to_csv(OUTPUT_DIR / "table2_article_results_full.csv", index=False)

    caption_helper = """Figure 1. Pareto frontier: detection quality versus efficiency.
Figure 2. End-to-end time decomposition (DMatrix construction and prediction).
Figure 3. Precision-Recall and ROC curves on S2 (future window).
Figure 4. Negative-day reliability: Thursday benign FPR versus threshold.
Figure 5. Score distributions on S2 with calibrated thresholds.
Figure 6. Confusion matrices on S2 at calibrated thresholds.

Table 1. Selected Pareto-optimal XGBoost configurations with key hyperparameters and efficiency-quality indicators.
Table 2. Final calibrated results on S2 and Thursday negative-day reliability.
"""
    with open(OUTPUT_DIR / "article_caption_helper.txt", "w", encoding="utf-8") as f:
        f.write(caption_helper)

    print("Artifacts saved to:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {DIAG_DIR}")
    print(f"  {PAPER_FIG_DIR}")
    print(f"  {OUTPUT_DIR / 'table1_selected_pareto_configs.csv'}")
    print(f"  {OUTPUT_DIR / 'table2_article_results.csv'}")
    print(f"  {OUTPUT_DIR / 'table2_article_results_full.csv'}")
    print(f"  {OUTPUT_DIR / 'article_caption_helper.txt'}")


if __name__ == "__main__":
    main()
