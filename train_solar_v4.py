#!/usr/bin/env python3 -u
"""
Train LiquidNet v4 — maximum accuracy.

Changes from v3:
  1. More OMNI2 features: +electric field, plasma beta, Alfven Mach, Dst, AE
  2. Bigger model: 96 hidden x 2 layers (~68K params, was 3.7K)
  3. Ensemble of 5 models (was 3)
  4. 72h sequences (was 48h)
  5. Temporal smoothing (3h rolling mean on output)
  6. More engineered features: d_bt, d_density, d_dst, dst lags

Usage:
    python3 -u train_solar_v4.py
"""

import os
import sys
import copy
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from lnn import LiquidNet

# Unbuffered output
print = lambda *a, **k: (sys.stdout.write(" ".join(map(str, a)) + k.get("end", "\n")), sys.stdout.flush())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "data"
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{}.dat"

YEARS_TRAIN = list(range(1963, 2023))
YEARS_VAL = [2023, 2024]
YEARS_TEST = [2025]

SEQ_LEN = 72          # 72h (3 days)
LEAD_TIME = 1
HIDDEN_SIZE = 96       # bigger
NUM_LAYERS = 2         # deeper
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 256
LR = 1e-3
N_ENSEMBLE = 5         # more models

# OMNI2 columns: (index, fill_value)
# v3 had 6 raw features. v4 adds 5 more.
COLS_INPUT = {
    # v3 features
    "bt":            (8,  999.9),       # |B| total field, nT
    "bz_gsm":        (16, 999.9),       # Bz GSM, nT — #1 storm driver
    "speed":         (24, 9999.),       # solar wind speed, km/s
    "density":       (23, 999.9),       # proton density, N/cm³
    "temp":          (22, 9999999.),    # proton temperature, K
    "pressure":      (28, 99.99),       # flow pressure, nPa
    # v4 additions
    "electric_field": (35, 999.99),     # E field, mV/m (v × B)
    "plasma_beta":   (36, 999.99),      # plasma/magnetic pressure ratio
    "alfven_mach":   (37, 999.9),       # Alfven Mach number
    "dst":           (40, 99999),       # Dst index, nT — storm indicator (hourly)
    "ae":            (41, 9999),        # AE auroral electrojet, nT — substorm indicator
}
COL_KP = (38, 99)
FEAT_NAMES = list(COLS_INPUT.keys())


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def download(years):
    os.makedirs(DATA_DIR, exist_ok=True)
    to_download = [y for y in years if not os.path.exists(os.path.join(DATA_DIR, f"omni2_{y}.dat"))]
    if not to_download:
        print(f"  All {len(years)} files cached")
        return
    print(f"  {len(years) - len(to_download)} cached, {len(to_download)} to download")
    for i, year in enumerate(to_download):
        sys.stdout.write(f"\r  Downloading: {year} ({i+1}/{len(to_download)})   ")
        sys.stdout.flush()
        urllib.request.urlretrieve(OMNI_URL.format(year), os.path.join(DATA_DIR, f"omni2_{year}.dat"))
    print(f"\r  Downloaded {len(to_download)} files                         ")


def parse(years):
    timestamps, features, kp_vals = [], [], []
    col_indices = [v[0] for v in COLS_INPUT.values()]
    fill_vals = [v[1] for v in COLS_INPUT.values()]

    for year in years:
        with open(os.path.join(DATA_DIR, f"omni2_{year}.dat")) as f:
            for line in f:
                p = line.split()
                yr, doy, hr = int(p[0]), int(p[1]), int(p[2])
                timestamps.append(datetime(yr, 1, 1) + timedelta(days=doy - 1, hours=hr))

                row = []
                for ci, fv in zip(col_indices, fill_vals):
                    val = float(p[ci])
                    row.append(np.nan if abs(val) >= abs(fv) * 0.99 else val)
                features.append(row)

                kp_raw = float(p[COL_KP[0]])
                kp_vals.append(np.nan if kp_raw >= COL_KP[1] else kp_raw / 10.0)

    return timestamps, np.array(features, dtype=np.float32), np.array(kp_vals, dtype=np.float32)


def fill_forward(arr):
    if arr.ndim == 1:
        mask = np.isnan(arr)
        if not mask.any():
            return arr
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]
    for col in range(arr.shape[1]):
        arr[:, col] = fill_forward(arr[:, col])
    return arr


def engineer_features(features, kp):
    """v4: more engineered features than v3."""
    n = len(features)

    # Name lookup for raw features
    fi = {name: i for i, name in enumerate(FEAT_NAMES)}

    bz = features[:, fi["bz_gsm"]]
    bt = features[:, fi["bt"]]
    speed = features[:, fi["speed"]]
    density = features[:, fi["density"]]
    dst = features[:, fi["dst"]]

    def diff(x):
        d = np.zeros(n, dtype=np.float32)
        d[1:] = x[1:] - x[:-1]
        return d

    def lag(x, k):
        l = np.zeros(n, dtype=np.float32)
        l[k:] = x[:-k]
        return l

    def rolling_mean(x, w):
        return np.convolve(x, np.ones(w) / w, mode='same').astype(np.float32)

    engineered = {
        # Rate of change (how fast things are shifting)
        "d_bz":       diff(bz),
        "d_bt":       diff(bt),
        "d_speed":    diff(speed),
        "d_density":  diff(density),
        "d_dst":      diff(dst),
        # Rolling means (trends)
        "bz_3h":      rolling_mean(bz, 3),
        "bt_6h":      rolling_mean(bt, 6),
        "speed_6h":   rolling_mean(speed, 6),
        "density_3h": rolling_mean(density, 3),
        # Coupling functions (physics-based)
        "epsilon":    (speed * np.clip(-bz, 0, None)).astype(np.float32),
        # Autoregressive Kp
        "kp_lag1":    lag(kp, 1),
        "kp_lag3":    lag(kp, 3),
        "kp_lag6":    lag(kp, 6),
        # Autoregressive Dst
        "dst_lag1":   lag(dst, 1),
        "dst_lag3":   lag(dst, 3),
    }

    eng_names = list(engineered.keys())
    eng_arr = np.column_stack([engineered[k] for k in eng_names])
    return np.column_stack([features, eng_arr]), eng_names


def make_sequences_fast(features, kp, timestamps, seq_len, lead):
    n = len(features)
    total = n - seq_len - lead + 1

    row_has_nan = np.any(np.isnan(features), axis=1)
    kp_is_nan = np.isnan(kp)

    nan_cs = np.concatenate([[0], np.cumsum(row_has_nan.astype(np.int32))])
    window_nan = nan_cs[seq_len:seq_len + total] - nan_cs[:total]

    target_idx = np.arange(total) + seq_len + lead - 1
    valid = (window_nan == 0) & (~kp_is_nan[target_idx])
    vi = np.where(valid)[0]

    print(f"{len(vi):,} valid / {total:,} ({100*len(vi)/total:.0f}%)")

    idx_mat = vi[:, None] + np.arange(seq_len)[None, :]
    X = features[idx_mat]
    Y = kp[target_idx[vi]]
    dates = [timestamps[target_idx[i]] for i in vi]

    return X, Y, dates


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class StormWeightedHuber(nn.Module):
    def __init__(self, storm_threshold=4.0, storm_weight=5.0, delta=1.0):
        super().__init__()
        self.threshold = storm_threshold
        self.weight = storm_weight
        self.delta = delta

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        weights[target >= self.threshold] = self.weight
        diff = torch.abs(pred - target)
        huber = torch.where(diff <= self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return (weights * huber).mean()


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_one(model, X_tr, Y_tr, X_val, Y_val, seed, device):
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = StormWeightedHuber()
    eval_fn = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    n = len(X_tr)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        eloss, nb = 0.0, 0

        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            pred = model(X_tr[idx])[:, -1, 0]
            loss = loss_fn(pred, Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item()
            nb += 1

        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                ps = [model(X_val[s:s+BATCH_SIZE])[:, -1, 0] for s in range(0, len(X_val), BATCH_SIZE)]
                vl = eval_fn(torch.cat(ps), Y_val).item()

            if vl < best_val:
                best_val = vl
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
                mk = " *"
            else:
                patience += 5
                mk = ""

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:4d} | train: {eloss/nb:.4f} | val: {vl:.4f}{mk}")

            if patience >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"    Best val: {best_val:.4f}")
    return model, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Download
    all_years = YEARS_TRAIN + YEARS_VAL + YEARS_TEST
    print(f"Downloading {len(all_years)} OMNI2 files...")
    download(all_years)
    print()

    # Parse
    print("Parsing 60 years of training data...")
    train_ts, train_feat, train_kp = parse(YEARS_TRAIN)
    print(f"  Train: {len(train_ts):,} hours ({YEARS_TRAIN[0]}-{YEARS_TRAIN[-1]})")
    val_ts, val_feat, val_kp = parse(YEARS_VAL)
    print(f"  Val:   {len(val_ts):,} hours")
    test_ts, test_feat, test_kp = parse(YEARS_TEST)
    print(f"  Test:  {len(test_ts):,} hours")

    # Missing data
    miss = 100 * np.isnan(train_feat).mean()
    print(f"\n  Overall missing: {miss:.1f}%")
    for i, name in enumerate(FEAT_NAMES):
        pct = 100 * np.isnan(train_feat[:, i]).mean()
        if pct > 1:
            print(f"    {name}: {pct:.1f}%")
    print()

    # Clean
    print("Forward-filling missing data...")
    for arr in [train_feat, val_feat, test_feat]:
        fill_forward(arr)
    train_kp = fill_forward(train_kp)
    val_kp = fill_forward(val_kp)
    test_kp = fill_forward(test_kp)

    # Log-scale skewed features
    temp_idx = FEAT_NAMES.index("temp")
    for feat in [train_feat, val_feat, test_feat]:
        feat[:, temp_idx] = np.log1p(np.clip(feat[:, temp_idx], 0, None))

    # Engineer features
    print("Engineering features...")
    train_feat, eng_names = engineer_features(train_feat, train_kp)
    val_feat, _ = engineer_features(val_feat, val_kp)
    test_feat, _ = engineer_features(test_feat, test_kp)
    all_feat_names = FEAT_NAMES + eng_names
    print(f"  {len(all_feat_names)} features: {', '.join(all_feat_names)}")
    print()

    # Normalize
    print("Normalizing...")
    mu = np.nanmean(train_feat, axis=0)
    std = np.nanstd(train_feat, axis=0)
    std[std < 1e-8] = 1.0
    train_feat = (train_feat - mu) / std
    val_feat = (val_feat - mu) / std
    test_feat = (test_feat - mu) / std

    # Sequences
    print("Building sequences...")
    print("  Train: ", end="")
    X_tr, Y_tr, _ = make_sequences_fast(train_feat, train_kp, train_ts, SEQ_LEN, LEAD_TIME)
    print("  Val:   ", end="")
    X_val, Y_val, val_dates = make_sequences_fast(val_feat, val_kp, val_ts, SEQ_LEN, LEAD_TIME)
    print("  Test:  ", end="")
    X_te, Y_te, test_dates = make_sequences_fast(test_feat, test_kp, test_ts, SEQ_LEN, LEAD_TIME)

    print(f"\n  Storm %: train {100*(Y_tr>=5).mean():.1f}%, test {100*(Y_te>=5).mean():.1f}%")
    print()

    # To device
    print("Moving to device...")
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    Y_tr_t = torch.FloatTensor(Y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)
    Y_te_t = torch.FloatTensor(Y_te).to(device)
    print(f"  Train: {X_tr_t.shape} ({X_tr_t.nbytes/1e9:.2f} GB)")
    print()

    input_size = train_feat.shape[1]

    # ===================================================================
    # TRAIN ENSEMBLE
    # ===================================================================
    print("=" * 70)
    print(f"  TRAINING ENSEMBLE OF {N_ENSEMBLE} MODELS (v4)")
    print(f"  {input_size} in -> {HIDDEN_SIZE} hidden x {NUM_LAYERS} layers")
    print(f"  Sequence: {SEQ_LEN}h -> Kp {LEAD_TIME}h ahead")
    print("=" * 70)
    print()

    models = []
    n_params = None
    for i in range(N_ENSEMBLE):
        print(f"  --- Model {i+1}/{N_ENSEMBLE} (seed={42+i}) ---")
        m = LiquidNet(input_size, HIDDEN_SIZE, 1, NUM_LAYERS, DROPOUT)
        if n_params is None:
            n_params = sum(p.numel() for p in m.parameters())
            print(f"    Parameters: {n_params:,}")
        m, _ = train_one(m, X_tr_t, Y_tr_t, X_val_t, Y_val_t, 42 + i, device)
        models.append(m)
        print()

    # ===================================================================
    # CALIBRATE THRESHOLD
    # ===================================================================
    print("Calibrating storm threshold...")
    for m in models:
        m.eval()

    def ensemble_predict(X_t):
        with torch.no_grad():
            all_p = []
            for m in models:
                ps = [m(X_t[s:s+BATCH_SIZE])[:, -1, 0] for s in range(0, len(X_t), BATCH_SIZE)]
                all_p.append(torch.cat(ps).cpu().numpy())
            return np.mean(all_p, axis=0)

    val_pred = ensemble_predict(X_val_t)
    val_actual = Y_val[:len(val_pred)]
    val_storm = val_actual >= 5.0

    best_f1, best_thresh = 0, 4.0
    for th in np.arange(3.0, 6.0, 0.1):
        ps = val_pred >= th
        tp = (ps & val_storm).sum()
        fp = (ps & ~val_storm).sum()
        fn = (~ps & val_storm).sum()
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-8)
        if f1 > best_f1:
            best_f1, best_thresh = f1, th

    print(f"  Threshold: {best_thresh:.1f} (F1={best_f1:.3f} on val)")
    print()

    # ===================================================================
    # TEST EVALUATION
    # ===================================================================
    raw_pred = ensemble_predict(X_te_t)
    actual = Y_te[:len(raw_pred)]

    # Temporal smoothing (3h rolling mean)
    smoothed = np.convolve(raw_pred, np.ones(3) / 3, mode='same').astype(np.float32)

    for label, predicted in [("Raw", raw_pred), ("Smoothed (3h)", smoothed)]:
        mse = np.mean((predicted - actual) ** 2)
        mae = np.mean(np.abs(predicted - actual))
        corr = np.corrcoef(actual, predicted)[0, 1]

        print("=" * 70)
        print(f"  RESULTS v4 [{label}]: Predictions vs Reality (2025)")
        print("=" * 70)
        print(f"\n  MSE:         {mse:.4f}")
        print(f"  MAE:         {mae:.4f} Kp")
        print(f"  Correlation: {corr:.3f}")

        storm_mask = actual >= 5.0
        n_storm = storm_mask.sum()
        if n_storm > 0:
            ps = predicted >= best_thresh
            tp = (ps & storm_mask).sum()
            fp = (ps & ~storm_mask).sum()
            fn = (~ps & storm_mask).sum()
            pr = tp / max(tp + fp, 1)
            rc = tp / n_storm
            f1 = 2 * pr * rc / max(pr + rc, 1e-8)
            print(f"\n  Storm Detection (threshold={best_thresh:.1f}):")
            print(f"    Detected:    {tp}/{n_storm} ({100*rc:.0f}%)")
            print(f"    False alarms: {fp}")
            print(f"    Precision:   {100*pr:.0f}%")
            print(f"    Recall:      {100*rc:.0f}%")
            print(f"    F1:          {f1:.3f}")
        print()

    # Use smoothed for final report
    predicted = smoothed

    # Top 15
    print("=" * 70)
    print("  Top 15 Most Active Periods in 2025")
    print("=" * 70)
    print(f"  {'Date':>20}  {'Actual':>8}  {'Pred':>8}  {'Error':>7}  {'Level':>8}")
    top = np.argsort(actual)[-15:][::-1]
    for i in top:
        err = abs(actual[i] - predicted[i])
        lvl = "STRONG" if actual[i] >= 7 else "STORM" if actual[i] >= 5 else "active" if actual[i] >= 4 else "quiet"
        print(f"  {test_dates[i].strftime('%Y-%m-%d %H:%M'):>20}  {actual[i]:8.1f}  {predicted[i]:8.2f}  {err:7.2f}  {lvl:>8}")

    # Full version comparison
    print(f"\n{'='*70}")
    print(f"  Version Comparison (all on 2025 test set)")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'v1':>8} {'v2':>8} {'v3':>8} {'v4':>8}")
    print(f"  {'Parameters':<25} {'29,633':>8} {'3,585':>8} {'3,681':>8} {n_params:>8,}")
    print(f"  {'Train data':<25} {'5 yr':>8} {'10 yr':>8} {'60 yr':>8} {'60 yr':>8}")
    print(f"  {'Features':<25} {'6':>8} {'11':>8} {'14':>8} {len(all_feat_names):>8}")
    print(f"  {'Seq length':<25} {'24h':>8} {'24h':>8} {'48h':>8} {'72h':>8}")
    print(f"  {'Ensemble':<25} {'1':>8} {'1':>8} {'3':>8} {'5':>8}")
    print(f"  {'Hidden':<25} {'32':>8} {'32':>8} {'32':>8} {'96':>8}")
    print(f"  {'Layers':<25} {'2':>8} {'1':>8} {'1':>8} {'2':>8}")

    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    corr = np.corrcoef(actual, predicted)[0, 1]
    print(f"  {'Test MSE':<25} {'0.859':>8} {'0.398':>8} {'0.280':>8} {mse:>8.3f}")
    print(f"  {'Test MAE':<25} {'0.730':>8} {'0.493':>8} {'0.356':>8} {mae:>8.3f}")
    print(f"  {'Correlation':<25} {'0.743':>8} {'0.886':>8} {'0.926':>8} {corr:>8.3f}")

    # Speed
    import time
    x1 = X_te_t[:1]
    models[0].eval()
    for _ in range(10):
        models[0](x1)
    t0 = time.perf_counter()
    for _ in range(1000):
        models[0](x1)
    us = (time.perf_counter() - t0) / 1000 * 1e6
    print(f"\n  Inference: {us:.0f} us (single) | {us*N_ENSEMBLE:.0f} us (ensemble)")
    print(f"  Total params: {n_params:,} x {N_ENSEMBLE} = {n_params*N_ENSEMBLE:,}")
    print()


if __name__ == "__main__":
    main()
