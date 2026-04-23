#!/usr/bin/env python3 -u
"""
Train LiquidNet v3 — all improvements.

Changes from v2:
  1. ALL data: 1963-2022 train, 2023-2024 val, 2025 test (63 individual files)
  2. Autoregressive: past Kp values fed as input features
  3. 48h sequences instead of 24h
  4. Proper val split (early stopping on val, untouched test)
  5. Huber loss with storm weighting
  6. Ensemble of 3 models
  7. Calibrated storm threshold from validation set

Usage:
    python3 -u train_solar_v3.py
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

# Force unbuffered output
print = lambda *a, **k: (sys.stdout.write(" ".join(map(str, a)) + k.get("end", "\n")), sys.stdout.flush())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "data"
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{}.dat"

YEARS_TRAIN = list(range(1963, 2023))  # 60 years
YEARS_VAL = [2023, 2024]
YEARS_TEST = [2025]

SEQ_LEN = 48
LEAD_TIME = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 256
LR = 1e-3
N_ENSEMBLE = 3


# OMNI2 columns
COLS_INPUT = {
    "bt":       (8,  999.9),
    "bz_gsm":   (16, 999.9),
    "speed":    (24, 9999.),
    "density":  (23, 999.9),
    "temp":     (22, 9999999.),
    "pressure": (28, 99.99),
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
        print(f"  All {len(years)} files cached (1963-2025)")
        return

    print(f"  {len(years) - len(to_download)} cached, {len(to_download)} to download")
    for i, year in enumerate(to_download):
        path = os.path.join(DATA_DIR, f"omni2_{year}.dat")
        sys.stdout.write(f"\r  Downloading: {year} ({i+1}/{len(to_download)})   ")
        sys.stdout.flush()
        urllib.request.urlretrieve(OMNI_URL.format(year), path)
    print(f"\r  Downloaded {len(to_download)} files                         ")


def parse(years):
    timestamps, features, kp_vals = [], [], []
    col_indices = [v[0] for v in COLS_INPUT.values()]
    fill_vals = [v[1] for v in COLS_INPUT.values()]

    for year in years:
        path = os.path.join(DATA_DIR, f"omni2_{year}.dat")
        with open(path) as f:
            for line in f:
                p = line.split()
                yr, doy, hr = int(p[0]), int(p[1]), int(p[2])
                timestamps.append(datetime(yr, 1, 1) + timedelta(days=doy - 1, hours=hr))

                row = []
                for ci, fv in zip(col_indices, fill_vals):
                    val = float(p[ci])
                    row.append(np.nan if val >= fv * 0.99 else val)
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
    n = len(features)
    bz = features[:, 1]
    speed = features[:, 2]

    dbz = np.zeros(n, dtype=np.float32)
    dbz[1:] = bz[1:] - bz[:-1]

    dspeed = np.zeros(n, dtype=np.float32)
    dspeed[1:] = speed[1:] - speed[:-1]

    bz_3h = np.convolve(bz, np.ones(3) / 3, mode='same').astype(np.float32)
    speed_6h = np.convolve(speed, np.ones(6) / 6, mode='same').astype(np.float32)

    epsilon = (speed * np.clip(-bz, 0, None)).astype(np.float32)

    kp_lag1 = np.zeros(n, dtype=np.float32)
    kp_lag3 = np.zeros(n, dtype=np.float32)
    kp_lag6 = np.zeros(n, dtype=np.float32)
    kp_lag1[1:] = kp[:-1]
    kp_lag3[3:] = kp[:-3]
    kp_lag6[6:] = kp[:-6]

    return np.column_stack([features, dbz, dspeed, bz_3h, speed_6h, epsilon, kp_lag1, kp_lag3, kp_lag6])


def make_sequences_fast(features, kp, timestamps, seq_len, lead):
    """Vectorized sequence building — 100x faster than a Python loop."""
    n = len(features)
    n_feat = features.shape[1]
    total = n - seq_len - lead + 1

    # Build a NaN mask for features: True where ANY feature is NaN in that row
    row_has_nan = np.any(np.isnan(features), axis=1)
    kp_is_nan = np.isnan(kp)

    # For each possible sequence, check if any row in the window has NaN
    # Use a cumulative sum trick: if cumsum of NaN flags is the same at start and end, no NaN in window
    nan_cumsum = np.cumsum(row_has_nan.astype(np.int32))
    nan_cumsum = np.concatenate([[0], nan_cumsum])

    # Count NaN rows in each window of length seq_len
    window_nan_count = nan_cumsum[seq_len:seq_len + total] - nan_cumsum[:total]

    # Target Kp NaN check
    target_indices = np.arange(total) + seq_len + lead - 1
    target_valid = ~kp_is_nan[target_indices]

    # Valid sequences: no NaN in window AND target is valid
    valid_mask = (window_nan_count == 0) & target_valid
    valid_indices = np.where(valid_mask)[0]

    print(f"  {len(valid_indices):,} valid out of {total:,} possible ({100*len(valid_indices)/total:.0f}%)")

    # Build arrays using fancy indexing
    # X[i] = features[valid_indices[i] : valid_indices[i] + seq_len]
    idx_matrix = valid_indices[:, None] + np.arange(seq_len)[None, :]  # (n_valid, seq_len)
    X = features[idx_matrix]  # (n_valid, seq_len, n_feat)
    Y = kp[target_indices[valid_indices]]  # (n_valid,)

    dates = [timestamps[target_indices[i]] for i in valid_indices]

    return X, Y, dates


# ---------------------------------------------------------------------------
# Storm-weighted Huber loss
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
# Training one model
# ---------------------------------------------------------------------------

def train_one(model, X_train_t, Y_train_t, X_val_t, Y_val_t, seed, device):
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    loss_fn = StormWeightedHuber()
    eval_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n = len(X_train_t)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss, n_batch = 0.0, 0

        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            pred = model(X_train_t[idx])[:, -1, 0]
            loss = loss_fn(pred, Y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = []
                for s in range(0, len(X_val_t), BATCH_SIZE):
                    preds.append(model(X_val_t[s:s + BATCH_SIZE])[:, -1, 0])
                val_loss = eval_fn(torch.cat(preds), Y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 5
                marker = ""

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:4d} | train: {epoch_loss/n_batch:.4f} | val: {val_loss:.4f}{marker}")

            if patience_counter >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"    Best val loss: {best_val_loss:.4f}")
    return model, best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    # --- Download ---
    all_years = YEARS_TRAIN + YEARS_VAL + YEARS_TEST
    print(f"Downloading {len(all_years)} individual OMNI2 files (1963-2025)...")
    download(all_years)
    print()

    # --- Parse ---
    print("Parsing 60 years of training data...")
    train_ts, train_feat, train_kp = parse(YEARS_TRAIN)
    print(f"  Train: {len(train_ts):,} hours ({YEARS_TRAIN[0]}-{YEARS_TRAIN[-1]})")

    print("Parsing validation data...")
    val_ts, val_feat, val_kp = parse(YEARS_VAL)
    print(f"  Val:   {len(val_ts):,} hours ({YEARS_VAL[0]}-{YEARS_VAL[-1]})")

    print("Parsing test data...")
    test_ts, test_feat, test_kp = parse(YEARS_TEST)
    print(f"  Test:  {len(test_ts):,} hours ({YEARS_TEST[0]})")

    # Missing data
    total_miss = 100 * np.isnan(train_feat).mean()
    kp_miss = 100 * np.isnan(train_kp).mean()
    print(f"\n  Missing data: {total_miss:.1f}% features, {kp_miss:.1f}% Kp")
    for i, name in enumerate(FEAT_NAMES):
        pct = 100 * np.isnan(train_feat[:, i]).mean()
        if pct > 1:
            print(f"  {name}: {pct:.1f}% missing")
    print()

    # --- Clean ---
    print("Cleaning & forward-filling...")
    train_feat = fill_forward(train_feat)
    val_feat = fill_forward(val_feat)
    test_feat = fill_forward(test_feat)
    train_kp = fill_forward(train_kp)
    val_kp = fill_forward(val_kp)
    test_kp = fill_forward(test_kp)

    # Log-scale temperature
    temp_idx = FEAT_NAMES.index("temp")
    for feat in [train_feat, val_feat, test_feat]:
        feat[:, temp_idx] = np.log1p(np.clip(feat[:, temp_idx], 0, None))

    # Engineer features
    print("Engineering features...")
    train_feat = engineer_features(train_feat, train_kp)
    val_feat = engineer_features(val_feat, val_kp)
    test_feat = engineer_features(test_feat, test_kp)
    all_feat_names = FEAT_NAMES + ["d_bz", "d_speed", "bz_3h", "speed_6h", "epsilon", "kp_lag1", "kp_lag3", "kp_lag6"]
    print(f"  {len(all_feat_names)} features: {', '.join(all_feat_names)}")
    print()

    # Normalize
    print("Normalizing...")
    feat_mean = np.nanmean(train_feat, axis=0)
    feat_std = np.nanstd(train_feat, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    train_feat = (train_feat - feat_mean) / feat_std
    val_feat = (val_feat - feat_mean) / feat_std
    test_feat = (test_feat - feat_mean) / feat_std

    # --- Sequences (vectorized) ---
    print("\nBuilding sequences (vectorized)...")
    print("  Train:", end=" ")
    X_train, Y_train, _ = make_sequences_fast(train_feat, train_kp, train_ts, SEQ_LEN, LEAD_TIME)
    print("  Val:", end=" ")
    X_val, Y_val, val_dates = make_sequences_fast(val_feat, val_kp, val_ts, SEQ_LEN, LEAD_TIME)
    print("  Test:", end=" ")
    X_test, Y_test, test_dates = make_sequences_fast(test_feat, test_kp, test_ts, SEQ_LEN, LEAD_TIME)

    storm_train = 100 * (Y_train >= 5).mean()
    storm_test = 100 * (Y_test >= 5).mean()
    print(f"\n  Storm hours (Kp>=5): train {storm_train:.1f}%, test {storm_test:.1f}%")
    print()

    # To GPU/MPS
    print("Moving data to device...")
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)
    print(f"  Train tensor: {X_train_t.shape} ({X_train_t.nbytes / 1e9:.2f} GB)")
    print()

    input_size = train_feat.shape[1]
    n_params = None

    # =======================================================================
    # ENSEMBLE TRAINING
    # =======================================================================
    print(f"{'='*70}")
    print(f"  TRAINING ENSEMBLE OF {N_ENSEMBLE} MODELS")
    print(f"  {input_size} inputs -> {HIDDEN_SIZE} hidden x {NUM_LAYERS} layer")
    print(f"  Dropout: {DROPOUT}, Weight decay: {WEIGHT_DECAY}, Batch: {BATCH_SIZE}")
    print(f"  Sequence: {SEQ_LEN}h -> Kp {LEAD_TIME}h ahead")
    print(f"{'='*70}\n")

    ensemble_models = []
    for m_idx in range(N_ENSEMBLE):
        print(f"  --- Model {m_idx + 1}/{N_ENSEMBLE} (seed={42 + m_idx}) ---")
        model = LiquidNet(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            output_size=1,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {n_params:,}")

        trained, val_loss = train_one(
            model, X_train_t, Y_train_t, X_val_t, Y_val_t,
            seed=42 + m_idx, device=device
        )
        ensemble_models.append(trained)
        print()

    # =======================================================================
    # CALIBRATE STORM THRESHOLD
    # =======================================================================
    print("Calibrating storm threshold on validation set...")
    for m in ensemble_models:
        m.eval()

    with torch.no_grad():
        val_preds_all = []
        for model in ensemble_models:
            preds = []
            for s in range(0, len(X_val_t), BATCH_SIZE):
                preds.append(model(X_val_t[s:s + BATCH_SIZE])[:, -1, 0])
            val_preds_all.append(torch.cat(preds).cpu().numpy())
        val_pred = np.mean(val_preds_all, axis=0)

    val_actual = Y_val[:len(val_pred)]
    val_storm = val_actual >= 5.0

    best_f1, best_thresh = 0, 4.0
    for thresh in np.arange(3.0, 6.0, 0.1):
        ps = val_pred >= thresh
        tp = (ps & val_storm).sum()
        fp = (ps & ~val_storm).sum()
        fn = (~ps & val_storm).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"  Best threshold: {best_thresh:.1f} (F1={best_f1:.3f})")
    print()

    # =======================================================================
    # TEST EVALUATION
    # =======================================================================
    print("=" * 70)
    print("  RESULTS v3: LiquidNet Storm Predictions vs Reality (2025)")
    print("=" * 70)

    with torch.no_grad():
        test_preds_all = []
        for model in ensemble_models:
            preds = []
            for s in range(0, len(X_test_t), BATCH_SIZE):
                preds.append(model(X_test_t[s:s + BATCH_SIZE])[:, -1, 0])
            test_preds_all.append(torch.cat(preds).cpu().numpy())
        predicted = np.mean(test_preds_all, axis=0)

    actual = Y_test[:len(predicted)]

    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    corr = np.corrcoef(actual, predicted)[0, 1]
    print(f"\n  MSE:         {mse:.4f}")
    print(f"  MAE:         {mae:.4f} Kp")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Kp range:    {actual.min():.1f} - {actual.max():.1f}")

    # Storm detection
    print(f"\n  {'='*55}")
    print(f"  Storm Detection (threshold={best_thresh:.1f}, calibrated)")
    print(f"  {'='*55}")
    storm_mask = actual >= 5.0
    n_storm = storm_mask.sum()
    if n_storm > 0:
        ps = predicted >= best_thresh
        tp = (ps & storm_mask).sum()
        fp = (ps & ~storm_mask).sum()
        fn = (~ps & storm_mask).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / n_storm
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  Storm hours in 2025:   {n_storm}")
        print(f"  Correctly detected:    {tp} ({100*rec:.0f}%)")
        print(f"  Missed:                {fn}")
        print(f"  False alarms:          {fp}")
        print(f"  Precision:             {100*prec:.0f}%")
        print(f"  Recall:                {100*rec:.0f}%")
        print(f"  F1 Score:              {f1:.3f}")

    # Top 15
    print(f"\n  {'='*55}")
    print(f"  Top 15 Most Active Periods in 2025")
    print(f"  {'='*55}")
    print(f"  {'Date':>20}  {'Actual Kp':>10}  {'Predicted':>10}  {'Error':>7}  {'Level':>8}")
    top = np.argsort(actual)[-15:][::-1]
    for i in top:
        err = abs(actual[i] - predicted[i])
        level = "STRONG" if actual[i] >= 7 else "STORM" if actual[i] >= 5 else "active" if actual[i] >= 4 else "quiet"
        print(f"  {test_dates[i].strftime('%Y-%m-%d %H:%M'):>20}  {actual[i]:10.1f}  {predicted[i]:10.2f}  {err:7.2f}  {level:>8}")

    # Comparison
    print(f"\n  {'='*55}")
    print(f"  Version Comparison")
    print(f"  {'='*55}")
    print(f"  {'Metric':<25} {'v1':>10} {'v2':>10} {'v3':>10}")
    print(f"  {'Parameters':<25} {'29,633':>10} {'3,585':>10} {n_params:>10,}")
    print(f"  {'Train data':<25} {'5 yrs':>10} {'10 yrs':>10} {'60 yrs':>10}")
    print(f"  {'Features':<25} {'6':>10} {'11':>10} {'14':>10}")
    print(f"  {'Seq length':<25} {'24h':>10} {'24h':>10} {'48h':>10}")
    print(f"  {'Ensemble':<25} {'no':>10} {'no':>10} {'3':>10}")
    print(f"  {'Test MSE':<25} {'0.8592':>10} {'0.3975':>10} {mse:>10.4f}")
    print(f"  {'Test MAE':<25} {'0.7296':>10} {'0.4932':>10} {mae:>10.4f}")
    print(f"  {'Correlation':<25} {'0.743':>10} {'0.886':>10} {corr:>10.3f}")

    # Speed
    import time
    x1 = X_test_t[:1]
    ensemble_models[0].eval()
    for _ in range(10):
        ensemble_models[0](x1)
    t0 = time.perf_counter()
    for _ in range(1000):
        ensemble_models[0](x1)
    elapsed = time.perf_counter() - t0
    us = elapsed / 1000 * 1e6
    print(f"\n  Inference (single): {us:.0f} us | ensemble: {us * N_ENSEMBLE:.0f} us")
    print(f"  Model: {n_params:,} params x {N_ENSEMBLE} = {n_params * N_ENSEMBLE:,} total")
    print()


if __name__ == "__main__":
    main()
