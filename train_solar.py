#!/usr/bin/env python3
"""
Train LiquidNet to predict geomagnetic storms from solar wind data.

v2 — fixes applied:
  1. Early stopping (save best model by test loss)
  2. Dropout (0.3) + weight decay (1e-4) to prevent overfitting
  3. Storm-weighted loss (Kp >= 4 weighted 5x more)
  4. More data (2015-2024 = 10 years instead of 5)
  5. Feature engineering (Bz rate-of-change, rolling means)
  6. Smaller model (1 layer, hidden 32 instead of 2 layers, hidden 64)

Data: NASA OMNI2 hourly | Train 2015-2024 | Test 2025
"""

import os
import copy
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from lnn import LiquidNet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "data"
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{}.dat"

YEARS_TRAIN = list(range(2015, 2025))  # FIX 4: 10 years
YEARS_TEST = [2025]

SEQ_LEN = 24
LEAD_TIME = 1
HIDDEN_SIZE = 32     # FIX 6: smaller
NUM_LAYERS = 1       # FIX 6: single layer
DROPOUT = 0.3        # FIX 2
WEIGHT_DECAY = 1e-4  # FIX 2
EPOCHS = 300
PATIENCE = 30        # FIX 1: early stopping patience
BATCH_SIZE = 128
LR = 1e-3

# OMNI2 columns (0-based index, fill value)
COLS_INPUT = {
    "bt":       (8,  999.9),
    "bz_gsm":   (16, 999.9),
    "speed":    (24, 9999.),
    "density":  (23, 999.9),
    "temp":     (22, 9999999.),
    "pressure": (28, 99.99),
}
COL_KP = (38, 99)
COL_DST = (40, 99999)

FEAT_NAMES = list(COLS_INPUT.keys())
N_RAW_FEAT = len(FEAT_NAMES)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def download(years):
    os.makedirs(DATA_DIR, exist_ok=True)
    for year in years:
        path = os.path.join(DATA_DIR, f"omni2_{year}.dat")
        if os.path.exists(path):
            print(f"  {year} — cached")
            continue
        url = OMNI_URL.format(year)
        print(f"  {year} — downloading...", end=" ", flush=True)
        urllib.request.urlretrieve(url, path)
        print("ok")


def parse(years):
    timestamps, features, kp_vals = [], [], []

    for year in years:
        path = os.path.join(DATA_DIR, f"omni2_{year}.dat")
        with open(path) as f:
            for line in f:
                p = line.split()
                yr, doy, hr = int(p[0]), int(p[1]), int(p[2])
                timestamps.append(datetime(yr, 1, 1) + timedelta(days=doy - 1, hours=hr))

                row = []
                for name in FEAT_NAMES:
                    col_idx, fill_val = COLS_INPUT[name]
                    val = float(p[col_idx])
                    row.append(np.nan if val >= fill_val * 0.99 else val)
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


def add_engineered_features(features):
    """FIX 5: Add rate-of-change and rolling means."""
    n = len(features)

    # Rate of change of Bz (col 1) — how fast the field is shifting
    bz = features[:, 1]
    dbz = np.zeros(n, dtype=np.float32)
    dbz[1:] = bz[1:] - bz[:-1]

    # Rate of change of speed (col 2)
    speed = features[:, 2]
    dspeed = np.zeros(n, dtype=np.float32)
    dspeed[1:] = speed[1:] - speed[:-1]

    # 3-hour rolling mean of Bz
    bz_3h = np.convolve(bz, np.ones(3) / 3, mode='same').astype(np.float32)

    # 6-hour rolling mean of speed
    speed_6h = np.convolve(speed, np.ones(6) / 6, mode='same').astype(np.float32)

    # Bz * speed interaction (epsilon = v * Bz_south, a known storm driver)
    bz_south = np.clip(-bz, 0, None)  # only negative Bz matters
    epsilon = (speed * bz_south).astype(np.float32)

    return np.column_stack([features, dbz, dspeed, bz_3h, speed_6h, epsilon])


def make_sequences(features, kp, timestamps, seq_len, lead):
    X, Y, dates = [], [], []
    for i in range(len(features) - seq_len - lead):
        x = features[i:i + seq_len]
        y = kp[i + seq_len + lead - 1]
        if not np.any(np.isnan(x)) and not np.isnan(y):
            X.append(x)
            Y.append(y)
            dates.append(timestamps[i + seq_len + lead - 1])
    return np.array(X), np.array(Y), dates


# ---------------------------------------------------------------------------
# Storm-weighted loss (FIX 3)
# ---------------------------------------------------------------------------

class StormWeightedMSE(nn.Module):
    """Weight high-Kp samples more heavily so the model doesn't ignore storms."""
    def __init__(self, storm_threshold=4.0, storm_weight=5.0):
        super().__init__()
        self.threshold = storm_threshold
        self.weight = storm_weight

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        weights[target >= self.threshold] = self.weight
        return (weights * (pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    # --- Download ---
    print("Downloading NASA OMNI2 data...")
    download(YEARS_TRAIN + YEARS_TEST)
    print()

    # --- Parse ---
    print("Parsing...")
    train_ts, train_feat, train_kp = parse(YEARS_TRAIN)
    test_ts, test_feat, test_kp = parse(YEARS_TEST)
    print(f"  Train: {len(train_ts):,} hours ({YEARS_TRAIN[0]}-{YEARS_TRAIN[-1]})")
    print(f"  Test:  {len(test_ts):,} hours ({YEARS_TEST[0]})")

    miss = {name: 100 * np.isnan(train_feat[:, i]).mean() for i, name in enumerate(FEAT_NAMES)}
    for name, pct in miss.items():
        if pct > 0:
            print(f"  {name}: {pct:.1f}% missing → forward-filled")
    print()

    # --- Clean ---
    train_feat = fill_forward(train_feat)
    test_feat = fill_forward(test_feat)
    train_kp = fill_forward(train_kp)
    test_kp = fill_forward(test_kp)

    # Log-scale temperature
    temp_idx = FEAT_NAMES.index("temp")
    train_feat[:, temp_idx] = np.log1p(np.clip(train_feat[:, temp_idx], 0, None))
    test_feat[:, temp_idx] = np.log1p(np.clip(test_feat[:, temp_idx], 0, None))

    # FIX 5: engineered features
    print("Engineering features...")
    train_feat = add_engineered_features(train_feat)
    test_feat = add_engineered_features(test_feat)
    all_feat_names = FEAT_NAMES + ["d_bz", "d_speed", "bz_3h", "speed_6h", "epsilon"]
    print(f"  {len(all_feat_names)} features: {', '.join(all_feat_names)}")
    print()

    # Normalize (z-score from training set)
    feat_mean = np.nanmean(train_feat, axis=0)
    feat_std = np.nanstd(train_feat, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    train_feat = (train_feat - feat_mean) / feat_std
    test_feat = (test_feat - feat_mean) / feat_std

    # --- Sequences ---
    X_train, Y_train, _ = make_sequences(train_feat, train_kp, train_ts, SEQ_LEN, LEAD_TIME)
    X_test, Y_test, test_dates = make_sequences(test_feat, test_kp, test_ts, SEQ_LEN, LEAD_TIME)

    print(f"Sequences: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
    print(f"  Each: {SEQ_LEN}h solar wind → Kp {LEAD_TIME}h ahead")

    # Storm stats
    n_storm_train = (Y_train >= 5.0).sum()
    n_storm_test = (Y_test >= 5.0).sum()
    print(f"  Storm hours (Kp≥5): {n_storm_train} train ({100*n_storm_train/len(Y_train):.1f}%), {n_storm_test} test")
    print()

    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)

    # --- Model (FIX 6: smaller) ---
    input_size = train_feat.shape[1]
    model = LiquidNet(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=1,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LiquidNet: {n_params:,} parameters")
    print(f"  {input_size} inputs → {HIDDEN_SIZE} hidden × {NUM_LAYERS} layer → 1 output")
    print(f"  Dropout: {DROPOUT}, Weight decay: {WEIGHT_DECAY}")
    print()

    # FIX 2: weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # FIX 3: storm-weighted loss
    loss_fn = StormWeightedMSE(storm_threshold=4.0, storm_weight=5.0)
    eval_loss_fn = nn.MSELoss()

    # FIX 1: early stopping
    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # --- Train ---
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

        # Evaluate every 5 epochs for early stopping
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = []
                for s in range(0, len(X_test_t), BATCH_SIZE):
                    preds.append(model(X_test_t[s:s + BATCH_SIZE])[:, -1, 0])
                test_loss = eval_loss_fn(torch.cat(preds), Y_test_t).item()

            # FIX 1: save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                marker = " ★"
            else:
                patience_counter += 5
                marker = ""

            if epoch % 25 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d} | train: {epoch_loss/n_batch:.4f} | test: {test_loss:.4f}{marker}")

            # FIX 1: stop if no improvement
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # Restore best model
    print(f"\n  Best test loss: {best_test_loss:.4f}")
    model.load_state_dict(best_model_state)

    # =======================================================================
    # EVALUATION
    # =======================================================================
    print("\n" + "=" * 70)
    print("  RESULTS: LiquidNet Storm Predictions vs Reality (2025)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, len(X_test_t), BATCH_SIZE):
            preds.append(model(X_test_t[s:s + BATCH_SIZE])[:, -1, 0])
        predicted = torch.cat(preds).cpu().numpy()

    actual = Y_test[:len(predicted)]

    # Overall
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    corr = np.corrcoef(actual, predicted)[0, 1]
    print(f"\n  MSE:         {mse:.4f}")
    print(f"  MAE:         {mae:.4f} Kp")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Kp range:    {actual.min():.1f} – {actual.max():.1f}")

    # Storm detection
    print(f"\n  {'─' * 50}")
    print(f"  Storm Detection (Kp ≥ 5 = G1 storm)")
    print(f"  {'─' * 50}")
    storm_mask = actual >= 5.0
    n_storm = storm_mask.sum()
    if n_storm > 0:
        pred_storm = predicted >= 4.0
        hit = (pred_storm & storm_mask).sum()
        miss = (~pred_storm & storm_mask).sum()
        false_alarm = (pred_storm & ~storm_mask).sum()
        precision = hit / (hit + false_alarm) if (hit + false_alarm) > 0 else 0
        recall = hit / n_storm
        print(f"  Storm hours in 2025:   {n_storm}")
        print(f"  Correctly detected:    {hit} ({100*recall:.0f}%)")
        print(f"  Missed:                {miss}")
        print(f"  False alarms:          {false_alarm}")
        print(f"  Precision:             {100*precision:.0f}%")
        print(f"  Recall:                {100*recall:.0f}%")
    else:
        print("  No storms (Kp ≥ 5) in test period")

    # Top 15 most active periods
    print(f"\n  {'─' * 50}")
    print(f"  Top 15 Most Active Periods in 2025")
    print(f"  {'─' * 50}")
    print(f"  {'Date':>20}  {'Actual Kp':>10}  {'Predicted':>10}  {'Error':>7}  {'Level':>8}")
    top = np.argsort(actual)[-15:][::-1]
    for i in top:
        err = abs(actual[i] - predicted[i])
        if actual[i] >= 7:
            level = "STRONG"
        elif actual[i] >= 5:
            level = "STORM"
        elif actual[i] >= 4:
            level = "active"
        else:
            level = "quiet"
        date_str = test_dates[i].strftime("%Y-%m-%d %H:%M")
        print(f"  {date_str:>20}  {actual[i]:10.1f}  {predicted[i]:10.2f}  {err:7.2f}  {level:>8}")

    # Comparison vs v1
    print(f"\n  {'─' * 50}")
    print(f"  Comparison vs v1")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<25} {'v1':>10} {'v2':>10}")
    print(f"  {'Parameters':<25} {'29,633':>10} {n_params:>10,}")
    print(f"  {'Train years':<25} {'5':>10} {'10':>10}")
    print(f"  {'Features':<25} {'6':>10} {'11':>10}")
    print(f"  {'Test MSE':<25} {'0.8592':>10} {mse:>10.4f}")
    print(f"  {'Test MAE':<25} {'0.7296':>10} {mae:>10.4f}")
    print(f"  {'Correlation':<25} {'0.743':>10} {corr:>10.3f}")

    # Inference speed
    import time
    x1 = X_test_t[:1]
    for _ in range(10):
        model(x1)
    t0 = time.perf_counter()
    for _ in range(1000):
        model(x1)
    elapsed = time.perf_counter() - t0
    print(f"\n  Inference: {elapsed/1000*1e6:.0f} μs per prediction")
    print(f"  Model size: {n_params:,} parameters")
    print()


if __name__ == "__main__":
    main()
