# LNN — Liquid Neural Network for Geomagnetic Storm Prediction

A Closed-form Continuous-depth (CfC) Liquid Neural Network that predicts geomagnetic storms from NASA solar wind satellite data.

The model learns from 60 years of space weather data (1963-2024) and predicts the Kp geomagnetic storm index 1 hour ahead — validated against real 2025 storm events.

## How It Works

```
DSCOVR Satellite (1.5M km from Earth)          Earth
        |                                         |
  Measures solar wind                       Kp storm index
  (magnetic field, speed,                   (0-9 severity)
   density, temperature)                         |
        |                                         |
        +--- 30-60 min travel time -------------->+
        |                                         |
   [LNN INPUT: 72h of solar wind] -----> [OUTPUT: Kp 1h ahead]
```

The solar wind hits the DSCOVR satellite before it hits Earth. The LNN learns this relationship from historical data and predicts storms before they arrive.

## The Core Math

Each CfC neuron follows a continuous-time ODE with a closed-form solution:

```python
gate = exp(-dt * sigmoid(B) / tau)
h_new = (1 - gate) * A + gate * h_old
```

- `dt` = actual elapsed time between observations (handles irregular data natively)
- `tau` = learnable time constant (some neurons react fast, some slow)
- `A` = target state, `B` = approach speed (both learned from input + hidden state)

This is what makes it "liquid" — the hidden state flows continuously through time.

## Results

Trained on 1963-2024, tested on 2025 (data the model never saw):

### v3 Results (latest validated)

| Metric | Value |
|--------|-------|
| Correlation | 0.926 |
| MAE | 0.36 Kp (on 0-9 scale) |
| Storm detection | 78% recall, 81% precision |
| False alarms | 84 |
| F1 Score | 0.790 |
| Model size | 3,681 parameters |

### Predictions on Real 2025 Storms

| Date | Actual Kp | Predicted | Error |
|------|-----------|-----------|-------|
| Nov 12 2025, 05:00 | 8.7 (G4 severe) | 8.40 | 0.30 |
| Nov 12 2025, 11:00 | 7.7 (G3 strong) | 7.69 | 0.01 |
| Jun 1 2025, 08:00 | 7.7 (G3 strong) | 7.58 | 0.12 |
| Apr 16 2025, 20:00 | 7.7 (G3 strong) | 7.53 | 0.17 |

### Version Progression

| Metric | v1 | v2 | v3 | v4 |
|--------|-----|-----|-----|-----|
| Parameters | 29,633 | 3,585 | 3,681 | ~75,000 |
| Train data | 5 yrs | 10 yrs | 60 yrs | 60 yrs |
| Features | 6 | 11 | 14 | 33 |
| Seq length | 24h | 24h | 48h | 72h |
| Ensemble | 1 | 1 | 3 | 5 |
| MSE | 0.859 | 0.398 | 0.280 | pending |
| MAE | 0.730 | 0.493 | 0.356 | pending |
| Correlation | 0.743 | 0.886 | 0.926 | pending |

## Data

All data comes from NASA's OMNI2 hourly dataset — 63 years of solar wind measurements combined with geomagnetic indices. The training scripts auto-download the data from NASA servers on first run.

### 16 Raw Features (from OMNI2)

| Feature | OMNI2 Col | What it measures |
|---------|-----------|-----------------|
| bt | 8 | Total magnetic field strength (nT) |
| bx_gsm | 12 | Magnetic field, sun-earth direction (nT) |
| by_gsm | 15 | Magnetic field, dusk-dawn direction (nT) |
| bz_gsm | 16 | Magnetic field, north-south (nT) — #1 storm driver |
| speed | 24 | Solar wind speed (km/s) |
| density | 23 | Proton density (N/cm3) |
| temp | 22 | Proton temperature (K) |
| pressure | 28 | Flow pressure (nPa) |
| electric_field | 35 | Electric field (mV/m) |
| plasma_beta | 36 | Plasma/magnetic pressure ratio |
| alfven_mach | 37 | Alfven Mach number |
| mach_magneto | 54 | Magnetosonic Mach number |
| sunspot | 39 | Sunspot number (solar cycle phase) |
| f107 | 50 | F10.7 solar radio flux (solar activity) |
| dst | 40 | Dst index (nT) — hourly storm indicator |
| ae | 41 | AE auroral electrojet (nT) — substorm indicator |

### 17 Engineered Features

Rates of change (d_bz, d_bt, d_speed, d_density, d_dst), rolling means (bz_3h, bt_6h, speed_6h, density_3h), physics-based coupling (epsilon = v * Bz_south, IMF clock angle), and autoregressive lags (kp_lag1/3/6, dst_lag1/3, ae_lag1).

### Target

Kp index (0-9 scale), 1 hour ahead. Kp >= 5 = geomagnetic storm.

## Project Structure

```
LNN/
├── lnn/
│   ├── __init__.py
│   ├── cell.py              # CfC cell — the core LNN math
│   └── model.py             # LiquidNet sequence model
├── data/                    # Auto-downloaded NASA OMNI2 files (gitignored)
├── train.py                 # v1: synthetic sine wave demo
├── train_solar.py           # v2: 10yr data, basic training
├── train_solar_v3.py        # v3: 60yr, ensemble of 3, 0.926 correlation
├── train_solar_v4.py        # v4: 33 features, 96 hidden, ensemble of 5
├── FUTURE.md                # Future project ideas
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/bhavanishankar25/LNN.git
cd LNN
pip install torch

# Run the synthetic demo (30 seconds, no data download)
python3 train.py

# Run the real NASA solar wind model (auto-downloads data)
python3 -u train_solar_v4.py
```

### On a GPU cluster (recommended for v4)

```bash
python3 -u train_solar_v4.py | tee v4_results.txt
```

Estimated time: ~30-60 min on GPU, ~8-12 hours on CPU/MPS.

## Architecture

The CfC cell (`lnn/cell.py`) is ~30 lines of PyTorch. No external LNN libraries — built from scratch from the underlying ODE math.

```
Input (33 features) → CfC Layer 1 (96 hidden) → CfC Layer 2 (96 hidden) → Linear → Kp prediction
                         ↑                          ↑
                    time-aware gate             time-aware gate
                    (dt-dependent)              (dt-dependent)
```

Each layer's gate depends on `dt` (elapsed time), making the network naturally handle irregular timestamps — unlike Transformers or LSTMs which assume uniform spacing.

## Why Liquid Neural Networks?

1. **Time-aware**: The gate equation includes `dt`, so irregular time gaps are handled natively
2. **Tiny**: 3,681 params (v3) achieves 0.926 correlation — a Transformer would need orders of magnitude more
3. **Fast inference**: Microseconds per prediction (closed-form solution, no ODE solver)
4. **Interpretable**: Each neuron has a learnable time constant (tau) — fast neurons react to sudden changes, slow neurons track trends
