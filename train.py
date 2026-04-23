"""
Train the smallest possible LiquidNet on synthetic time-series data.

Generates a noisy multi-frequency sine wave, trains the network to predict
the next value at each step, and prints a side-by-side comparison at the end.

Usage:
    python train.py                  # regular time steps
    python train.py --irregular      # irregular time steps (LNN's advantage)
"""

import argparse
import torch
import torch.nn as nn
from lnn import LiquidNet


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_samples: int = 256,
    seq_len: int = 64,
    irregular: bool = False,
):
    """Generate noisy sine-wave sequences for next-step prediction."""
    t_regular = torch.linspace(0, 4 * torch.pi, seq_len + 1).unsqueeze(0).expand(n_samples, -1)

    if irregular:
        # Jitter timestamps by up to +/-30% of the regular spacing
        spacing = t_regular[:, 1:2] - t_regular[:, 0:1]
        jitter = 1.0 + 0.6 * (torch.rand(n_samples, seq_len + 1) - 0.5)
        jitter[:, 0] = 1.0  # keep start fixed
        t = t_regular * jitter
        t, _ = torch.sort(t, dim=1)  # ensure monotonic
    else:
        t = t_regular

    # Random frequency and phase per sample for variety
    freq = 0.8 + 0.4 * torch.rand(n_samples, 1)
    phase = 2 * torch.pi * torch.rand(n_samples, 1)

    values = torch.sin(freq * t + phase) + 0.3 * torch.sin(2.7 * freq * t) + 0.05 * torch.randn_like(t)

    x = values[:, :-1].unsqueeze(-1)       # (n, seq_len, 1) — input
    y = values[:, 1:].unsqueeze(-1)         # (n, seq_len, 1) — target (next step)
    dt = (t[:, 1:] - t[:, :-1]).unsqueeze(-1)  # (n, seq_len, 1) — time gaps

    return x, y, dt


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--irregular", action="store_true", help="use irregular time steps")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # --- Data ---
    x_train, y_train, dt_train = generate_data(256, seq_len=64, irregular=args.irregular)
    x_test, y_test, dt_test = generate_data(64, seq_len=64, irregular=args.irregular)
    x_train, y_train, dt_train = x_train.to(device), y_train.to(device), dt_train.to(device)
    x_test, y_test, dt_test = x_test.to(device), y_test.to(device), dt_test.to(device)

    # --- Model ---
    model = LiquidNet(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_layers=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Time steps: {'irregular' if args.irregular else 'regular'}")
    print(f"Training: {x_train.shape[0]} sequences x {x_train.shape[1]} steps")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # --- Train ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(x_train, dt_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_pred = model(x_test, dt_test)
                test_loss = loss_fn(test_pred, y_test)
            print(f"Epoch {epoch:4d} | train loss: {loss.item():.6f} | test loss: {test_loss.item():.6f}")

    # --- Final evaluation ---
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test, dt_test)

    print("\n--- Sample predictions (first test sequence, last 10 steps) ---")
    print(f"{'Step':>5}  {'Actual':>9}  {'Predicted':>9}  {'Error':>9}")
    actual = y_test[0, -10:, 0].cpu()
    predicted = test_pred[0, -10:, 0].cpu()
    for i in range(10):
        err = abs(actual[i].item() - predicted[i].item())
        print(f"{i+55:5d}  {actual[i].item():9.4f}  {predicted[i].item():9.4f}  {err:9.4f}")

    # --- Inference speed ---
    import time
    single_input = x_test[:1].to(device)
    single_dt = dt_test[:1].to(device)

    # Warmup
    for _ in range(10):
        model(single_input, single_dt)

    start = time.perf_counter()
    n_runs = 1000
    for _ in range(n_runs):
        model(single_input, single_dt)
    elapsed = time.perf_counter() - start

    print(f"\nInference: {elapsed/n_runs*1e6:.1f} us per sequence ({n_runs} runs)")
    print(f"That's {elapsed/n_runs*1e6/64:.1f} us per time step")


if __name__ == "__main__":
    main()
