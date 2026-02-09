"""Pre-train a Neural SDE on historical price data.

Usage (MLE, minute-resolution):
    cd /Users/matthew/Projects/quantammsim-synthetic-paths && \\
      source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-synth && \\
      python scripts/fit_price_sde.py --method mle --tokens ETH USDC --output models/eth_usdc_sde.eqx

Usage (Sig-W1, daily-resolution):
    cd /Users/matthew/Projects/quantammsim-synthetic-paths && \\
      source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-synth && \\
      python scripts/fit_price_sde.py --method sigw1 --tokens ETH BTC --output models/eth_btc_sde_v4.eqx

This script:
1. Loads minute-resolution parquet data for the specified tokens.
2. Fits a Neural SDE via MLE (minute) or Sig-W1 (daily) training.
3. Saves the trained model to the specified path.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from quantammsim.synthetic.model import NeuralSDE, save_sde, LatentNeuralSDE, save_latent_sde
from quantammsim.synthetic.training import fit_neural_sde, fit_neural_sde_sigw1, fit_latent_sde_sigw1, compute_daily_log_prices
from quantammsim.utils.data_processing.historic_data_utils import get_historic_parquet_data


def main():
    parser = argparse.ArgumentParser(description="Fit a Neural SDE to historical price data")
    parser.add_argument(
        "--tokens", nargs="+", required=True,
        help="Token tickers (e.g. ETH USDC BTC)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the trained model (.eqx file)"
    )
    parser.add_argument(
        "--method", type=str, default="mle", choices=["mle", "sigw1"],
        help="Training method: 'mle' (minute-resolution MLE) or 'sigw1' (daily Sig-W1)"
    )
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width")
    parser.add_argument("--diagonal-only", action="store_true", help="Use diagonal diffusion")
    parser.add_argument("--learn-drift", action="store_true", help="Learn drift (MLE only; Sig-W1 always learns drift)")
    parser.add_argument("--n-epochs", type=int, default=2000, help="Max training epochs/steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=8192, help="Minibatch size (MLE: transitions, Sig-W1: initial conditions)")
    parser.add_argument(
        "--max-train-transitions", type=int, default=50_000,
        help="(MLE only) Max training transitions per epoch. 0 = no cap."
    )
    # Sig-W1 specific args
    parser.add_argument(
        "--window-lens", type=int, nargs="+", default=[10, 20, 50],
        help="(Sig-W1) Window lengths in days for multi-scale matching"
    )
    parser.add_argument("--depth", type=int, default=3, help="(Sig-W1) Signature truncation depth")
    parser.add_argument("--mc-samples", type=int, default=200, help="(Sig-W1) Monte Carlo samples per initial condition")
    parser.add_argument(
        "--augmentation", type=str, default="minimal", choices=["minimal", "standard"],
        help="(Sig-W1) Path augmentation before signature"
    )
    parser.add_argument(
        "--architecture", type=str, default="standard", choices=["standard", "latent"],
        help="SDE architecture: 'standard' (original NeuralSDE) or 'latent' (LatentNeuralSDE with hidden dims)"
    )
    parser.add_argument("--n-hidden", type=int, default=4, help="(Latent only) Number of hidden latent dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-root", type=str, default=None, help="Root dir for parquet files")
    args = parser.parse_args()

    print(f"Loading data for tokens: {args.tokens}")
    price_df = get_historic_parquet_data(args.tokens, cols=["close"], root=args.data_root)

    # Extract close prices as numpy array (T_minutes, n_assets)
    close_cols = [f"close_{t}" for t in args.tokens]
    minute_prices = price_df[close_cols].values.astype(np.float64)

    # Drop any leading/trailing NaN rows
    valid_mask = ~np.any(np.isnan(minute_prices), axis=1)
    first_valid = np.argmax(valid_mask)
    last_valid = len(valid_mask) - np.argmax(valid_mask[::-1])
    minute_prices = minute_prices[first_valid:last_valid]

    n_assets = len(args.tokens)
    n_days = minute_prices.shape[0] // 1440
    print(f"Price data shape: {minute_prices.shape} ({n_days} days, {minute_prices.shape[0]:,} minutes)")
    print(f"Training method: {args.method}")

    key = jax.random.PRNGKey(args.seed)

    if args.method == "mle":
        trained_sde, loss_history = fit_neural_sde(
            jnp.array(minute_prices),
            n_assets=n_assets,
            key=key,
            n_epochs=args.n_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            diagonal_only=args.diagonal_only,
            learn_drift=args.learn_drift,
            val_fraction=args.val_fraction,
            patience=args.patience,
            batch_size=args.batch_size,
            max_train_transitions=args.max_train_transitions,
        )
        save_sde(trained_sde, args.output)
        print(f"\nModel saved to {args.output}")
        print(f"Architecture: hidden_dim={args.hidden_dim}, diagonal_only={args.diagonal_only}, "
              f"learn_drift={args.learn_drift}")
        print(f"Final train NLL: {loss_history[-1][0]:.4f}, val NLL: {loss_history[-1][1]:.4f}")
        print(f"Best val NLL: {min(v for _, v in loss_history):.4f}")

    elif args.method == "sigw1":
        # Default depth for latent architecture is 2 (reduced overfitting)
        depth = args.depth
        if args.architecture == "latent" and depth == 3 and "--depth" not in " ".join(f'--{k}' for k in vars(args)):
            depth = 2

        if args.architecture == "latent":
            trained_sde, loss_history = fit_latent_sde_sigw1(
                jnp.array(minute_prices),
                n_assets=n_assets,
                key=key,
                n_hidden=args.n_hidden,
                hidden_dim=args.hidden_dim,
                window_lens=args.window_lens,
                depth=depth,
                mc_samples=args.mc_samples,
                batch_size=min(args.batch_size, 16),
                augmentation=args.augmentation,
                n_steps=args.n_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                val_fraction=args.val_fraction,
                patience=args.patience,
            )
            save_latent_sde(trained_sde, args.output)
            print(f"\nModel saved to {args.output}")
            print(f"Architecture: latent, n_hidden={args.n_hidden}, latent_dim={n_assets + args.n_hidden}, "
                  f"hidden_dim={args.hidden_dim}")
            print(f"Final train Sig-W1: {loss_history[-1][0]:.6f}, val Sig-W1: {loss_history[-1][1]:.6f}")
            print(f"Best val Sig-W1: {min(v for _, v in loss_history):.6f}")
        else:
            trained_sde, loss_history = fit_neural_sde_sigw1(
                jnp.array(minute_prices),
                n_assets=n_assets,
                key=key,
                hidden_dim=args.hidden_dim,
                diagonal_only=args.diagonal_only,
                window_lens=args.window_lens,
                depth=depth,
                mc_samples=args.mc_samples,
                batch_size=min(args.batch_size, 16),
                augmentation=args.augmentation,
                n_steps=args.n_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                val_fraction=args.val_fraction,
                patience=args.patience,
            )
            save_sde(trained_sde, args.output)
            print(f"\nModel saved to {args.output}")
            print(f"Architecture: standard, hidden_dim={args.hidden_dim}, diagonal_only={args.diagonal_only}, "
                  f"learn_drift=True (always for Sig-W1)")
            print(f"Final train Sig-W1: {loss_history[-1][0]:.6f}, val Sig-W1: {loss_history[-1][1]:.6f}")
            print(f"Best val Sig-W1: {min(v for _, v in loss_history):.6f}")


if __name__ == "__main__":
    main()
