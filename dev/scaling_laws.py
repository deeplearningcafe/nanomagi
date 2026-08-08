import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# based on https://github.com/karpathy/nanochat/blob/master/dev/scaling_analysis.ipynb
def plot_scaling_laws(
    csv_path_or_df,
    target_col: str = "val_bpb",
    param_col: str = "total_params",
    flops_col: str = "target_flops",
    tokens_col: str = "tokens_trained",
    save_path: str = None,
):
    """Plots IsoFLOP curves, Optimal Model Size, and Optimal Training Tokens from

    scaling experiment results.

    Args:
        csv_path_or_df (str or pd.DataFrame): Path to CSV or a loaded
          DataFrame.
        target_col (str): Column name for the metric to minimize (default:
          'val_bpb').
        param_col (str): Parameter column to use (default: 'total_params', could
          be 'non_embed_params').
        flops_col (str): Column name for target FLOPs budget (default:
          'target_flops').
        tokens_col (str): Column name for tokens trained (default:
          'tokens_trained').
        save_path (str, optional): Path to save the output plot figure.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()

    if df.empty:
        print(f"Error: No valid data found for column '{target_col}'.")
        return

    flops_budgets = sorted(df[flops_col].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: IsoFLOP Curves
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(flops_budgets)))
    optimal_points = []

    for flops, color in zip(flops_budgets, colors):
        subset = df[df[flops_col] == flops].sort_values(param_col)
        if subset.empty:
            continue

        ax1.plot(
            subset[param_col],
            subset[target_col],
            "o",
            color=color,
            label=f"{flops:.0e}",
            markersize=8,
        )

        # Fit quadratic curve in log-space: y = a*(log10 N)^2 + b*(log10 N) + c
        log_params = np.log10(subset[param_col])
        coeffs = np.polyfit(log_params, subset[target_col], 2)
        a, b, c = coeffs

        log_fit_x = np.linspace(
            log_params.min() - 0.1, log_params.max() + 0.1, 100
        )
        fit_y = a * log_fit_x**2 + b * log_fit_x + c
        ax1.plot(10**log_fit_x, fit_y, "--", color=color, linewidth=2)

        # Find parabola minimum: d/dx(ax^2 + bx + c) = 0 => x = -b/(2a)
        if a > 0:
            log_opt = -b / (2 * a)
            opt_params = 10**log_opt
            opt_val = a * log_opt**2 + b * log_opt + c

            # Mark fitted minimum with a star
            ax1.scatter(
                [opt_params],
                [opt_val],
                s=150,
                color=color,
                zorder=5,
                edgecolors="black",
                linewidths=2,
                marker="*",
            )

            # Interpolate optimal tokens corresponding to optimal parameters
            opt_tokens = np.interp(log_opt, log_params, subset[tokens_col])
            optimal_points.append(
                {
                    "flops": flops,
                    "params": opt_params,
                    "tokens": opt_tokens,
                    "val": opt_val,
                }
            )
        else:
            best_idx = subset[target_col].idxmin()
            best_row = subset.loc[best_idx]
            ax1.scatter(
                [best_row[param_col]],
                [best_row[target_col]],
                s=150,
                color=color,
                zorder=5,
                edgecolors="black",
                linewidths=2,
            )
            optimal_points.append(
                {
                    "flops": flops,
                    "params": best_row[param_col],
                    "tokens": best_row[tokens_col],
                    "val": best_row[target_col],
                }
            )

    ax1.set_xscale("log")
    ax1.set_xlabel("Parameters")

    metric_name = target_col.replace("val_", "").upper()
    ax1.set_ylabel(f"Validation Loss ({metric_name})")
    ax1.set_title("IsoFLOP Curves")
    ax1.legend(title="FLOPs", loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    opt_df = pd.DataFrame(optimal_points)

    # Subplot 2: Optimal Model Size vs Compute
    ax2 = axes[1]
    ax2.loglog(
        opt_df["flops"], opt_df["params"], "o", markersize=10, color="#2ecc71"
    )
    ax2.set_xlabel("FLOPs")
    ax2.set_ylabel("Optimal Parameters")
    ax2.set_title("Optimal Model Size")
    ax2.grid(True, linestyle="--", alpha=0.3)

    if len(opt_df) >= 2:
        log_f = np.log10(opt_df["flops"])
        log_p = np.log10(opt_df["params"])
        slope, intercept = np.polyfit(log_f, log_p, 1)

        fit_f = np.logspace(log_f.min() - 0.2, log_f.max() + 0.2, 100)
        fit_p = 10 ** (intercept + slope * np.log10(fit_f))
        ax2.plot(
            fit_f, fit_p, "r--", alpha=0.8, label=f"N \u221d C^{slope:.2f}"
        )
        ax2.legend()

    # Subplot 3: Optimal Training Tokens vs Compute
    ax3 = axes[2]
    ax3.loglog(
        opt_df["flops"], opt_df["tokens"], "o", markersize=10, color="#e74c3c"
    )
    ax3.set_xlabel("FLOPs")
    ax3.set_ylabel("Optimal Tokens")
    ax3.set_title("Optimal Training Tokens")
    ax3.grid(True, linestyle="--", alpha=0.3)

    if len(opt_df) >= 2:
        log_f = np.log10(opt_df["flops"])
        log_t = np.log10(opt_df["tokens"])
        slope, intercept = np.polyfit(log_f, log_t, 1)

        fit_f = np.logspace(log_f.min() - 0.2, log_f.max() + 0.2, 100)
        fit_t = 10 ** (intercept + slope * np.log10(fit_f))
        ax3.plot(
            fit_f, fit_t, "r--", alpha=0.8, label=f"D \u221d C^{slope:.2f}"
        )
        ax3.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    csv_path = "results/isoflop_results.csv"

    # target_col 'val_bpb', 'val_ppl', or 'val_loss'
    # param_col 'total_params' or 'non_embed_params'
    if os.path.exists(csv_path):
        plot_scaling_laws(
            csv_path_or_df=csv_path,
            target_col="val_bpb",
            param_col="total_params", 
            save_path="results/scaling_laws.png",
        )
    else:
        print(f"File not found: '{csv_path}'. Please update the path in main().")


if __name__ == "__main__":
    main()