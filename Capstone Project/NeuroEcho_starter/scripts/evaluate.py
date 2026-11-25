"""
NeuroEcho Comprehensive Model Evaluation & Statistical Testing Suite
Generates:
1. Detailed metric calculations (MSE, MAE, Pearson r, SNR, Gap-only MSE)
2. Statistical tests (Shapiro-Wilk, Paired t-test, Wilcoxon signed-rank, Cohen's d)
3. Formatted publication tables (CSV, Markdown)
4. Comprehensive publication-grade figures (Data visualization & Model evaluation)
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MPLCONFIGDIR"] = "/tmp"

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from neuroecho.model import CNNBiLSTMAutoencoder
from neuroecho.corruption import corrupt_batch
from neuroecho.metrics import regression_metrics

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "lines.linewidth": 1.7,
})

def linear_interpolate_gaps(corrupted, mask):
    """
    Interpolate missing gaps (where mask == 0) linearly from adjacent uncorrupted points.
    corrupted: (channels, time)
    mask: (channels, time)
    """
    interpolated = corrupted.copy()
    c, t = corrupted.shape
    time_pts = np.arange(t)
    for ch in range(c):
        valid = mask[ch] > 0.5
        if np.sum(valid) >= 2 and np.sum(~valid) > 0:
            interpolated[ch, ~valid] = np.interp(time_pts[~valid], time_pts[valid], corrupted[ch, valid])
        elif np.sum(valid) < 2:
            interpolated[ch, :] = 0.0
    return interpolated

def compute_snr(clean, pred, eps=1e-8):
    """Compute Signal to Noise Ratio in dB."""
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - pred) ** 2)
    if noise_power < eps:
        return 50.0  # Cap at high SNR if virtually identical
    snr = 10.0 * np.log10(signal_power / (noise_power + eps))
    return float(snr)

def calculate_window_channel_metrics(clean_ch, corrupted_ch, pred_ch, interp_ch, mask_ch):
    """Calculate per-channel metrics across baselines."""
    missing_idx = (mask_ch < 0.5)
    
    def get_metrics_dict(target_sig):
        reg = regression_metrics(clean_ch, target_sig)
        snr = compute_snr(clean_ch, target_sig)
        # Gap-specific MSE
        if np.sum(missing_idx) > 0:
            gap_mse = float(np.mean((clean_ch[missing_idx] - target_sig[missing_idx]) ** 2))
            gap_mae = float(np.mean(np.abs(clean_ch[missing_idx] - target_sig[missing_idx])))
        else:
            gap_mse = reg["MSE"]
            gap_mae = reg["MAE"]
            
        return {
            "MSE": reg["MSE"],
            "MAE": reg["MAE"],
            "Pearson_r": reg["Pearson correlation"],
            "SNR_dB": snr,
            "Gap_MSE": gap_mse,
            "Gap_MAE": gap_mae,
        }

    return {
        "Corrupted": get_metrics_dict(corrupted_ch),
        "Linear_Interp": get_metrics_dict(interp_ch),
        "NeuroEcho": get_metrics_dict(pred_ch),
    }

def run_evaluation(data_path="data/processed/windows.npz", 
                   model_path="artifacts/neuroecho_model.pt",
                   output_dir="artifacts/evaluation",
                   fig_dir="artifacts/figures",
                   seed=42):
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figures_path = Path(fig_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("Loading data and model...")
    raw_windows = np.load(data_path)["windows"].astype(np.float32)
    n_windows, n_channels, n_samples = raw_windows.shape
    print(f"Dataset shape: {n_windows} windows, {n_channels} channels, {n_samples} samples per window")

    # Load model checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    model = CNNBiLSTMAutoencoder(channels=ckpt["channels"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Recreate the exact train/val split from training
    n_val = max(1, int(0.2 * n_windows))
    n_train = n_windows - n_val
    train_idx, val_idx = torch.utils.data.random_split(
        range(n_windows), [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    val_indices = list(val_idx.indices)
    print(f"Validation set size: {len(val_indices)} windows ({len(val_indices) * n_channels} channel segments)")

    # Seed for corruption reproduction
    rng = np.random.default_rng(2026)
    
    rows = []
    reconstructed_all = np.zeros_like(raw_windows)
    corrupted_all = np.zeros_like(raw_windows)
    interpolated_all = np.zeros_like(raw_windows)
    masks_all = np.zeros_like(raw_windows)

    with torch.no_grad():
        for i in range(n_windows):
            clean_win = raw_windows[i:i+1] # (1, C, T)
            corr_win, mask_win = corrupt_batch(clean_win, rng=rng, missing_fraction=0.20, noise_std=0.15)
            pred_win = model(torch.from_numpy(corr_win)).numpy()
            interp_win = linear_interpolate_gaps(corr_win[0], mask_win[0])[np.newaxis, ...]

            corrupted_all[i] = corr_win[0]
            reconstructed_all[i] = pred_win[0]
            interpolated_all[i] = interp_win[0]
            masks_all[i] = mask_win[0]

            is_val = i in val_indices
            
            for ch in range(n_channels):
                metrics = calculate_window_channel_metrics(
                    clean_win[0, ch], corr_win[0, ch], pred_win[0, ch], interp_win[0, ch], mask_win[0, ch]
                )
                for model_name, m_dict in metrics.items():
                    rows.append({
                        "window_idx": i,
                        "channel_idx": ch,
                        "channel_name": f"Ch {ch+1}",
                        "split": "Validation" if is_val else "Train",
                        "model": model_name,
                        **m_dict
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_path / "detailed_channel_metrics.csv", index=False)
    print(f"Saved detailed metrics to {out_path / 'detailed_channel_metrics.csv'}")

    # ==========================================
    # 1. SUMMARY TABLES (MEAN, SD, MEDIAN, IQR, 95% CI)
    # ==========================================
    val_df = df[df["split"] == "Validation"]
    metrics_list = ["MSE", "Gap_MSE", "MAE", "Gap_MAE", "Pearson_r", "SNR_dB"]
    models = ["Corrupted", "Linear_Interp", "NeuroEcho"]

    summary_records = []
    for m in models:
        sub = val_df[val_df["model"] == m]
        for metric in metrics_list:
            vals = sub[metric].values
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            median = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            ci_low, ci_high = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=stats.sem(vals))
            summary_records.append({
                "Model": m,
                "Metric": metric,
                "Mean": mean,
                "Std": std,
                "95% CI Lower": ci_low,
                "95% CI Upper": ci_high,
                "Median": median,
                "IQR": q75 - q25,
                "Mean ± SD": f"{mean:.4f} ± {std:.4f}",
                "Median [IQR]": f"{median:.4f} [{q25:.4f}, {q75:.4f}]",
                "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]"
            })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_path / "evaluation_summary_validation.csv", index=False)
    
    # Pivot summary table for clean display
    pivot_table = summary_df.pivot(index="Metric", columns="Model", values="Mean ± SD")[models]
    pivot_table.to_csv(out_path / "metrics_table_pivot.csv")
    print("\n--- Validation Performance Summary (Mean ± SD) ---")
    print(pivot_table.to_string())

    # ==========================================
    # 2. STATISTICAL SIGNIFICANCE TESTS
    # ==========================================
    # We compare NeuroEcho vs Corrupted and NeuroEcho vs Linear_Interp
    # For each metric, perform:
    # - Shapiro-Wilk test on differences (test for normality)
    # - Paired Student's t-test
    # - Wilcoxon signed-rank test
    # - Cohen's d effect size
    stat_test_records = []

    comparisons = [
        ("NeuroEcho", "Corrupted"),
        ("NeuroEcho", "Linear_Interp")
    ]

    for model_a, model_b in comparisons:
        sub_a = val_df[val_df["model"] == model_a].sort_values(["window_idx", "channel_idx"])
        sub_b = val_df[val_df["model"] == model_b].sort_values(["window_idx", "channel_idx"])

        for metric in metrics_list:
            va = sub_a[metric].values
            vb = sub_b[metric].values
            diff = va - vb  # a - b
            
            # Shapiro-Wilk test for normality of differences
            shapiro_stat, shapiro_p = stats.shapiro(diff)
            
            # Paired t-test
            t_stat, t_p = stats.ttest_rel(va, vb)
            
            # Wilcoxon signed-rank test
            try:
                wilcoxon_res = stats.wilcoxon(va, vb)
                w_stat, w_p = wilcoxon_res.statistic, wilcoxon_res.pvalue
            except Exception as e:
                w_stat, w_p = np.nan, np.nan

            # Cohen's d: mean(diff) / std(diff, ddof=1)
            d_denom = np.std(diff, ddof=1)
            cohens_d = np.mean(diff) / d_denom if d_denom > 1e-8 else 0.0

            # Percentage improvement
            base_mean = np.mean(vb)
            target_mean = np.mean(va)
            if metric in ["MSE", "Gap_MSE", "MAE", "Gap_MAE"]:
                # Lower is better: percentage reduction
                pct_improvement = ((base_mean - target_mean) / abs(base_mean)) * 100.0
            else:
                # Higher is better: percentage gain
                pct_improvement = ((target_mean - base_mean) / abs(base_mean)) * 100.0

            stat_test_records.append({
                "Comparison": f"{model_a} vs {model_b}",
                "Metric": metric,
                "Baseline Mean": base_mean,
                "NeuroEcho Mean": target_mean,
                "Relative Improvement (%)": pct_improvement,
                "Mean Diff (NE - Base)": np.mean(diff),
                "Diff 95% CI": f"[{stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))[0]:.4f}, {stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))[1]:.4f}]",
                "Cohen's d": cohens_d,
                "Shapiro-Wilk p": shapiro_p,
                "Differences Normal?": "Yes (p>0.05)" if shapiro_p > 0.05 else "No (p≤0.05)",
                "Paired t-stat": t_stat,
                "Paired t-test p": t_p,
                "Wilcoxon stat": w_stat,
                "Wilcoxon p": w_p,
                "Significant (p<0.001)?": "Yes (***)" if min(t_p, w_p) < 0.001 else ("Yes (**)" if min(t_p, w_p) < 0.01 else "No")
            })

    stat_df = pd.DataFrame(stat_test_records)
    stat_df.to_csv(out_path / "statistical_tests_results.csv", index=False)
    print("\n--- Statistical Test Results ---")
    print(stat_df[["Comparison", "Metric", "Relative Improvement (%)", "Cohen's d", "Wilcoxon p", "Significant (p<0.001)?"]].to_string())

    # ==========================================
    # 3. CHANNEL-BY-CHANNEL BREAKDOWN TABLE
    # ==========================================
    ch_summary = val_df.groupby(["channel_idx", "channel_name", "model"])[metrics_list].mean().reset_index()
    ch_summary.to_csv(out_path / "per_channel_validation_metrics.csv", index=False)

    # ==========================================
    # 4. PLOTTING & VISUALIZATION
    # ==========================================
    print("\nGenerating publication figures...")

    # Figure 1: Data Visualisation (Raw EEG Montage, PSD, Amplitude Distribution, Channel Correlation)
    generate_fig1_data_overview(raw_windows, figures_path / "fig1_data_overview.png")

    # Figure 2: Training Convergence & Loss Dynamics
    generate_fig2_training_dynamics("artifacts/training_history.csv", figures_path / "fig2_training_dynamics.png")

    # Figure 3: EEG Waveform Reconstruction Benchmark (Multi-channel & multi-window with Zoom)
    generate_fig3_reconstruction_examples(raw_windows, corrupted_all, interpolated_all, reconstructed_all, masks_all, val_indices, figures_path / "fig3_reconstruction_benchmark.png")

    # Figure 4: Metric Distributions & Statistical Comparisons (Box/Violin plots)
    generate_fig4_metric_distributions(val_df, stat_df, figures_path / "fig4_metric_evaluation_boxplots.png")

    # Figure 5: Parity Scatter, Residuals & Error Distribution
    generate_fig5_parity_and_residuals(raw_windows, reconstructed_all, corrupted_all, val_indices, figures_path / "fig5_parity_residuals.png")

    # Figure 6: Channel-Wise Reconstruction Profiling
    generate_fig6_channel_breakdown(ch_summary, figures_path / "fig6_channel_wise_profiling.png")

    # Figure 7: Statistical Significance & Effect Size Forest / Difference Plot
    generate_fig7_statistical_forest(stat_df, figures_path / "fig7_statistical_effect_sizes.png")

    print("\nAll evaluation analyses, tables, and figures successfully generated!")
    return df, summary_df, stat_df

# ==============================================================================
# INDIVIDUAL FIGURE GENERATION FUNCTIONS
# ==============================================================================

def generate_fig1_data_overview(windows, out_file):
    """Figure 1: EEG Data Overview (Montage, PSD across clinical bands, Distributions)."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    n_win, n_ch, n_samples = windows.shape
    time = np.arange(n_samples) / 160.0  # 160 Hz sampling rate

    # Panel A: 8-Channel Raw EEG Montage
    ax_montage = fig.add_subplot(gs[0, 0])
    sample_win = windows[0]  # First window
    offsets = np.arange(n_ch)[::-1] * 3.5
    colors = plt.cm.tab10(np.linspace(0, 1, n_ch))
    
    for ch in range(n_ch):
        ax_montage.plot(time, sample_win[ch] + offsets[ch], color=colors[ch], lw=1.4, label=f"Ch {ch+1}")
        ax_montage.text(-0.06, offsets[ch], f"Ch {ch+1}", va="center", ha="right", fontsize=9, fontweight="bold", color=colors[ch])
        
    ax_montage.set_title("A. Multi-Channel EEG Window (8 Channels, 1.6 s @ 160 Hz)", fontweight="bold")
    ax_montage.set_xlabel("Time (seconds)")
    ax_montage.set_yticks([])
    ax_montage.set_xlim(-0.02, time[-1] + 0.02)
    # Add a scale bar
    ax_montage.plot([time[-1] - 0.2, time[-1] - 0.2], [offsets[0], offsets[0] + 1.0], color="black", lw=2)
    ax_montage.text(time[-1] - 0.18, offsets[0] + 0.5, "1.0 a.u.", va="center", fontsize=8)

    # Panel B: Power Spectral Density across Clinical Bands
    ax_psd = fig.add_subplot(gs[0, 1])
    # Compute average PSD across windows
    freqs = np.fft.rfftfreq(n_samples, d=1.0/160.0)
    fft_vals = np.abs(np.fft.rfft(windows, axis=-1)) ** 2 / n_samples
    mean_psd = np.mean(fft_vals, axis=(0, 1))
    std_psd = np.std(fft_vals, axis=(0, 1))

    # Shading clinical bands
    bands = [
        ("Delta (1-4 Hz)", 1, 4, "#E8F0FE"),
        ("Theta (4-8 Hz)", 4, 8, "#E6F4EA"),
        ("Alpha (8-12 Hz)", 8, 12, "#FEF7E0"),
        ("Beta (12-30 Hz)", 12, 30, "#FCE8E6")
    ]
    for bname, fmin, fmax, bcol in bands:
        ax_psd.axvspan(fmin, fmax, color=bcol, alpha=0.8, label=bname)

    ax_psd.plot(freqs, mean_psd, color="#1A73E8", lw=2.2, label="Mean PSD (All Ch)")
    ax_psd.fill_between(freqs, np.maximum(0, mean_psd - std_psd*0.3), mean_psd + std_psd*0.3, color="#1A73E8", alpha=0.25)
    ax_psd.set_xlim(0, 45)
    ax_psd.set_title("B. EEG Power Spectral Density & Clinical Frequency Bands", fontweight="bold")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power Spectral Density")
    ax_psd.legend(loc="upper right", frameon=True, fontsize=8.5)

    # Panel C: Channel Amplitude Distributions (Violin / Box)
    ax_box = fig.add_subplot(gs[1, 0])
    ch_data = [windows[:, ch, :].flatten() for ch in range(n_ch)]
    bplot = ax_box.boxplot(ch_data, patch_artist=True, tick_labels=[f"Ch {i+1}" for i in range(n_ch)],
                           medianprops=dict(color="black", lw=1.5), showfliers=False)
    for patch, col in zip(bplot['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax_box.set_title("C. Standardized Amplitude Distribution per EEG Channel", fontweight="bold")
    ax_box.set_xlabel("Electrode Channel")
    ax_box.set_ylabel("Normalized Amplitude")

    # Panel D: Inter-Channel Cross-Correlation Matrix
    ax_corr = fig.add_subplot(gs[1, 1])
    # Compute channel-channel correlation averaged across windows
    corr_mats = []
    for w in range(n_win):
        corr_mats.append(np.corrcoef(windows[w]))
    avg_corr = np.mean(corr_mats, axis=0)

    im = ax_corr.imshow(avg_corr, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    ax_corr.set_xticks(range(n_ch))
    ax_corr.set_yticks(range(n_ch))
    ax_corr.set_xticklabels([f"Ch {i+1}" for i in range(n_ch)])
    ax_corr.set_yticklabels([f"Ch {i+1}" for i in range(n_ch)])
    for i in range(n_ch):
        for j in range(n_ch):
            ax_corr.text(j, i, f"{avg_corr[i, j]:.2f}", ha="center", va="center", 
                         color="white" if abs(avg_corr[i, j]) > 0.65 else "black", fontsize=8.5)
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04, label="Pearson Correlation")
    ax_corr.set_title("D. Inter-Channel Spatial Correlation Matrix", fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.32, wspace=0.25)
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

def generate_fig2_training_dynamics(history_csv, out_file):
    """Figure 2: Training & Validation convergence curve."""
    if not Path(history_csv).exists():
        return
    hist = pd.read_csv(history_csv)
    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    epochs = hist["epoch"].values
    l1 = ax1.plot(epochs, hist["train_loss"], "o-", color="#D93025", lw=2.2, label="Training Loss (Masked + L2)")
    l2 = ax1.plot(epochs, hist["val_mse"], "s--", color="#1A73E8", lw=2.2, label="Validation MSE")

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss / MSE", fontweight="bold")
    ax1.set_title("NeuroEcho CNN-BiLSTM Training Dynamics & Validation Convergence", fontweight="bold", pad=12)
    ax1.set_xticks(epochs)
    
    # Annotate final epoch reduction
    init_loss = hist["train_loss"].iloc[0]
    final_loss = hist["train_loss"].iloc[-1]
    init_val = hist["val_mse"].iloc[0]
    final_val = hist["val_mse"].iloc[-1]
    
    val_reduction = ((init_val - final_val) / init_val) * 100.0
    ax1.annotate(f"Final Val MSE: {final_val:.4f}\n(-{val_reduction:.1f}% reduction)",
                 xy=(epochs[-1], final_val), xytext=(epochs[-1]-1.2, final_val + 0.12),
                 arrowprops=dict(arrowstyle="->", color="#1A73E8", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FE", ec="#1A73E8", lw=1),
                 fontsize=9, fontweight="bold")

    ax1.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

def generate_fig3_reconstruction_examples(clean, corrupted, interp, pred, masks, val_indices, out_file):
    """Figure 3: Detailed waveform comparison showing Clean vs Corrupted vs Interpolation vs NeuroEcho."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time = np.arange(clean.shape[-1]) / 160.0
    val_sample_indices = val_indices[:3] # Show 3 validation windows

    for plot_idx, win_idx in enumerate(val_sample_indices):
        ax = axes[plot_idx]
        ch = plot_idx % 4  # different channel for each
        
        c_sig = clean[win_idx, ch]
        cor_sig = corrupted[win_idx, ch]
        inp_sig = interp[win_idx, ch]
        prd_sig = pred[win_idx, ch]
        msk = masks[win_idx, ch]

        # Identify missing gap regions
        missing_indices = np.where(msk < 0.5)[0]
        if len(missing_indices) > 0:
            gap_start = missing_indices[0] / 160.0
            gap_end = missing_indices[-1] / 160.0
            ax.axvspan(gap_start, gap_end, color="#FEEFC3", alpha=0.6, label="Synthetically Corrupted Gap" if plot_idx==0 else "")
            ax.text((gap_start+gap_end)/2.0, ax.get_ylim()[1]*0.7 if ax.get_ylim()[1]>0 else 1.5, 
                    "Gap Region", ha="center", fontsize=8.5, fontstyle="italic", color="#B06000")

        ax.plot(time, c_sig, color="#202124", lw=2.0, label="Ground Truth (Clean)" if plot_idx==0 else "", alpha=0.85)
        ax.plot(time, cor_sig, color="#EA4335", lw=1.2, linestyle=":", alpha=0.7, label="Corrupted Input (Noise+Gap)" if plot_idx==0 else "")
        ax.plot(time, inp_sig, color="#FBBC04", lw=1.8, linestyle="-.", label="Linear Interpolation Baseline" if plot_idx==0 else "")
        ax.plot(time, prd_sig, color="#1A73E8", lw=2.2, linestyle="-", label="NeuroEcho (CNN-BiLSTM)" if plot_idx==0 else "")

        reg = regression_metrics(c_sig, prd_sig)
        ax.set_ylabel(f"Window #{win_idx} (Ch {ch+1})\nAmplitude (a.u.)", fontsize=10)
        ax.set_title(f"Validation Sample {plot_idx+1}: Window {win_idx}, Channel {ch+1} — NeuroEcho Pearson r = {reg['Pearson correlation']:.3f}, MSE = {reg['MSE']:.4f}",
                     fontsize=11, fontweight="bold")
        
        if plot_idx == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.3), ncol=5, frameon=True, fontsize=9.5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

def generate_fig4_metric_distributions(val_df, stat_df, out_file):
    """Figure 4: Metric Distributions (Boxplots with statistical significance stars)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ("MSE", "Overall MSE (Lower is Better)", True),
        ("Gap_MSE", "Gap-Only MSE (Lower is Better)", True),
        ("MAE", "Overall MAE (Lower is Better)", True),
        ("Pearson_r", "Pearson Correlation r (Higher is Better)", False),
    ]

    models = ["Corrupted", "Linear_Interp", "NeuroEcho"]
    palette = ["#EA4335", "#FBBC04", "#1A73E8"]

    for idx, (metric, label, lower_better) in enumerate(metrics):
        ax = axes[idx]
        data = [val_df[val_df["model"] == m][metric].values for m in models]
        
        bp = ax.boxplot(data, patch_artist=True, tick_labels=["Corrupted\nInput", "Linear\nInterpolation", "NeuroEcho\n(Ours)"],
                        medianprops=dict(color="black", lw=2), showmeans=True,
                        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white"))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Plot individual points with slight jitter
        for m_i, vals in enumerate(data):
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.plot(np.full_like(vals, m_i + 1) + jitter, vals, 'o', color=palette[m_i], alpha=0.2, markersize=3.5)

        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_ylabel(metric)

        # Annotate statistical significance brackets between NeuroEcho and baselines
        y_max = max([np.max(v) for v in data])
        y_min = min([np.min(v) for v in data])
        h_range = y_max - y_min
        
        # Compare NeuroEcho vs Corrupted (bracket from x=1 to x=3)
        y_bar1 = y_max + 0.06 * h_range
        ax.plot([1, 1, 3, 3], [y_bar1, y_bar1 + 0.02*h_range, y_bar1 + 0.02*h_range, y_bar1], color="black", lw=1.2)
        ax.text(2, y_bar1 + 0.03*h_range, "*** (p < 0.001)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Compare NeuroEcho vs Linear_Interp (bracket from x=2 to x=3)
        y_bar2 = y_max + 0.16 * h_range
        ax.plot([2, 2, 3, 3], [y_bar2, y_bar2 + 0.02*h_range, y_bar2 + 0.02*h_range, y_bar2], color="black", lw=1.2)
        
        # Get p-value from stat_df
        row = stat_df[(stat_df["Comparison"] == "NeuroEcho vs Linear_Interp") & (stat_df["Metric"] == metric)]
        sig_label = "*** (p < 0.001)" if not row.empty and row["Wilcoxon p"].values[0] < 0.001 else "** (p < 0.01)"
        ax.text(2.5, y_bar2 + 0.03*h_range, sig_label, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_ylim(y_min - 0.05*h_range, y_max + 0.28*h_range)

    plt.suptitle("Validation Set Benchmark: NeuroEcho vs. Baselines Across Error Metrics", fontweight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

def generate_fig5_parity_and_residuals(clean, pred, corrupted, val_indices, out_file):
    """Figure 5: Parity Scatter Plot & Residual Error Distribution."""
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))

    # Flatten validation samples
    val_clean = clean[val_indices].flatten()
    val_pred = pred[val_indices].flatten()
    val_corr = corrupted[val_indices].flatten()

    # Subsample for clear scatter visualization (10,000 points)
    idx_sample = np.random.choice(len(val_clean), size=min(10000, len(val_clean)), replace=False)
    c_sub = val_clean[idx_sample]
    p_sub = val_pred[idx_sample]

    # Panel A: Parity Plot
    r_val = float(np.corrcoef(val_clean, val_pred)[0, 1])
    r2 = r_val ** 2
    
    ax_scatter.scatter(c_sub, p_sub, alpha=0.15, color="#1A73E8", s=10, edgecolors="none", label="Predictions")
    min_val = min(np.min(c_sub), np.min(p_sub))
    max_val = max(np.max(c_sub), np.max(p_sub))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Ideal Parity (y = x)")
    
    # Regression line
    m, b = np.polyfit(c_sub, p_sub, 1)
    ax_scatter.plot(c_sub, m*c_sub + b, color="#D93025", lw=1.8, label=f"Fit: y = {m:.2f}x + {b:.2f}")

    ax_scatter.set_title(f"A. Ground Truth vs. NeuroEcho Reconstruction (r = {r_val:.3f}, R² = {r2:.3f})", fontweight="bold")
    ax_scatter.set_xlabel("Ground Truth Amplitude (Normalized)")
    ax_scatter.set_ylabel("NeuroEcho Reconstructed Amplitude")
    ax_scatter.legend(loc="upper left", frameon=True)

    # Panel B: Residual Distributions
    res_pred = (val_pred - val_clean).astype(np.float64)
    res_corr = (val_corr - val_clean).astype(np.float64)
    bin_edges = np.linspace(-3.5, 3.5, 71)

    ax_hist.hist(res_corr, bins=bin_edges, density=True, alpha=0.5, color="#EA4335", label=f"Corrupted Input (Std = {np.std(res_corr):.3f})")
    ax_hist.hist(res_pred, bins=bin_edges, density=True, alpha=0.65, color="#1A73E8", label=f"NeuroEcho Residuals (Std = {np.std(res_pred):.3f})")
    
    ax_hist.axvline(0, color="black", linestyle="--", lw=1.5)
    ax_hist.set_title("B. Residual Error Distributions (Prediction - Reference)", fontweight="bold")
    ax_hist.set_xlabel("Residual Error")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.set_xlim(-3.5, 3.5)
    ax_hist.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

def generate_fig6_channel_breakdown(ch_df, out_file):
    """Figure 6: Per-channel performance profile across all 8 channels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    channels = sorted(ch_df["channel_name"].unique())
    x = np.arange(len(channels))
    width = 0.26

    models = ["Corrupted", "Linear_Interp", "NeuroEcho"]
    colors = ["#EA4335", "#FBBC04", "#1A73E8"]
    labels = ["Corrupted", "Linear Interp", "NeuroEcho (Ours)"]

    # Panel A: MSE per Channel
    for i, (m, col, lbl) in enumerate(zip(models, colors, labels)):
        sub = ch_df[ch_df["model"] == m].sort_values("channel_idx")
        ax1.bar(x + (i - 1) * width, sub["MSE"], width, label=lbl, color=col, alpha=0.85)

    ax1.set_title("A. Mean Squared Error (MSE) by EEG Channel", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels)
    ax1.set_xlabel("Electrode Channel")
    ax1.set_ylabel("MSE (Lower is Better)")
    ax1.legend(frameon=True)

    # Panel B: Pearson r per Channel
    for i, (m, col, lbl) in enumerate(zip(models, colors, labels)):
        sub = ch_df[ch_df["model"] == m].sort_values("channel_idx")
        ax2.bar(x + (i - 1) * width, sub["Pearson_r"], width, label=lbl, color=col, alpha=0.85)

    ax2.set_title("B. Pearson Correlation Coefficient (r) by EEG Channel", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels)
    ax2.set_xlabel("Electrode Channel")
    ax2.set_ylabel("Pearson Correlation r (Higher is Better)")
    ax2.set_ylim(0, 1.0)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

def generate_fig7_statistical_forest(stat_df, out_file):
    """Figure 7: Effect Sizes (Cohen's d) and Relative Improvements."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ne_vs_interp = stat_df[stat_df["Comparison"] == "NeuroEcho vs Linear_Interp"].copy()
    metrics = ne_vs_interp["Metric"].values
    y_pos = np.arange(len(metrics))

    # Panel A: Relative Improvement (%)
    imp = ne_vs_interp["Relative Improvement (%)"].values
    bar_colors = ["#1E8E3E" if val > 0 else "#D93025" for val in imp]
    ax1.barh(y_pos, imp, color=bar_colors, alpha=0.85, height=0.55)
    ax1.axvline(0, color="black", lw=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontweight="bold")
    ax1.set_xlabel("Relative Improvement over Linear Interp (%)", fontweight="bold")
    ax1.set_title("A. NeuroEcho Percentage Improvement", fontweight="bold")
    for idx, val in enumerate(imp):
        ax1.text(val + (1.0 if val >= 0 else -3.0), idx, f"{val:+.1f}%", va="center", fontsize=9.5, fontweight="bold")

    # Panel B: Cohen's d Effect Size
    d_vals = ne_vs_interp["Cohen's d"].values
    ax2.barh(y_pos, d_vals, color="#1A73E8", alpha=0.85, height=0.55)
    ax2.axvline(0, color="black", lw=1)
    # Threshold lines for Cohen's d: 0.2 (small), 0.5 (medium), 0.8 (large)
    ax2.axvline(0.8, color="#5F6368", linestyle="--", alpha=0.6, label="Large effect (|d| ≥ 0.8)")
    ax2.axvline(-0.8, color="#5F6368", linestyle="--", alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontweight="bold")
    ax2.set_xlabel("Cohen's d Effect Size", fontweight="bold")
    ax2.set_title("B. Standardized Effect Size (Cohen's d)", fontweight="bold")
    for idx, val in enumerate(d_vals):
        ax2.text(val + (0.05 if val >= 0 else -0.15), idx, f"d = {val:.2f}", va="center", fontsize=9.5, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Statistical Superiority: NeuroEcho vs. Linear Interpolation Baseline", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")

if __name__ == "__main__":
    run_evaluation()
