#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extreme_value_analysis.py

Extreme value analysis (EVA) of chlorophyll-a concentrations using the
block-maxima method and Bayesian (Emcee MCMC) fitting via pyextremes.

Steps
-----
1. Load the dataset and extract the chlorophyll-a time series.
2. Specify a block-maxima model (14-day blocks).
3. Fit a GEV distribution using the Emcee MCMC sampler.
4. Plot block-maxima overlay, MCMC traces, and corner plot.
5. Compute return-period summary and individual return periods.
6. Plot diagnostic figures (return-level, PDF, Q-Q, P-P).

Outputs (figures)
-----------------
- ../figs/fig6a.png  : Block maxima overlaid on the full time series.
- ../figs/fig6b.png  : MCMC trace plots.
- ../figs/fig6c.png  : Corner (pair) plot of posterior samples.
- ../figs/fig7.png   : Diagnostic panel (return level, PDF, Q-Q, P-P).

Outputs (data)
--------------
- ../processed_data/bm_rp_sum.csv         : Return-period summary table.
- ../processed_data/high_return_periods.csv: Per-event return periods.

Outputs (report)
----------------
- ../reports/eva.txt : Model summary and interpretation.

Author  : Sandy Herho <sh001@ucr.edu>
Date    : 2026/02/21
License : MIT
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyextremes import EVA, get_extremes, get_return_periods

warnings.filterwarnings("ignore")
plt.style.use("bmh")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("..", "raw_data", "all_data.csv")
FIG_DIR = os.path.join("..", "figs")
PROCESSED_DIR = os.path.join("..", "processed_data")
REPORT_PATH = os.path.join("..", "reports", "eva.txt")
DPI = 400
BLOCK_SIZE = "14D"
N_WALKERS = 500
N_SAMPLES = 2500
RETURN_PERIODS_LIST = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]


# ===========================================================================
# ExtChlorophyllModel
# ===========================================================================
class ExtChlorophyllModel:
    """Model and analyse extreme chlorophyll-a concentrations.

    This class wraps the pyextremes EVA workflow: block-maxima extraction,
    Bayesian GEV fitting via Emcee, return-period estimation, and
    diagnostic plotting.

    Parameters
    ----------
    data : pd.Series
        Time-indexed chlorophyll-a series.
    """

    def __init__(self, data):
        self.data = data
        self.model = None
        self.summary_bm = None
        self.extremes = None
        self.return_periods = None

    # -- model specification -----------------------------------------------
    def specify_model(self, block_size="14D"):
        """Extract block maxima and initialise the EVA model.

        Parameters
        ----------
        block_size : str
            Pandas offset alias for the block width (default ``'14D'``).
        """
        self.model = EVA(data=self.data)
        self.model.get_extremes(
            method="BM",
            extremes_type="high",
            block_size=block_size,
            errors="ignore",
        )

    # -- fitting -----------------------------------------------------------
    def fit_model(self, n_walkers=500, n_samples=2500):
        """Fit the GEV distribution via Emcee MCMC.

        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers.
        n_samples : int
            Number of samples per walker.
        """
        self.model.fit_model(
            model="Emcee", n_walkers=n_walkers, n_samples=n_samples
        )

    # -- plotting ----------------------------------------------------------
    def plot_extremes(self, filepath, dpi=400):
        """Plot block maxima over the original time series.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, ax = self.model.plot_extremes(figsize=(12, 8))
        ax.set_xlabel("Time [days]", fontsize=20)
        ax.set_ylabel("Chlorophyll-a [mg/m$^3$]", fontsize=20)
        fig.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    def plot_trace(self, filepath, dpi=400):
        """Save MCMC trace plot.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, _ = self.model.plot_trace(figsize=(15, 8))
        fig.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    def plot_corner(self, filepath, dpi=400):
        """Save posterior corner (pair) plot.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, _ = self.model.plot_corner(figsize=(15, 15), levels=10)
        fig.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    def plot_diagnostic(self, filepath, dpi=400):
        """Save the four-panel diagnostic figure.

        Panels: return-level plot, PDF, Q-Q, P-P.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, _ = self.model.plot_diagnostic(alpha=0.95, figsize=(18, 12))

        # Adjust aesthetics
        for a in fig.axes:
            a.set_title(" ")
        fig.axes[0].set_xlabel("Return Period", fontsize=20)
        fig.axes[0].set_ylabel("Chlorophyll-a [mg/m$^3$]", fontsize=20)
        fig.axes[1].set_xlabel("Chlorophyll-a [mg/m$^3$]", fontsize=20)
        fig.axes[1].set_ylabel("Probability Density", fontsize=20)
        fig.axes[2].set_xlabel("Theoretical", fontsize=20)
        fig.axes[2].set_ylabel("Observed", fontsize=20)
        fig.axes[3].set_xlabel("Theoretical", fontsize=20)
        fig.axes[3].set_ylabel("Observed", fontsize=20)

        sns.despine(left=True)
        fig.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    # -- return periods ----------------------------------------------------
    def compute_summary(self, return_periods_list, alpha=0.95):
        """Compute the return-period summary table.

        Parameters
        ----------
        return_periods_list : list of int
            Desired return periods (in blocks).
        alpha : float
            Confidence level for the CI.

        Returns
        -------
        summary_bm : pd.DataFrame

        Notes
        -----
        The ``n_samples`` parameter is intentionally omitted because the
        Emcee MCMC model already provides posterior samples for
        uncertainty estimation.  Passing ``n_samples`` raises a
        ``TypeError`` with Emcee-fitted models.
        """
        self.summary_bm = self.model.get_summary(
            return_period=return_periods_list,
            alpha=alpha,
        )
        return self.summary_bm

    def compute_return_periods(self, block_size="14D"):
        """Compute per-event return periods using Weibull plotting position.

        Parameters
        ----------
        block_size : str
            Block size matching the model specification.

        Returns
        -------
        rp : pd.DataFrame
            Sorted by return period (descending).
        """
        self.extremes = get_extremes(
            ts=self.data, method="BM", block_size=block_size
        )
        self.return_periods = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method="BM",
            extremes_type="high",
            block_size=block_size,
            return_period_size=block_size,
            plotting_position="weibull",
        )
        rp = self.return_periods.sort_values("return period", ascending=False)
        return rp

    # -- model summary string ----------------------------------------------
    def model_summary_str(self):
        """Return the string representation of the fitted model."""
        return str(self.model)


# ===========================================================================
# Report
# ===========================================================================
def write_report(filepath, model_str, summary_bm, rp_top):
    """Write the EVA report to a text file.

    Parameters
    ----------
    filepath : str
        Output report path.
    model_str : str
        ``str(model)`` output from pyextremes.
    summary_bm : pd.DataFrame
        Return-period summary table.
    rp_top : pd.DataFrame
        Top 5 extreme events by return period.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sep = "=" * 60
    lines = [
        sep,
        "Extreme Value Analysis Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "--- Model Configuration ---",
        f"  Block size   : {BLOCK_SIZE}",
        f"  MCMC walkers : {N_WALKERS}",
        f"  MCMC samples : {N_SAMPLES}",
        "",
        "--- Fitted Model Summary ---",
        model_str,
        "",
        "--- Return-Period Summary Table (95% CI) ---",
        summary_bm.to_string(),
        "",
        "--- Top 5 Extreme Events by Return Period ---",
        rp_top.to_string(),
        "",
        "--- Interpretation ---",
        "The GEV distribution was fitted via Bayesian MCMC (Emcee)",
        "to 14-day block maxima of chlorophyll-a concentration.",
        "Return periods quantify the expected recurrence interval",
        "of extreme HAB-like events. Longer return periods correspond",
        "to rarer, more intense bloom events.",
        "",
        "Diagnostic plots (fig4.png) show return-level curve with 95%",
        "confidence envelope, the fitted PDF, and Q-Q / P-P plots",
        "for goodness-of-fit assessment.",
        sep,
    ]
    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[INFO] Report saved -> {filepath}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Run the full EVA pipeline."""
    # Ensure output directories
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    chl = df["Chlorophyll_a"]
    print(f"[INFO] Loaded {len(chl)} observations from {DATA_PATH}")

    # 2. Build and fit model
    eva_model = ExtChlorophyllModel(data=chl)
    eva_model.specify_model(block_size=BLOCK_SIZE)
    eva_model.fit_model(n_walkers=N_WALKERS, n_samples=N_SAMPLES)
    model_str = eva_model.model_summary_str()
    print(model_str)

    # 3. Plots
    eva_model.plot_extremes(os.path.join(FIG_DIR, "fig6a.png"), dpi=DPI)
    eva_model.plot_trace(os.path.join(FIG_DIR, "fig6b.png"), dpi=DPI)
    eva_model.plot_corner(os.path.join(FIG_DIR, "fig6c.png"), dpi=DPI)

    # 4. Return-period computations
    summary_bm = eva_model.compute_summary(
        return_periods_list=RETURN_PERIODS_LIST
    )
    summary_bm.to_csv(os.path.join(PROCESSED_DIR, "bm_rp_sum.csv"))
    print(f"[INFO] Saved -> {os.path.join(PROCESSED_DIR, 'bm_rp_sum.csv')}")

    rp = eva_model.compute_return_periods(block_size=BLOCK_SIZE)
    rp.to_csv(os.path.join(PROCESSED_DIR, "high_return_periods.csv"))
    print(
        f"[INFO] Saved -> "
        f"{os.path.join(PROCESSED_DIR, 'high_return_periods.csv')}"
    )
    rp_top = rp.head(5)
    print(rp_top)

    # 5. Diagnostic figure
    eva_model.plot_diagnostic(os.path.join(FIG_DIR, "fig7.png"), dpi=DPI)

    # 6. Report
    write_report(REPORT_PATH, model_str, summary_bm, rp_top)


if __name__ == "__main__":
    main()
