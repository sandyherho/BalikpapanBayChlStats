#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eda.py

Exploratory data analysis of chlorophyll-a and ancillary oceanographic
variables from the Indonesian Maritime Continent study site.

Steps
-----
1. Load and summarise the multivariate dataset.
2. Plot the chlorophyll-a time series and identify HAB events.
3. Plot the seasonal (monthly) cycle of chlorophyll-a.
4. Compute descriptive statistics (skewness, kurtosis).
5. Perform normality tests (Shapiro-Wilk, D'Agostino K^2).
6. Perform stationarity tests (ADF, KPSS).
7. Plot histogram + KDE and ACF / PACF.

Outputs
-------
- ../figs/fig2.png          : Chlorophyll-a time series.
- ../figs/fig3.png          : Seasonal monthly bar plot.
- ../figs/fig4.png          : Histogram with KDE.
- ../figs/fig5.png          : ACF and PACF plots.
- ../reports/eda.txt        : Full statistical report.

Author  : Sandy Herho <sh001@ucr.edu>
Date    : 2026/02/21
License : MIT
"""

import os
import warnings
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
plt.style.use("bmh")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("..", "raw_data", "all_data.csv")
FIG_DIR = os.path.join("..", "figs")
REPORT_PATH = os.path.join("..", "reports", "eda.txt")
DPI = 400


# ===========================================================================
# SeasonalPlotter
# ===========================================================================
class SeasonalPlotter:
    """Visualise monthly seasonal trends for a DataFrame column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Time-indexed DataFrame containing oceanographic variables.
    """

    def __init__(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.dataframe = dataframe

    def plot_seasonal_monthly(
        self,
        column_name,
        title,
        x_label,
        y_label,
        color,
        save_path=None,
        dpi=300,
    ):
        """Create a bar plot of mean monthly values.

        Parameters
        ----------
        column_name : str
            Column to aggregate.
        title : str
            Plot title.
        x_label, y_label : str
            Axis labels.
        color : str
            Bar colour.
        save_path : str or None
            If provided, save the figure to this path.
        dpi : int
            Resolution for saved figure.

        Returns
        -------
        seasonal_trends : pd.Series
            Monthly mean values indexed 1..12.
        min_month : int
            Month number with the lowest average.
        max_month : int
            Month number with the highest average.
        """
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' not in DataFrame.")

        # Resample to monthly averages, then group by calendar month
        monthly_avg = self.dataframe[column_name].resample("M").mean()
        plot_data = pd.DataFrame({column_name: monthly_avg})
        plot_data["Month"] = plot_data.index.month
        seasonal_trends = plot_data.groupby("Month")[column_name].mean()

        min_month = seasonal_trends.idxmin()
        max_month = seasonal_trends.idxmax()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=seasonal_trends.index,
            y=seasonal_trends.values,
            color=color,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=15)
        ax.set_ylabel(y_label, fontsize=15)
        ax.set_xticks(range(12))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi)
            print(f"[INFO] Figure saved -> {save_path}")

        plt.close(fig)
        return seasonal_trends, min_month, max_month


# ===========================================================================
# TimeSeriesAnalysis
# ===========================================================================
class TimeSeriesAnalysis:
    """Statistical characterisation of a univariate time series.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series (e.g. daily chlorophyll-a).
    """

    def __init__(self, series):
        if not isinstance(series, pd.Series):
            raise TypeError("Expected a pandas Series.")
        self.series = series

    # -- descriptive -------------------------------------------------------
    def skewness_kurtosis(self):
        """Return (skewness, excess kurtosis)."""
        return float(stats.skew(self.series)), float(stats.kurtosis(self.series))

    # -- normality tests ---------------------------------------------------
    def normality_tests(self):
        """Run Shapiro-Wilk and D'Agostino K^2 tests.

        Returns
        -------
        results : dict
            ``{test_name: (statistic, p_value, interpretation_str)}``.
        """
        results = {}
        sw_stat, sw_p = stats.shapiro(self.series)
        sw_interp = (
            "Data is not normally distributed."
            if sw_p < 0.05
            else "Data follows a normal distribution."
        )
        results["Shapiro-Wilk"] = (sw_stat, sw_p, sw_interp)

        da_stat, da_p = stats.normaltest(self.series)
        da_interp = (
            "Data is not normally distributed."
            if da_p < 0.05
            else "Data follows a normal distribution."
        )
        results["D'Agostino K^2"] = (da_stat, da_p, da_interp)
        return results

    # -- stationarity tests ------------------------------------------------
    def stationarity_tests(self):
        """Run ADF and KPSS tests.

        Returns
        -------
        results : dict
            ``{test_name: (statistic, p_value, interpretation_str)}``.
        """
        results = {}

        adf_result = adfuller(self.series, autolag="AIC")
        adf_stat, adf_p = adf_result[0], adf_result[1]
        adf_interp = (
            "Series is stationary."
            if adf_p < 0.05
            else "Series is not stationary."
        )
        results["ADF"] = (adf_stat, adf_p, adf_interp)

        kpss_result = kpss(self.series, regression="c")
        kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
        kpss_interp = (
            "Series is stationary."
            if kpss_p >= 0.05
            else "Series is not stationary."
        )
        results["KPSS"] = (kpss_stat, kpss_p, kpss_interp)
        return results

    # -- plots -------------------------------------------------------------
    def plot_timeseries(self, filepath, dpi=400):
        """Plot the time series with markers and save.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.series.index, self.series.values, marker="o")
        ax.set_xlabel("Time [days]", fontsize=20)
        ax.set_ylabel("Chlorophyll-a [mg/m$^3$]", fontsize=20)
        plt.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    def plot_distribution_with_kde(self, filepath, dpi=400):
        """Plot histogram + KDE and save.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(self.series, kde=True, bins=15, ax=ax)
        ax.set_xlabel("Chlorophyll-a [mg/m$^3$]", fontsize=20)
        ax.set_ylabel("Probability Density", fontsize=20)
        plt.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    def plot_acf_pacf(self, filepath, dpi=400):
        """Plot ACF and PACF in a two-panel figure and save.

        Parameters
        ----------
        filepath : str
            Output figure path.
        dpi : int
            Resolution.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(self.series, ax=ax1)
        ax1.set_title("Autocorrelation Function", fontsize=30)
        plot_pacf(self.series, ax=ax2)
        ax2.set_title("Partial Autocorrelation Function", fontsize=30)
        plt.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)
        print(f"[INFO] Figure saved -> {filepath}")

    # -- HAB identification ------------------------------------------------
    def identify_hab_events(self, n=2):
        """Return the *n* largest values and their dates.

        Returns
        -------
        top_n : pd.Series
            Top *n* chlorophyll-a concentrations.
        """
        return self.series.nlargest(n)


# ===========================================================================
# Report writer
# ===========================================================================
def build_report(df, chl_analysis, seasonal_trends, min_month, max_month):
    """Compile a full EDA report string.

    Parameters
    ----------
    df : pd.DataFrame
        Full multivariate dataset.
    chl_analysis : TimeSeriesAnalysis
        Analysis object for the chlorophyll-a series.
    seasonal_trends : pd.Series
        Monthly mean chlorophyll-a values.
    min_month, max_month : int
        Month numbers for extremes.

    Returns
    -------
    report : str
    """
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    sep = "=" * 60
    lines = [
        sep,
        "Exploratory Data Analysis Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
    ]

    # --- DataFrame overview ------------------------------------------------
    lines.append("\n--- Dataset Overview ---")
    buf = StringIO()
    df.info(buf=buf)
    lines.append(buf.getvalue())

    lines.append("--- Descriptive Statistics (all variables) ---")
    lines.append(df.describe().to_string())

    lines.append("\n--- First 5 Rows ---")
    lines.append(df.head().to_string())

    # --- Chlorophyll-a series info -----------------------------------------
    chl = chl_analysis.series
    lines.append(f"\n--- Chlorophyll-a Series ---")
    lines.append(f"  Count  : {chl.count()}")
    lines.append(f"  Mean   : {chl.mean():.4f} mg/m^3")
    lines.append(f"  Std    : {chl.std():.4f} mg/m^3")
    lines.append(f"  Min    : {chl.min():.4f} mg/m^3")
    lines.append(f"  Max    : {chl.max():.4f} mg/m^3")
    lines.append(f"  Median : {chl.median():.4f} mg/m^3")

    # --- HAB events --------------------------------------------------------
    top_two = chl_analysis.identify_hab_events(n=2)
    lines.append("\n--- Potential HAB Events (top 2 Chl-a concentrations) ---")
    for date, value in top_two.items():
        lines.append(f"  {value:.4f} mg/m^3 on {date.date()}")

    # --- Seasonal cycle ----------------------------------------------------
    lines.append("\n--- Seasonal Cycle (monthly mean Chl-a) ---")
    for m, val in seasonal_trends.items():
        lines.append(f"  {month_names[m]} : {val:.4f} mg/m^3")
    lines.append(
        f"  Minimum month : {month_names[min_month]} "
        f"({seasonal_trends[min_month]:.4f})"
    )
    lines.append(
        f"  Maximum month : {month_names[max_month]} "
        f"({seasonal_trends[max_month]:.4f})"
    )

    # --- Skewness / kurtosis -----------------------------------------------
    skew, kurt = chl_analysis.skewness_kurtosis()
    lines.append("\n--- Skewness & Kurtosis ---")
    lines.append(f"  Skewness         : {skew:.4f}")
    lines.append(f"  Excess kurtosis  : {kurt:.4f}")
    if np.isclose(skew, 0, atol=0.1):
        lines.append("  Interpretation   : Distribution is approximately symmetric.")
    else:
        direction = "right (positive)" if skew > 0 else "left (negative)"
        lines.append(f"  Interpretation   : Distribution is skewed to the {direction}.")
    if kurt < 0:
        lines.append("  Kurtosis interp. : Platykurtic (lighter tails than normal).")
    else:
        lines.append("  Kurtosis interp. : Leptokurtic (heavier tails than normal).")

    # --- Normality tests ---------------------------------------------------
    norm = chl_analysis.normality_tests()
    lines.append("\n--- Normality Tests ---")
    for name, (stat, p, interp) in norm.items():
        lines.append(f"  {name}: statistic={stat:.4f}, p-value={p:.4f}")
        lines.append(f"    -> {interp}")

    # --- Stationarity tests ------------------------------------------------
    stat_tests = chl_analysis.stationarity_tests()
    lines.append("\n--- Stationarity Tests ---")
    for name, (stat, p, interp) in stat_tests.items():
        lines.append(f"  {name}: statistic={stat:.4f}, p-value={p:.4f}")
        lines.append(f"    -> {interp}")

    lines.append("\n" + sep)
    return "\n".join(lines) + "\n"


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Run the full EDA pipeline."""
    # Ensure output directories exist
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    chl = df["Chlorophyll_a"]
    print(f"[INFO] Loaded {len(df)} rows from {DATA_PATH}")

    # 2. Time series plot  (fig2)
    analysis = TimeSeriesAnalysis(chl)
    analysis.plot_timeseries(os.path.join(FIG_DIR, "fig2.png"), dpi=DPI)

    # 3. Seasonal cycle plot (fig3)
    plotter = SeasonalPlotter(df)
    seasonal_trends, min_month, max_month = plotter.plot_seasonal_monthly(
        column_name="Chlorophyll_a",
        title=" ",
        x_label="Month",
        y_label="Chlorophyll-a [mg/m$^3$]",
        color="#5aad65",
        save_path=os.path.join(FIG_DIR, "fig3.png"),
        dpi=DPI,
    )

    # 4. Distribution + KDE (fig4)
    analysis.plot_distribution_with_kde(
        os.path.join(FIG_DIR, "fig4.png"), dpi=DPI
    )

    # 5. ACF / PACF (fig5)
    analysis.plot_acf_pacf(os.path.join(FIG_DIR, "fig5.png"), dpi=DPI)

    # 6. Compile and save report
    report = build_report(df, analysis, seasonal_trends, min_month, max_month)
    with open(REPORT_PATH, "w") as fh:
        fh.write(report)
    print(f"[INFO] Report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
