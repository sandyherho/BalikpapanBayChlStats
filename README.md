# Supplementary Materials for "Towards Statistical Modeling of Chlorophyll-a Concentrations in Balikpapan Bay, Indonesia: Implications for Algal Bloom Detection"

[![DOI](https://zenodo.org/badge/878714186.svg)](https://doi.org/10.5281/zenodo.18719868)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![No Maintenance Intended](https://unmaintained.tech/badge.svg)](https://unmaintained.tech/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?logo=matplotlib&logoColor=black)](https://matplotlib.org/)
[![SciPy](https://img.shields.io/badge/scipy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-tabular-orange)](https://auto.gluon.ai/)
[![PyGMT](https://img.shields.io/badge/PyGMT-mapping-green)](https://www.pygmt.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-statistics-brightgreen)](https://www.statsmodels.org/)
[![pyextremes](https://img.shields.io/badge/pyextremes-EVA-red)](https://georgebv.github.io/pyextremes/)
[![emcee](https://img.shields.io/badge/emcee-MCMC-blueviolet)](https://emcee.readthedocs.io/)
[![Seaborn](https://img.shields.io/badge/seaborn-visualization-9cf)](https://seaborn.pydata.org/)

This repository contains the processed data and Python scripts accompanying the following peer-reviewed publication:

> Anwar, I. P., Herho, S. H. S., Khadami, F., Putri, M. R., & Syahrial, S. C. (2026). Towards statistical modeling of chlorophyll-a concentrations in Balikpapan Bay, Indonesia: Implications for algal bloom detection. *Environmental Research Communications*. https://doi.org/10.1088/2515-7620/ae4680


## Data

The dataset comprises 1,096 daily observations from January 1, 2019 to December 31, 2021, with the following variables:

| Variable | Description | Source |
|---|---|---|
| Chlorophyll-a | Ocean color reanalysis (mg/m3) | CMEMS OCEANCOLOUR_GLO_BGC_L4_MY_009_104 |
| Temperature | Sea surface temperature | HAMSOM |
| Salinity | Sea surface salinity | HAMSOM |
| Nitrate | Surface nitrate concentration | CMEMS GLOBAL_ANALYSISFORECAST_BGC_001_028 |
| Phosphate | Surface phosphate concentration | CMEMS GLOBAL_ANALYSISFORECAST_BGC_001_028 |
| Silicate | Surface silicate concentration | CMEMS GLOBAL_ANALYSISFORECAST_BGC_001_028 |
| Dissolved Oxygen | Surface dissolved oxygen | CMEMS GLOBAL_ANALYSISFORECAST_BGC_001_028 |
| River Discharge | Total river discharge | GloFAS-ERA5 |
| Solar Radiation | Daily accumulated solar radiation | ERA5 |
| Rainfall | Daily precipitation | BMKG Sepinggan Station |

## Scripts

**`study_area_map.py`** -- Generates a PyGMT relief map of the Indonesian Maritime Continent with Balikpapan Bay (116.71 E, 0.97 S) indicated. Outputs `fig1.png`.

**`eda.py`** -- Performs exploratory data analysis including time series visualization, monthly seasonality decomposition, probability distribution characterization with normality testing (Shapiro-Wilk, D'Agostino K2), stationarity testing (ADF, KPSS), and autocorrelation/partial autocorrelation analysis. Outputs `fig2.png` through `fig5.png`.

**`autogluon_ml.py`** -- Trains an AutoGluon WeightedEnsemble_L2 model (ExtraTreesMSE: 0.462, CatBoost: 0.346, LightGBMXT: 0.192) for chlorophyll-a prediction using an 80:20 train-test split with random seed 42. Computes permutation feature importance with significance testing. Serializes the trained model via pickle for operational deployment.

**`extreme_value_analysis.py`** -- Applies Block Maxima extreme value analysis with 14-day blocks, fitting a Generalized Extreme Value distribution via MCMC (emcee; 500 walkers, 2,500 samples per walker). Generates return period plots, diagnostic Q-Q/P-P plots, and MCMC trace diagnostics. Outputs `fig6a.png` through `fig6c.png` and `fig7.png`.

## Requirements

- Python >= 3.9
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- autogluon.tabular
- pygmt
- pyextremes
- emcee

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn autogluon pygmt pyextremes emcee
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{Anwar_2026,
  author    = {Anwar, Iwan P. and Herho, Sandy H. S. and Khadami, Faruq and Putri, Mutiara R. and Syahrial, Sabitha C.},
  title     = {Towards statistical modeling of chlorophyll-a concentrations in {Balikpapan Bay}, {Indonesia}: Implications for algal bloom detection},
  journal   = {Environmental Research Communications},
  year      = {2026},
  publisher = {IOP Publishing},
  doi       = {10.1088/2515-7620/ae4680},
  url       = {https://doi.org/10.1088/2515-7620/ae4680}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Financial support was provided by ITB's Research, Community Service and Innovation Program (PPMI-ITB) 2025 under grant No. FITB.PPMI-1-04-2025 and the Dean's Distinguished Fellowship at the University of California, Riverside 2023.
