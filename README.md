# Who Is Still Being Left Behind? Health Care for Mothers and Children in Kenya

Secondary analysis of the **2025–26 Kenya Demographic and Health Survey (KDHS)**. It looks at who uses maternal and child health services, and who is left behind, across wealth, education, residence and region.

📄 **Full analytical brief:** [Read the brief](Kenya_MCH_Equity_Brief.pdf)

## Key findings

| Indicator | Kenya | Lowest group |
|---|---|---|
| Facility delivery | 91.4% | 52.1% (North Eastern) · 56.8% (no education) · 71.0% (poorest quintile) |
| ANC 4+ visits | 73.7% | 41.7% (North Eastern) |
| ANC in first trimester | 36.2% | — |
| Maternal postnatal check ≤ 2 days | 85.5% | 42.5% (North Eastern) |
| Measles 1 (children 12–23 months) | 87.9% | 64.2% (North Eastern) |

- After adjusting for background factors, women with **4+ ANC visits had 2.7× the odds** of a facility birth (95% CI 2.1–3.3).
- Maternal education was the strongest predictor: adjusted OR **19.9** for higher education vs none.
- **Half of recent mothers (50.9%)** said getting money for treatment is a big problem.

## Methods

- **Data:** KDHS 2025–26 children's recode (`KEKR91FL.dta`); 6,634 live births in the 3 years before the survey, across 47 counties and 998 clusters
- **Estimates:** weighted proportions (`v005`) with Taylor-linearised 95% confidence intervals that account for clusters (`v021`) and strata (`v022`)
- **Model:** weighted logistic regression for facility delivery, with design-based (stratified, clustered) standard errors
- **Validation:** Taylor standard errors cross-checked against a stratified delete-one-PSU jackknife
- **Indicators:** coded following the DHS Guide to Statistics (definitions in the brief's Annex A)

## Repository structure

```
├── kenya_mch_equity.py   # full analysis: indicators, CIs, regression, charts
├── figures/              # fig1–fig4.png
├── results.json          # all estimates and regression output
└── README.md
```

## How to run

```bash
pip install pandas numpy scipy matplotlib
python kenya_mch_equity.py KEKR91FL.dta
```

The script writes `results.json` and `fig1.png`–`fig4.png` to the current folder.

## Data access

The DHS microdata are **not included** in this repository. DHS terms of use do not allow redistribution. To reproduce the analysis, register at [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm), request access to the Kenya 2025–26 survey, and download the Stata children's recode.

## Author

**Maya Biswas**, MIS & Documentation Officer working on M&E and data systems for development projects in Bangladesh.
(https://www.linkedin.com/in/maya-biswas-b534561b1/) · Biswasmaya155@gmail.com

*Data source: Kenya Demographic and Health Survey 2025–26, The DHS Program. The findings and interpretation are the author's own.*
