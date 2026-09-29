# Retail Inventory Analytics — Project 2

## Challenge
**RETAIL — WHICH PRODUCTS SHOULD THE STORE STOCK MORE?** (DS_Day01_04)

The challenge asks for cleaning/validation, EDA, visualisations, fast/slow-moving product analysis, inventory turnover, stock-out frequency, promotion analysis, store comparisons, inventory recommendations, and a Random Forest model.

## Dataset and target note
The supplied challenge PDF defines the schema and tasks but does not include row-level data in the provided material. This repository therefore uses a **synthetic dataset** for implementation and demonstration.

The PDF also does not explicitly state the prediction label. This implementation derives `Fast_Moving` as whether `Units_Sold` is at or above the 75th percentile. For the Random Forest, the current day's `Units_Sold` and `Closing_Stock` are excluded from model inputs. A prior-demand feature, `Historical_Avg_Units_Sold`, is calculated using only earlier observations for each product.

Do not represent the synthetic dataset as real company data.

## Run
Use Python 3.11/3.12 with pandas, numpy, matplotlib, scikit-learn, and Jupyter.

Open `notebooks/retail_inventory_analysis.ipynb` and **Run All**, or run:
`python retail_project2.py`

## Outputs
- `data/retail_sales_clean.csv`
- `results/random_forest_metrics.csv`
- `results/random_forest_predictions.csv`
- `results/feature_importance.csv`
- `results/product_inventory_recommendations.csv`
- Four visualisations in `visualizations/`
- Final report in `report/`
