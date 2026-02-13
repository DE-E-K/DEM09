# Flight Fare Prediction Summary

## Business Question
How accurately can we estimate flight fares based on airline, route, travel date, and fare components to support pricing strategy and traveler recommendations?

## Data and Assumptions
- Dataset: `Flight_Price_Dataset_of_Bangladesh.csv` in `data/raw/`
- Target: `total_fare`
- Key assumptions:
  - Missing numeric values are median-imputed
  - Missing categorical values use mode fallback
  - Negative fare values are invalid and removed

## Modeling Approach
- Baseline: Linear Regression
- Additional models: Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting
- Optimization: GridSearchCV with cross-validation
- Metrics: R2, MAE, RMSE

## Key Insights (Fill After Running)
- Most influential fare drivers:
- Highest priced routes:
- Seasonal patterns:
- Airline pricing differences:

## Recommendation (Fill After Running)
- Best model selected:
- Suggested pricing action:
- Monitoring strategy:
