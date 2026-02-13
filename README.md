# Flight Fare Prediction Using Machine Learning

End-to-end regression pipeline for predicting flight fares from airline, route, date, and fare components.

## Project Structure
- `data/raw/`: place `Flight_Price_Dataset_of_Bangladesh.csv` here
- `data/processed/`: cleaned and feature-engineered outputs
- `src/`: preprocessing, EDA, training, tuning, interpretation, orchestration
- `notebooks/`: narrative notebook for analysis and reporting
- `reports/`: figures, model comparison table, stakeholder summary
- `models/`: saved best model artifact
- `django_app/`: prediction endpoint scaffold

## Quick Start
1. Create and activate a virtual environment
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Place dataset in:
   - `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`
4. Run pipeline:
   - `python -m src.main --run-all`

## Outputs
- Processed dataset: `data/processed/flight_fares_processed.csv`
- Model comparison: `reports/model_comparison.csv`
- Best model: `models/best_model.joblib`
- Visuals: `reports/figures/`
- Summary: `reports/summary.md`
