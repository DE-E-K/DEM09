# Django Prediction Service

Minimal Django scaffold for serving fare predictions from `models/best_model.joblib`.

## Quick Start
1. `pip install -r ../requirements.txt`
2. `python manage.py migrate`
3. `python manage.py runserver`

## Endpoint
- `POST /predict/`
- JSON payload example:

```json
{
  "airline": "Biman Bangladesh",
  "source": "Dhaka",
  "destination": "Chittagong",
  "base_fare": 4500,
  "tax_surcharge": 800,
  "month": 5,
  "day": 12,
  "weekday": "Sunday",
  "season": "Summer"
}
```
