from django.urls import path

from predictor.views import health, predict_fare

urlpatterns = [
    path("health/", health, name="health"),
    path("predict/", predict_fare, name="predict"),
]
