import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from predictor.service import predict


def health(request):
    return JsonResponse({"status": "ok"})


@csrf_exempt
def predict_fare(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
        predicted_fare = predict(payload)
        return JsonResponse({"predicted_total_fare": predicted_fare})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
