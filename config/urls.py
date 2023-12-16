from routes.controllers.prediction_controller import PredictionController
from django.urls import path
from django.contrib import admin
from routes.consumers.http_consumer import HTTPConsumer

urlpatterns = [
    path('admin/', admin.site.urls),
    path('v1/prediction', PredictionController.predict, name='prediction'),
]
