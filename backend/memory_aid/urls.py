from django.urls import path

from memory_aid.views import health_check


urlpatterns = [
    path("health/", health_check, name="health-check"),
]
