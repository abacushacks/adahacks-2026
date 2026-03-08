from django.urls import path

from memory_aid.consumers import MemoryAidConsumer


websocket_urlpatterns = [
    path("ws/session/", MemoryAidConsumer.as_asgi()),
]
