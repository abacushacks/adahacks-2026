from django.urls import re_path
from audio_stream import consumers

websocket_urlpatterns = [
    re_path(r'ws/audio/$', consumers.AudioConsumer.as_asgi()),
]
