from django.apps import AppConfig


class AudioStreamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'audio_stream'

    def ready(self):
        from . import signals
