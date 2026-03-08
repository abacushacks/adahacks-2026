"""
ASGI config for audio_backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from audio_backend import routing

# Load .env from project root
base_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(base_dir / '.env')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'audio_backend.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        )
    ),
})
