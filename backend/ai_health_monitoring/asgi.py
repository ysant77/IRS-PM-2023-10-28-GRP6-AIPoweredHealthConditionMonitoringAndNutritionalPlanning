"""
ASGI config for ai_health_monitoring project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_health_monitoring.settings")
django.setup()

from channels.routing import ProtocolTypeRouter, URLRouter
import diagnosis.routing
from channels.auth import AuthMiddlewareStack

application = ProtocolTypeRouter({
    "http": AuthMiddlewareStack(URLRouter(
        diagnosis.routing.http_urlpatterns
    )),

    "websocket": AuthMiddlewareStack(URLRouter(
        diagnosis.routing.websocket_urlpatterns  # This uses the WebSocket routes defined in routing.py
    )),
})