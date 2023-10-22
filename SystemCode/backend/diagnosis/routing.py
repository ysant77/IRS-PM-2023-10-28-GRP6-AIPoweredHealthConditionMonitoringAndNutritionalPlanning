from django.urls import re_path, path
from . import consumers
from django.core.asgi import get_asgi_application

websocket_urlpatterns = [
    re_path(r'^ws/chat/(?P<session_id>\w+)/$', consumers.ChatConsumer.as_asgi()),
    path('ws/check-login-info/', consumers.CheckLoginConsumer.as_asgi())
]

http_urlpatterns = [
    re_path(r'^api/', get_asgi_application()),
    re_path(r'^accounts/', get_asgi_application()),
]