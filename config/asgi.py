import os
from django.urls import path
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
from routes.consumers.http_consumer import HTTPConsumer
from channels.routing import ProtocolTypeRouter, URLRouter
from routes.consumers.websocket_consumer import WebsocketConsumer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

django_asgi = get_asgi_application()

websocket_routing = ProtocolTypeRouter({
    "websocket": AuthMiddlewareStack(
        URLRouter(
            [
                path("v1/websockets",
                     WebsocketConsumer.as_asgi()),
            ]
        )
    ),
})

http = ProtocolTypeRouter({
    'http':
        URLRouter(
            []
        )

})

application = ProtocolTypeRouter({
    "http": django_asgi,
    "websocket": websocket_routing,
})
