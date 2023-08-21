from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

import guided_redaction.asgi_routing
from server.router import get_asgi_router
import server.urls 

asgi_router = get_asgi_router()

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()
 
application = ProtocolTypeRouter({
    # Django's ASGI application to handle traditional HTTP requests
    "http": django_asgi_app,
    'websocket': AuthMiddlewareStack(URLRouter(asgi_router.urls))
})
