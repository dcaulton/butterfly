from rest_framework import routers

router = None


def get_router():
    global router
    if not router:
        router = routers.DefaultRouter(trailing_slash=False)
    return router


asgi_router = None
def get_asgi_router():
    global asgi_router
    if not asgi_router:
        asgi_router = AsgiRouter()
    return asgi_router


class AsgiRouter():
    def __init__(self):
        self.urls = []
        self.channels = []

