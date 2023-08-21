from server.router import get_router

from . import api

router = get_router()

router.register(r'v1/ask', api.ChatAskViewSet, basename='ask_chat')
router.register(r'v1/sessions', api.ChatSessionViewSet, basename='view_session')
