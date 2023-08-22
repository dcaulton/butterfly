import copy
import logging
from traceback import format_exc
import uuid

from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .redis_session_wrapper import RedisSessionWrapper
from .controller_ask import AskController
# There is one 'default' session per connection source, 
# overridable by specifying session_key in payload

logger = logging.getLogger(__name__)

class ChatAskViewSet(viewsets.ViewSet):
  def create(self, request):
    request_data = request.data
    input_text = request_data.get('input_text')
    if 'input_text' not in request_data:
        return Response('input_text field needed', status=500)
    project = request_data.get('project')
    if 'project' not in request_data:
        return Response('project field needed', status=500)

    r = RedisSessionWrapper()
    session_key = request_data.get('session_key')
    chat_data = None
    if session_key and not r.session_exists(session_key):
        return Response(f'session not found: {session_key}', status=500)
    if not session_key:
        session_key, chat_data = r.create_new_session(project)
        # give this 'transaction' time to process?
    else:
        chat_data = r.get_data_from_session(session_key)

    if chat_data.get('project') != project:
        return Response('cannot change projects within a session', status=500)

    ac = AskController(chat_data, request_data, session_key, project)
    answer_data = ac.ask()
    r.update_session_data(session_key, chat_data)

    ret_dict = {
      'session_key': session_key,
      'response': answer_data,
    }
    if 'errors' in ret_dict['response']:
        return Response(ret_dict, status=500)
    else:
        return Response(ret_dict)


class ChatSessionViewSet(viewsets.ViewSet):
  def list(self, request):
    r = RedisSessionWrapper()
    session_summary = r.get_session_list()
    resp_data = {
      'sessions': session_summary,
    }
    return Response(resp_data)

  def retrieve(self, request, pk):
    r = RedisSessionWrapper()
    if not r.session_exists(pk):
        return Response('session not found', status=500)
    chat_data = r.get_data_from_session(pk)
    resp_data = {
      'session_key': request.session.session_key,
      'chat_history': chat_data.get('chat_history', []),
      'conversation_summary': chat_data.get('conversation_summary', ''),
    }
    return Response(resp_data)

  def delete(self, request, pk):
    r = RedisSessionWrapper()
    if not r.session_exists(pk):
        return Response('session not found', status=500)
    data_dict = {
        'chat_history': [],
        'conversation_summary': '',
    }
    r.update_session_data(pk, data_dict)
    return Response()


class ChatAskStaticViewSet(viewsets.ViewSet):
  def create(self, request):
    fake_ids = []
    for i in range(10):
        fake_ids.append(str(uuid.uuid4()))
    answer_data = {
      'response_text': 'This is the static response text',
      'list_ids': fake_ids,
    }
    session_key = 'qelp_static_' + str(uuid.uuid4())
    ret_dict = {
      'session_key': session_key,
      'response': answer_data,
    }
    return Response(ret_dict)


