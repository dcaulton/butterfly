import base64
import copy
import datetime
import logging
import pickle
import redis
from traceback import format_exc
import uuid


SESSION_KEY_PREFIX = 'phone_support_'
MAX_SESSIONS_IN_LIST = 200

logger = logging.getLogger(__name__)


class RedisSessionWrapper():
    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    def session_exists(self, session_key):
        return self.r.exists(session_key)

    def get_data_from_session(self, session_key):
        return self.get_obj_from_redis(session_key)

    def update_session_data(self, session_key, data_dict):
        self.save_obj_to_redis(session_key, data_dict)

    def get_session_list(self):
        ret_dict = {}
        counter = 0
        for key in self.r.scan_iter("*"):
            if key.startswith(SESSION_KEY_PREFIX):
                counter += 1 
                if counter > MAX_SESSIONS_IN_LIST:
                    return ret_dict
                x = self.get_obj_from_redis(key)
                ret_dict[key] = x
        return ret_dict

    def create_new_session(self, project):
        session_key = SESSION_KEY_PREFIX + str(uuid.uuid4())
        session_data = {
            'chat_history': [],
            'conversation_summary': '',
            'project': project,
        }
        self.save_obj_to_redis(session_key, session_data)
        return session_key, session_data

    def save_obj_to_redis(self, key, data):
        try:
            data['last_modified'] = str(datetime.datetime.now())
            tl_p = pickle.dumps(data)
            tl_b = base64.b64encode(tl_p)
            self.r.set(key, tl_b)
        except Exception:
            logger.error(format_exc())
            return {}

    def get_obj_from_redis(self, session_key):
        j_b = self.r.get(session_key)
        try:
            j_d = base64.b64decode(j_b)
            j = pickle.loads(j_d)
            return j
        except Exception:
            logger.error(format_exc())
            return j_b
         



##################### to write to redis
# r = redis.Redis(host='localhost', port=6379, decode_responses=True)
# tl = ['a', 'b', 'c']
# tl_p = pickle.dumps(tl)   # b'\x80\x04\x95\x11\x00\x00\x00\x00\x00\x00\x00]\x94(\x8c\x01a\x94\x8c\x01b\x94\x8c\x01c\x94e.'
# tl_b = base64.b64encode(tl_p)   # b'gASVEQAAAAAAAABdlCiMAWGUjAFilIwBY5RlLg=='
# r.set('chunky_loser_22', tl_b)  # key is chunky_loser_22
##################### to load from redis
# j_b = r.get('chunky_loser_22')  # 'gASVEQAAAAAAAABdlCiMAWGUjAFilIwBY5RlLg=='
# j_d = base64.b64decode(j_b)  # b'\x80\x04\x95\x11\x00\x00\x00\x00\x00\x00\x00]\x94(\x8c\x01a\x94\x8c\x01b\x94\x8c\x01c\x94e.'
# j = pickle.loads(j_d)  # ['a', 'b', 'c']
