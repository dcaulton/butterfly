from celery import shared_task
import inspect
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
    retry_if_exception_type
)
from time import sleep

from chat.redis_session_wrapper import RedisSessionWrapper


@shared_task
def summarize_conversation(session_number):
    sleep(2)
    print(f'running the summarize_conversation task for {session_number}')
    r = RedisSessionWrapper()
    if not r.session_exists(session_number):
        print(f'error - session {session_number} not found')
    chat_data = r.get_data_from_session(session_number)
    transcript = chat_data.get('transcript')
    conversation_summary = summarise_history_3_5(transcript)
    chat_data['conversation_summary'] = conversation_summary
    r.update_session_data(session_number, chat_data)
    print(f'task finished successfully')

############################################### # Create a summary of the converstion so far to retain context of the conversation (understand back references from the user)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def summarise_history_3_5(transcript):
  with open("keys/openai_phone_support.txt","r") as f:
      my_API_key = f.read().strip()
  openai.api_key = my_API_key

  messages = [
      {"role": "user", "content" : "summarise the following conversation in as few words as possible\n" + transcript},
      {"role": "assistant", "content" :"shortest summary without stop words"},
  ]

  completion = openai.ChatCompletion.create(
    model="gpt-4",
    temperature = 0.4,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=2,
    presence_penalty=0.5,
    messages = messages
  )

#  #Extract info for tokens used
#  token_usage = completion.usage
#  token_usage["function"] = inspect.currentframe().f_code.co_name

  return ''.join(completion.choices[0].message.content)

