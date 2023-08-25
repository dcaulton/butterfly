import logging
import os
import inspect

import openai
openai.verify_ssl_certs = False

logger = logging.getLogger(__name__)

# TODO this is going to need to work with async programming, and maybe celery
#  chatGPT WILL time out on us some times, need to handle these kinds of conditions
class GptController():
    def __init__(self, project):
        api_key_path = os.path.join('keys', f'openai_{project}.txt')
        if not os.path.isfile(api_key_path):
            raise Exception(f'no openai api key found for project: {api_key_path}')
        with open(api_key_path, "r") as f:
            the_key = f.read()
            openai.api_key = the_key.strip()

    def same_context(self, previous_answer, question, global_cost):
      messages = [
        {"role": "system", "content" : previous_answer + "\n\nIs the following text a continuation of the previous conversation, [yes] or [no]\n"},
        {"role": "user", "content" : question}
      ]
      
      logger.info('calling openai for chat completion')
      completion = openai.ChatCompletion.create(
        model="gpt-4", 
        temperature = 0.0,
        max_tokens=1,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        messages = messages
      )
      logger.info('openai call complete')
      
      cost = completion.usage
      cost["function"] = inspect.currentframe().f_code.co_name
      logger.info(cost)

      in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
      out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
      global_cost[0] += (in_cost + out_cost)

      return ''.join(completion.choices[0].message.content)

    # Function to summarise the user sequence into a concise string of key words for searching the KB
    def summarise_question(self, questions, global_cost):

      messages = [
          {"role": "system", "content" : "Return search criteria"},
          {"role": "user", "content" : "convert the text into one concise search criteria which would work well in a search engine\n" + questions},
          {"role": "assistant", "content" :"my search query"}
      ]
      
      logger.info('calling openai for chat completion')
      completion = openai.ChatCompletion.create(
        model="gpt-4", 
        temperature = 0.1,
        max_tokens  = 500,
        top_p=1,
        frequency_penalty = 1.5,
        presence_penalty  = 0.0,
        messages = messages
      )
      logger.info('openai call complete')
      cost = completion.usage
      cost["function"] = inspect.currentframe().f_code.co_name
      logger.info(cost)

      in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
      out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
      # test that this works, if global_cost is passed by value it won't
      global_cost[0] += (in_cost + out_cost)

      return ''.join(completion.choices[0].message.content)


    def summarise_history_3_5(self, transcript, global_cost):
      messages = [ {"role": "user", "content" : "summarise the following conversation in as few words as possible\n\n" +
                                                transcript},
                  {"role": "assistant", "content" :"shortest summary without stop words"},
                  
                 ]
      
      logger.info('calling openai for chat completion')
      completion = openai.ChatCompletion.create(
        model="gpt-4", 
        temperature = 0.4,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=2,
        presence_penalty=0.5,
        messages = messages
      )
      logger.info('openai call complete')
      in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
      out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
      # test that this works, if global_cost is passed by value it won't
      global_cost[0] += (in_cost + out_cost)

      return ''.join(completion.choices[0].message.content)

    # This is the function which produces a response to the users question
    def run_prompt_3_5(self, prompt,knowledge,summary, global_cost):
      max_knowledge_tokens = 10000
      messages = [{"role": "system", "content" :"you are friendly and helpful technical support, this is your knowledgebase\n"  
                   + knowledge[:max_knowledge_tokens]
                   + "\nuse this knowledgebase to ask funelling questions until only one answer remains"},
                  {"role": "user", "content" : "decide which single knowledgebase topic answers the question\n" 
                   +prompt}
      ]
      #list the ids from the knowledgebase which might answer this question\nreturn ids as a comma delimited list
      logger.info('calling openai for chat completion')
      completion = openai.ChatCompletion.create(
        model="gpt-4", #3.5-turbo-16k
        temperature = 0.7,
        max_tokens=5000,
        top_p=1.0,
        frequency_penalty=0.9,
        presence_penalty=0.5,
        messages = messages
      )
      logger.info('openai call complete')

      token_usage = completion.usage
      token_usage["function"] = inspect.currentframe().f_code.co_name
      logger.info(token_usage)

      in_cost = (completion.usage['prompt_tokens'] * 0.003)/1000
      out_cost = (completion.usage['completion_tokens'] * 0.004)/1000
      global_cost[0] += (in_cost + out_cost)

      return ''.join(completion.choices[0].message.content)


    # This is the function which identifies the relevant item IDs
    def knowledge_ids(self, prompt, knowledge, summary, global_cost):
      max_knowledge_tokens = 10000
      messages = [
          {"role": "system", "content" :"you are friendly and helpful technical support, this is your knowledgebase\n"  
          + knowledge[:max_knowledge_tokens]},
          {"role": "user", "content" : "list the ids from the knowledgebase which answer this question or return an empty list if there are none\n" 
          +prompt
          +"\nreturn ids as a comma delimited list"}
      ]
      #
      logger.info('calling openai for chat completion')
      completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-16k", #3.5-turbo-16k
          temperature = 0,
          max_tokens=5000,
          top_p=1.0,
          frequency_penalty=0,
          presence_penalty=0,
          #stop=["."],
          messages = messages
      )
      logger.info('openai call complete')

      token_usage = completion.usage
      token_usage["function"] = inspect.currentframe().f_code.co_name
      logger.info(token_usage) 

      in_cost = (completion.usage['prompt_tokens'] * 0.003)/1000
      out_cost = (completion.usage['completion_tokens'] * 0.004)/1000
      global_cost[0] += (in_cost + out_cost)

      return ''.join(completion.choices[0].message.content)
