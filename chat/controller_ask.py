import logging
import openai
import inspect
from traceback import format_exc

from .controller_gpt import GptController
from .controller_kbot import KbotController

logger = logging.getLogger(__name__)

# TODO derive this from directory structure
known_projects = ['phone_support', 'tmobile']

class AskController():
    def __init__(self, chat_data, request_data, session_key, project):
        self.chat_data = chat_data
        self.request_data = request_data
        self.session_key = session_key
        self.project = project
        self.kbot_controller = KbotController(project)
        self.gpt_controller = GptController(project)
        self.global_cost = [0.0]
        self.question_summary = ''
        self.history = []
        self.conversation_summary = ''
        self.transcript = ''
        self.knowledge = ''
        self.current_response_text = ''
        self.input_txt = self.request_data.get('input_text')
        self.list_ids = ''

    def ask(self):
        # always return data or an errors array, never throw exceptions
        if self.project not in known_projects:
            return {
                'errors': ['unknown project specified',],
            }
        try:
            response = self.ask_qelp()
            return response
        except Exception as e:
            logger.error(f'error processing request for {self.project}')
            logger.error(format_exc())
            return {
                'errors': [str(e)],
            }

    def get_history(self):
        #Check to see if context changed before submitting the question to the CosSim KB function
        self.question_summary = self.input_txt # search criteria from new question only
        self.conversation_summary = self.chat_data.get('conversation_summary', '')
        if not self.chat_data.get('chat_history'): # new conversation
            logger.info('NEW CONVO CONTEXT')
            self.conversation_summary = self.gpt_controller.summarise_question(
                self.question_summary, self.global_cost) 
            return

        self.history = self.chat_data.get('chat_history')
#        self.conversation_sumary = self.chat_data.get('conversation_summary')

        context = self.gpt_controller.same_context(self.conversation_summary, self.input_txt, self.global_cost).lower()
        if context == 'yes':
            self.question_summary += (' ' + self.input_txt) # search criteria from whole conversation
            logger.info(f'UNCHANGED CONTEXT')
        else:
            self.question_summary = self.input_txt # search criteria from new question only
            self.history = []
            logger.info(f'CHANGED CONTEXT')
        self.conversation_summary = self.gpt_controller.summarise_question(self.question_summary, self.global_cost) 

    def add_q_and_a_to_chat_history(self):
        #add Q&A to a list tracking the conversation
        self.history.append({"role": "user", "content": self.input_txt}) 
        self.history.append({"role": "assistant", "content": self.current_response_text, "list_ids": self.list_ids}) 

    def save_conversation_data(self):
        #summarise transcription for question answer function (this is after the results to reduce wait time)

        #Format the list as text to feed back to GPT summary function
        transcript = ''
        for ind, item in enumerate(self.history):
            print(f'one item  is {item}')
            the_new = item['role'] + '\t' + item['content'] + '\n'
            transcript += the_new

        logger.info(f'\nTHE QUESTION IS: {self.input_txt}')
        logger.info(f"I SEARCHED FOR DOCUMENTS RELATED TO: {self.conversation_summary}")
        logger.info(f'I REPLIED: {self.current_response_text}')
        conversation_summary = self.gpt_controller.summarise_history_3_5(transcript, self.global_cost)

        self.chat_data['chat_history'] = self.history 
        self.chat_data['conversation_summary'] = conversation_summary


    def ask_qelp(self):
        self.get_history()
        df_answers = self.kbot_controller.K_BOT(self.conversation_summary, self.list_ids)
        #Convert relevant knowledge items into a 'table' to be included as context for the prompt
        self.knowledge = '\t'.join(('ID','manufacturer','operating system','product','answer','steps'))
        for index, row in df_answers.iterrows():
            back_string = '\t'.join(
                row['id'], 
                row['manufacturer_label'], 
                row['os_name'], 
                row['product_name'], 
                row['topic_name'], 
                row['steps_text']
            )
            self.knowledge = self.knowledge + '\n' +  back_string

        # Identify relevant knowledge IDs
        self.list_ids = self.gpt_controller.knowledge_ids(
            self.chat_data['conversation_summary'], 
            self.knowledge, 
            self.conversation_summary, 
            self.global_cost
        )

        #Come up with a response to the question
        self.current_response_text = self.gpt_controller.run_prompt_3_5(
            self.chat_data['conversation_summary'], 
            self.knowledge, 
            self.conversation_summary, 
            self.global_cost
        )

        self.add_q_and_a_to_chat_history()
        self.save_conversation_data()

        logger.info(f'CONVERSATION SUMMARY: {self.conversation_summary}')
        logger.info(f'Cost: ${self.global_cost[0]}')

        return {
            'response_text': self.current_response_text,
            'ids': self.list_ids,
        }
