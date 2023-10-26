import logging
import openai
import requests
import inspect
from traceback import format_exc
from urllib.parse import urljoin

from .controller_gpt import GptController
from .controller_kbot import KbotController
from chat import tasks as chat_tasks

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
        self.kbot_only = self.request_data.get('kbot_only')
        if self.kbot_only:
            print("user requested kbot_only processing")
        self.supplied_search_text = self.request_data.get('search_text')
        self.list_ids = ''

    def ask(self):
        # always return data or an errors array, never throw exceptions
        if self.project not in known_projects:
            return {
                'errors': ['unknown project specified',],
            }
        try:
            response = self.ask_qelp2()
            return response
        except Exception as e:
            logger.error(f'error processing request for {self.project}')
            logger.error(format_exc())
            self.chat_data['errors'] = str(e)
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
            back_string = '\t'.join((
                row['id'], 
                row['manufacturer_label'], 
                row['os_name'], 
                row['product_name'], 
                row['topic_name'], 
                row['steps_text']
            ))
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

    def ask_qelp2(self):
        ###############################################
        import numpy as np
        import pandas as pd
        import os
        import re
        import html
        import json
        import seaborn as sns
        import openai
        import requests
        import inspect
        import csv
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
            wait_exponential,
            retry_if_exception_type
        )
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import euclidean_distances
        from sentence_transformers import SentenceTransformer, util

        ###############################################
        #Load knowledgebase data
        dfk_path = f"data/dataset_qelp_{self.project}.csv"
        df_knowledge = pd.read_csv(dfk_path)
        df_knowledge = df_knowledge.fillna('none')
        df_knowledge.dropna(inplace=True)
        df_knowledge.reset_index(level=0, inplace=True)

        ###############################################
        # Load embedding model
        emb_model=SentenceTransformer(
            "all-mpnet-base-v2"
        )
        ###############################################
        def calc_embeddings(some_text):
            text_embeddings = emb_model.encode(some_text,normalize_embeddings=True)
            return text_embeddings.tolist()
        # calc_embeddings('Sitel Group is changing from using the Duo App on your smart phone')

        # Function to create embeddings for each item in a list (row of a df column)
        def embedding_list(df_column):
            column_embeddings_list = list(map(calc_embeddings, df_column))

            return column_embeddings_list
        ###############################################
        content_path = os.path.join('embeddings', self.project, 'embeddings_Content.npy')
        title_path = os.path.join('embeddings', self.project, 'embeddings_title.npy')
        embeddings_title = None
        embeddings_Content = None
        if not os.path.exists(content_path) or not os.path.exists(title_path):
            print('calculating embeddings')
            #Create embeddings for each column we want to compare our text with
            embeddings_title   = embedding_list(df_knowledge['topic_name'])
            embeddings_Content = embedding_list(df_knowledge['steps_text'])
            # Option to save embeddings if no change rather than re calc everytime
            np.save(title_path, np.array(embeddings_title))
            np.save(content_path, np.array(embeddings_Content))
        else:
            # Option to load saved embeddings if no change rather than re calc everytime
            embeddings_title = np.load(title_path, allow_pickle= True).tolist()
            embeddings_Content = np.load(content_path, allow_pickle= True).tolist()
        ###############################################
        # Calculate CosSim between question embeddings and article embeddings
        def cos_sim_list(embedding_question,embedding_list):
            list_cos_sim = []
            for i in embedding_list:
                sim_pair = util.cos_sim(embedding_question,i).numpy()
                list_cos_sim.append(sim_pair[0][0])
                
            return list_cos_sim

        #Calculate outliers within cos_sim_max data set, identified as possible answers
        def find_outliers_IQR(cos_sim_max):
           q1=cos_sim_max.quantile(0.25)
           q3=cos_sim_max.quantile(0.75)
           IQR=q3-q1
           outliers = cos_sim_max[((cos_sim_max>(q3+1.5*IQR)))]

           return outliers
        ###############################################
        #calculate: question embeddings, cosSim with articles, identify 'outliers', create DF of potential answers
        def K_BOT(input_question,language_name,list_ids):
            pd.set_option('display.max_colwidth', 5000)

            #question embeddings
            embeddings_q = calc_embeddings(input_question)

            #calculate cosSim for included fields
            cos_sim_max = list(map(max, cos_sim_list(embeddings_q,embeddings_title),
                                        cos_sim_list(embeddings_q,embeddings_title)))
            df_knowledge['cos_sim_max'] = cos_sim_max

            #calculate log cosSim
            cos_sim_log = np.log2(df_knowledge['cos_sim_max']+1)
            df_knowledge['cos_sim_log'] = cos_sim_log

            #Identify outliers
            df_outliers = find_outliers_IQR(df_knowledge['cos_sim_log']).to_frame().reset_index(level=0, inplace=False)
            
            print(f'KBOT: df outliers {df_outliers}')
            #Create df of potential answers
            df_answers = df_knowledge[['id','language_name','manufacturer_label','manufacturer_id','os_name','os_id','product_name','product_id','topic_name','flow','topic_type','topic_id','topic_slug','category_id','category_slug','steps_text','cos_sim_max','cos_sim_log',]].sort_values(by=['cos_sim_max'], ascending = False).head(len(df_outliers['index']))
            
            df_answers = df_answers[df_answers['language_name'] == language_name]

            df_answers['steps_text'] = df_answers['steps_text'].str.replace('<[^<]+?>', '')
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("[", "")
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("]", "")
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("*", "")
            #search_results = []

#            #If GPT has compiled a list of relevant IDs (after initial user question) filter using this list, save tokens
            if len(list_ids.split(',')) > 0:
                df_answers[df_answers.id.isin(list_ids.split(','))]

            print(f'KBOT: initial df_answers: {df_answers}')
            return df_answers
        ###############################################
        with open("keys/openai_phone_support.txt","r") as f:
            my_API_key = f.read().strip()
        openai.api_key = my_API_key
        ###############################################
#        # Same topic function
#        def same_context(previous_answer, question):
#          #prompt_tokens = len(prompt)
#          #knowledge_tokens = len(knowledge)
#          #summary_tokens = len(summary)
#          #max_knowledge_tokens = 10000
#
#          messages = [{"role": "system", "content" : previous_answer + "\n\nIs the following text a continuation of the previous conversation, [yes] or [no]\n"},
#
#                      {"role": "user", "content" : question}
#                      #{"role": "assistant", "content" :"if [yes] say '0'/nif [no] say '1'"},             
#                      ]
#          
#          completion = openai.ChatCompletion.create(
#            model="gpt-4", 
#            temperature = 0.0,
#            max_tokens=1,
#            top_p=1.0,
#            frequency_penalty=0.5,
#            presence_penalty=0.5,
#            #stop=["."],
#            messages = messages
#                        )
#          
#          #Extract info for tokens used
#          token_usage = completion.usage
#          token_usage["function"] = inspect.currentframe().f_code.co_name
#          #Display token info (or not)
#          #print(token_usage) 
#
#          global global_cost
#          in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
#          out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
#          global_cost = in_cost + out_cost
#
#          return ''.join(completion.choices[0].message.content)        
        ###############################################
        # Function to summarise the user sequence into a concise string of key words for searching the KB
        @retry( # Use the tenacity retry decorator
            wait=wait_exponential(multiplier=1, min=4, max=10), # Use exponential backoff with a minimum and maximum wait time
            stop=stop_after_attempt(10), # Stop retrying after 10 attempts
            retry=retry_if_exception_type(openai.error.RateLimitError), # Retry only if the exception is a RateLimitError
            reraise=True # Reraise the exception if the retrying fails
        )
        def summarise_question(questions):
          messages = [{"role": "system", "content" : "Return search criteria"},
                      {"role": "user", "content" : "convert the text into one concise search criteria which would work well in a search engine\n" + questions},
                      {"role": "assistant", "content" :"my search query"}
          ]
          
          completion = openai.ChatCompletion.create(
            model="gpt-4", 
            temperature = 0.1,
            max_tokens  = 500,
            top_p=1,
            frequency_penalty = 1.5,
            presence_penalty  = 0.0,
            messages = messages
                        )
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ############################################### # Create a summary of the converstion so far to retain context of the conversation (understand back references from the user)
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def summarise_history_3_5(transcript):
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
            #stop=["."],
            messages = messages
          )
          
          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ###############################################
        # This is the function which produces a response to the users question
        @retry( # Use the tenacity retry decorator
            wait=wait_exponential(multiplier=1, min=4, max=10), # Use exponential backoff with a minimum and maximum wait time
            stop=stop_after_attempt(10), # Stop retrying after 10 attempts
            retry=retry_if_exception_type(openai.error.RateLimitError), # Retry only if the exception is a RateLimitError
            reraise=True # Reraise the exception if the retrying fails
        )
        def run_prompt_3_5(prompt,knowledge,summary):
          max_knowledge_tokens = 10000
          messages = [{"role": "system", "content" :"You are an expert but friendly and helpful Qelp technical support agent, you use the context that you are given to answer users' questions.\n"  
                       + knowledge[:max_knowledge_tokens]
                       + "\nuse the previous context to ask funelling questions until your are left with only one answer, If the provided context is not relevant to the question, you say that you are unable to answer the question, and ask clarifying questions\n"},
                      {"role": "user", "content" : "Identify a single knowledgebase topic name that answers the question without including the step-by-step procedure. Ensure to verify the manufacturer and product name before giving an answer.\n if more information is needed ask for other information like OS version.\n" 
                       +prompt}
                      ]
          #list the ids from the knowledgebase which might answer this question\nreturn ids as a comma delimited list
          completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature = 0,
            max_tokens=5000,
            top_p=1.0,
            frequency_penalty=0.9,
            presence_penalty=0.5,
            messages = messages
          )
          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.003)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.004)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ###############################################
        # This is the function which identifies the relevant item IDs
        @retry( # Use the tenacity retry decorator
          wait=wait_exponential(multiplier=1, min=4, max=10), # Use exponential backoff with a minimum and maximum wait time
          stop=stop_after_attempt(10), # Stop retrying after 10 attempts
          retry=retry_if_exception_type(openai.error.RateLimitError), # Retry only if the exception is a RateLimitError
          reraise=True # Reraise the exception if the retrying fails
        )
        def knowledge_ids(prompt,knowledge,summary):
          max_knowledge_tokens = 10000
          messages = [{"role": "system", "content" :"You are an expert at identifying knowledgebase id's for the context that you are given that answer a given question\n"  
                       + knowledge[:max_knowledge_tokens]},
                      {"role": "user", "content" : "list only the ids from the knowledgebase which answer the following question or return an empty list if there are none, make sure to give actual id's, do not make them up or modify them\n" 
                       +prompt
                       +"\nreturn ids as a comma delimited list"}
                      ]
          
          completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature = 0,
            max_tokens=5000,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            messages = messages
          )

          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.003)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.004)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
#        ###############################################
#        def get_knowledgebase_details(filtered_df):
#            data_info = []
#            new_data=[]
#            for index, row in filtered_df.iterrows():
#                id = row['id']
#                url = f'https://horizoncms-251-staging.qelpcare.com/usecases/{id}'
#                response = requests.get(url)
#                data = response.json()
#                data_info.append(data)
#
#            pd.set_option('display.max_colwidth', 5000)
#            df_data = pd.DataFrame(data_info)
#            ids = []
#            manufacturer_labels = []
#            product_names = []
#            os_names = []
#            steps_texts = []
#            imageURLs =[]
##MAMA
#            if not df_data.empty:
#                    for index, row in df_data.iterrows():
#                        id = row['id']
#                        manufacturer_label = row['manufacturer']['label']
#                        product_name = row['product']['name']
#                        imageURL = row["product"]["image"]
#                        os_name = (row['os']['name'] if 'os' in row and row['os'] is not None and 'name' in row['os'] else 'UNKNOWN')
#                        steps_text = [step['text'] for step in row['steps']]
#                        first_three_steps = steps_text[:3]
#                # Append the extracted values to the respective lists
#                    #ids.append(id)
#                    #manufacturer_labels.append(manufacturer_label)
#                    #product_names.append(product_name)
#                    #os_names.append(os_name)
#                    #steps_texts.append(first_three_steps)
#                    #imageURLs.append(imageURL)
#                        data_dict  = {
#                        'id': id,
#                        'manufacturer': manufacturer_label,
#                        'product': product_name,
#                        'os': os_name,
#                        'steps': first_three_steps,
#                        'imgURL': imageURL
#                        }
#                        new_data.append(data_dict)
#                    new_df = pd.DataFrame(new_data)
#                    json_data = new_df.to_json(orient='index')
#                    parsed_data = json.loads(json_data)
#                    formatted_json = json.dumps(parsed_data, indent=4)
#            #print(formatted_json)    
#                    return formatted_json
#            else:
#                return  None     
#        ###############################################
#        def getdetails(listofids):
#            file_path = 'source_data\dataset_qelp.csv'  # Replace with the actual path to your CSV file
#            #target_ids = []  # Replace with the actual IDs you want to search for
#
#            data_for_target_ids = read_csv_data(file_path, listofids)
#            json_data = convert_to_json(data_for_target_ids)
#
#            print(json_data)
#        ###############################################
#        def read_csv_data(file_path, target_ids):
#            data_for_target_ids = []
#
#            with open(file_path, 'r', encoding='utf-8') as csv_file:
#                csv_reader = csv.reader(csv_file)
#                next(csv_reader)
#                for row in csv_reader:
#                    if row[0] in target_ids:  # Assuming the ID is in the first column
#                        data_dict = {
#                            "id":column[0],
#                            "manufacturer_label": column[2],  # Change column1 to an appropriate name
#                            "product_name": column[5],  # Change column2 to an appropriate name
#                            "imageURL":"https://horizon-cms.s3.eu-central-1.amazonaws.com/image-service/18383fd9c650223dfc8a3882d848c1ae.png",
#                            "os_name": column[9],
#                            "steps_text":column[10]
#                            # Add more columns as needed
#                        }
#                        data_for_target_ids.append(data_dict)
#
#            return data_for_target_ids
#        ###############################################
#        def convert_to_json(data):
#            if data:
#                return json.dumps(data, indent=4)
#            else:
#                return "No data found for the target IDs."
#
        def build_tutorial_url(kb_obj):
            # get the real data
            info_url = f"https://horizoncms-251-staging.qelpcare.com/usecases/{kb_obj.get('id')}"
            try:
                http_resp = requests.get(info_url)
                if http_resp.status_code != 200:
                    print(f'error, bad call for product data for {info_url}')
                    return
                info_resp = http_resp.json()
                if not info_resp:
                    print(f'error, empty product data found for {info_url}')
                    return 
            except Exception:
                err_string = f'*** problem finding data for object, error accessing url {info_url} ***'
                kb_obj['tutorial_link'] = err_string
                kb_obj['image_link'] = err_string
                kb_obj['steps'] = []
                return
            
            topic_type = kb_obj.get('topic_type')
            flow = kb_obj.get('flow')

            # be a little paranoid with the kb api, it has incomplete data
            product_slug = ''
            cat_slug = ''
            topic_slug = ''
            product_id = ''
            topic_id = ''
            topic_name = ''
            os_id = ''
            image_url = ''
            product_obj = info_resp.get('product')
            if product_obj and type(product_obj) == dict:
                product_slug = product_obj.get('slug')
                product_id = product_obj.get('id')
                image_url = product_obj.get('image')
            cat_obj = info_resp.get('category')
            if cat_obj and type(cat_obj) == dict:
                cat_slug = cat_obj.get('slug')
            topic_obj = info_resp.get('topic')
            if topic_obj and type(topic_obj) == dict:
                topic_slug = topic_obj.get('slug')
                topic_id = topic_obj.get('id')
                topic_name = topic_obj.get('name')
            os_obj = info_resp.get('os')
            if os_obj and type(os_obj) == dict:
                os_id = os_obj.get('id')

            if topic_type == 'regular': # its a usecase
                base_url = 'http://qelp-qc5-client-staging.s3-website.eu-west-1.amazonaws.com/qc5/qelp_test/en_UK/?page='
                last_segment = f'p5_d{product_id}_t{topic_id}_o{os_id}'
            elif (topic_type in ['flow', 'flow_continued']) and (flow == 'null'):  # its a TroubshootingWizard
                base_url = 'http://qelp-qc5-client-staging.s3-website.eu-west-1.amazonaws.com/qc5/hey-be/nl_BE/?page='
                last_segment = f'p14_d{product_id}_t{topic_id}'
            else:  # its an Installation Assistant
                base_url = 'http://qelp-qc5-client-staging.s3-website.eu-west-1.amazonaws.com/qc5/hey-be/nl_BE/?page='
                last_segment = f'p15_d{product_id}_t{topic_id}'

            # common build logic
            url_parts = [f'{base_url}{product_slug}']
            url_parts.append(cat_slug)
            url_parts.append(topic_slug)
            url_parts.append(last_segment)
            the_url = '/'.join(s.strip('/') for s in url_parts)
            kb_obj['tutorial_link'] = the_url
            kb_obj['image_link'] = image_url
            kb_obj['topic_name'] = topic_name

            info_steps = []
            for step_data in info_resp['steps']:
                info_steps.append(step_data['text'])
            kb_obj['steps'] = info_steps 


        ###############################################
        #Initialise and reset variables, run this once before starting a new chat session
        global_cost = 0
        search_txt = ''
        transcript = ''
        knowledge = ''
        data = ''
        language_name = 'en_UK'
        if self.project == 'tmobile':
            language_name = 'en_US'
        list_ids = ''
        ###############################################
        #run each time you want to add to the conversation
        history = self.chat_data.get('chat_history', [])
        conversation_summary = self.chat_data.get('conversation_summary', '')
        question_summary = self.chat_data.get('question_summary', '')

        #Take the users side of the conversation and summarise into a coherent question (as the chat evolves)
        input_txt = self.input_txt
        history.append({"role": "user", "content" :input_txt}) 

# DMC let's assume context stays the same.  Clarifying answers seem to break this logic, 
#    e.g. 'how to add wifi', 'its an iphone 11' will register as a context change 
#        #Check to see if context changed before submitting the question to the CosSim KB function
#        same_context = same_context(conversation_summary, input_txt).lower()
#        if same_context == 'yes':
#            question_summary = question_summary + ' ' + input_txt # search criteria from whole conversation
#        else:
#            question_summary = input_txt # search criteria from new question only
        question_summary = question_summary + ' ' + input_txt # search criteria from whole conversation
        print(f'MAIN: question summary is {question_summary}')
        if self.supplied_search_text:
            #MAMA
            search_txt = self.supplied_search_text
            if not search_txt.strip().startswith('"'):
                search_txt = '"' + self.supplied_search_text + '"'
            print(f'MAIN: user supplied search text is *{search_txt}*')
        else:
            search_txt = summarise_question(question_summary) 
            print(f'MAIN: summarised search text is *{search_txt}*')
        #Search and return relevant docs from the knowledge base
        df_answers = K_BOT(search_txt, language_name, list_ids)

        #Convert relevant knowledge items into a 'table' to be included as context for the prompt
        knowledge = 'ID\tmanufacturer\toperating system\tproduct\tanswer\tsteps'

        answer_as_list = []
        list_ids_as_arr = []
        counter = 0
        for index, row in df_answers.iterrows():
            if counter > 2: 
                break
            counter += 1 
            list_ids_as_arr.append(row['id'])
            knowledge =  knowledge + '\n' + row['id'] + '\t' + row['manufacturer_label'] + '\t' + str(row['manufacturer_id']) + '\t' + row['os_name'] + '\t' + str(row['os_id']) + '\t' + row['product_name'] + '\t' + str(row['product_id'])+ '\t' + str(row['flow'])  + '\t'+ str(row['topic_type']) + '\t'+ row['topic_name'] + '\t'+ str(row['topic_id']) +'\t' + str(row['category_id']) + '\t' + str(row['category_slug']) + '\t' + str(row['topic_slug']) + '\t' + row['steps_text']

            new_obj = {
                'id': row['id'],
                'manufacturer': row['manufacturer_label'],
                'os': row['os_name'],
                'product': row['product_name'],
                'flow': row['flow'],
                'topic_type': row['topic_type'],
            }
            build_tutorial_url(new_obj)
            answer_as_list.append(new_obj)

        if self.kbot_only == 'yes':
            return {
                'message': '',
                'kb_items': [x.get('id') for x in answer_as_list]
            }
        # Identify relevant knowledge IDs
        list_ids = knowledge_ids(search_txt, knowledge, conversation_summary)

        #Come up with a response to the question
        data = run_prompt_3_5(search_txt, knowledge, conversation_summary).split('\n')
        if type(data) == list: # some GPT weirdness, sometimes it gives us a string, sometimes an array
            while("" in data):
                data.remove("")
            data = ''.join(data)

        #add Q&A to a list tracking the conversation
        history.append({"role": "assistant", "content" :data}) 

        #Format the list as text to feed back to GPT summary function
        t = [f"{x.get('role')}\t{x.get('content')}" for x in history]
        transcript = '\n'.join(t)

        #summarise transcription for question answer function (this is after the results to reduce wait time)
#        conversation_summary = summarise_history_3_5(transcript)
#        self.chat_data['conversation_summary'] = conversation_summary
        self.chat_data['question_summary'] = question_summary
        self.chat_data['chat_history'] = history
        self.chat_data['latest_kb_items'] = answer_as_list
        self.chat_data['transcript'] = transcript
 
        chat_tasks.summarize_conversation.delay(self.session_key)
        return {
            'message': data,
            'kb_items': answer_as_list,
        }
