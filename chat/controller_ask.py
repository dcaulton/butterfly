import logging
import json
import openai
import requests
import time
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
        self.language_name = 'en_UK'
        if self.project == 'tmobile':
            self.language_name = 'en_US'

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

    def ask_qelp2(self):
        ###############################################
        import numpy as np
        import pandas as pd
        import os
        import re
        import seaborn as sns
        import openai
        import requests
        import inspect
        from concurrent.futures import ThreadPoolExecutor, as_completed
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
        concatlist_path = os.path.join('embeddings', self.project, 'embeddings_concat_columns.npy')
        embeddings_title = None
        embeddings_Content = None
        embeddings_concatlist = None
        if not os.path.exists(content_path) or not os.path.exists(title_path):
# TODO update this to calc the concatlist embedding when PM is available
# TODO also, add this to the calc_embeddings management command
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
            embeddings_concatlist = np.load(concatlist_path, allow_pickle= True).tolist()
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
        def K_BOT(input_question,list_ids):
            pd.set_option('display.max_colwidth', 5000)

            #question embeddings
            embeddings_q = calc_embeddings(input_question)

            #calculate cosSim for included fields
            cos_sim_max = list(map(max, cos_sim_list(embeddings_q,embeddings_title),
                                        cos_sim_list(embeddings_q,embeddings_concatlist),
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
            
            df_answers = df_answers[df_answers['language_name'] == self.language_name]

            df_answers['steps_text'] = df_answers['steps_text'].str.replace('<[^<]+?>', '')
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("[", "")
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("]", "")
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("*", "")

#            #If GPT has compiled a list of relevant IDs (after initial user question) filter using this list, save tokens
            if len(list_ids.split(',')) > 0:
                df_answers[df_answers.id.isin(list_ids.split(','))]

            print(f'KBOT: initial df_answers: {df_answers}')
            return df_answers
        ###############################################
        key_path = f"keys/openai_{self.project}.txt"
        with open(key_path,"r") as f:
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
        def call_gpt(p_messages, p_parameters):
          start = round(time.time())
          
          if 'stream' in p_parameters:
            completion = ''
            for chunk in openai.ChatCompletion.create(
            messages = p_messages,
            model = p_parameters['model'], 
            temperature = p_parameters['temperature'],
            max_tokens = p_parameters['max_tokens'],
            top_p = 1.0,
            frequency_penalty = 0.5,
            presence_penalty = 0.5,
            stream = True
            #stop=["."]
            ):
                
              content = chunk["choices"][0].get("delta", {}).get("content")
              if content is not None:
                  completion += content
                  print(content, end='')
            

            stop = round(time.time())
            duration = stop - start
            
            # token_count = completion.usage
            # token_count["function"] = inspect.currentframe().f_code.co_name
            # print(token_count["function"] + ' - ' + str(token_count["total_tokens"]) + ' tokens, ' + str(duration) + ' sec')
            #print(str(duration) + ' sec')
            
            return completion
            
          else:
            if 'functions' in p_parameters:
              completion = openai.ChatCompletion.create(
              messages = p_messages,
              
              model = p_parameters['model'], 
              temperature = p_parameters['temperature'],
              max_tokens = p_parameters['max_tokens'],
              functions = p_parameters['functions'],
              function_call = p_parameters['function_call'],
              top_p = 1.0,
              frequency_penalty = 0.5,
              presence_penalty = 0.5
              #stop=["."]
              )
              
            else: 
              if 'functions' and 'stream' not in p_parameters:
                completion = openai.ChatCompletion.create(
                messages = p_messages,
                
                model = p_parameters['model'], 
                temperature = p_parameters['temperature'],
                max_tokens = p_parameters['max_tokens'],
                top_p = 1.0,
                frequency_penalty = 0.5,
                presence_penalty = 0.5
                #stop=["."]
                )

              stop = round(time.time())
              duration = stop - start
              
              token_count = completion.usage
              token_count["function"] = inspect.currentframe().f_code.co_name
              #print(token_count["function"] + ' - ' + str(token_count["total_tokens"]) + ' tokens, ' + str(duration) + ' sec')
              
              return ''.join(completion.choices[0].message.content)

          stop = round(time.time())
          duration = stop - start
          
          token_count = completion.usage
          token_count["function"] = inspect.currentframe().f_code.co_name
          #print(token_count["function"] + ' - ' + str(token_count["total_tokens"]) + ' tokens, ' + str(duration) + ' sec')

          return completion


        # Create a summary of the converstion so far to retain context (understand back references from the user, and gradually build up knowledge)
        def create_summary_text(conversation_summary,input_txt):
            p_messages = [{'role': 'system', 'content' : "Here is the conversation so far"},
                          {'role': 'assistant', 'content' : conversation_summary},
                          {'role': 'user', 'content' : input_txt},
                          {'role': 'user', 'content' : "Summarise the conversation.\nKeep just the relevant facts\nDo NOT speculate or make anything up"}                
                         ]

            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.1,'max_tokens':1000}

            conversation_summary = call_gpt(p_messages,p_parameters)
            #print(summary)
            return conversation_summary

        def context_and_summarization(previous_answer, question):
            p_messages = [{'role': 'system', 'content' : "This is the conversation so far"},
                                {'role': 'assistant', 'content' : previous_answer},
                                {'role': 'user', 'content' : question},
                                {'role': 'user', 'content' : "Check if the question is a continuation of the previous conversation, answer [yes] or [no], then convert the text into one concise sentance which would work well in a search engine.\nNot a list.\n"}]
            
            ## Functions capability
            functions = [
                {
                "name": "confirm_context_and_summary",
                "description": "Evaluate if the current question relates to the ongoing conversation context, and summarize the current question into search criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "same_context": {
                        "type": "string",
                        "description": "Answer if the current question relates to the ongoing conversation with [yes] or [no]"
                    },
                    "question_summary": {
                        "type": "string",
                        "description": "Input to a search engine"
                    }
                    }
                }
                }
            ]

            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.1,'max_tokens':100, 'functions': functions, 'function_call': {'name': 'confirm_context_and_summary'}}

            context_and_summary= call_gpt(p_messages,p_parameters)
            context = json.loads(context_and_summary['choices'][0]['message']['function_call']['arguments'])['same_context']
            summary = json.loads(context_and_summary['choices'][0]['message']['function_call']['arguments'])['question_summary']
            print(context)
            return context, summary

        def call_gpt_1(p_messages, p_parameters):
          start = round(time.time())

          completion = openai.ChatCompletion.create(
            messages = p_messages,
            
            model = p_parameters['model'], 
            temperature = p_parameters['temperature'],
            max_tokens = p_parameters['max_tokens'],
            top_p = 1.0,
            frequency_penalty = 0.5,
            presence_penalty = 0.5
            
            #stop=["."]
            )

          stop = round(time.time())
          duration = stop - start
          
          token_count = completion.usage
          token_count["function"] = inspect.currentframe().f_code.co_name
          print(token_count["function"] + ' - ' + str(token_count["total_tokens"]) + ' tokens, ' + str(duration) + ' sec')


          return ''.join(completion.choices[0].message.content)

        #Did the context change?
        def same_context(previous_answer, question):
                p_messages = [{'role': 'system', 'content' : "This is the conversation so far"},
                                {'role': 'assistant', 'content' : previous_answer},
                                {'role': 'user', 'content' : question},
                                {'role': 'user', 'content' : "Is the following text a continuation of the previous conversation, [yes] or [no]\n"}]

                p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.5,'max_tokens':100}

                same_context= call_gpt(p_messages,p_parameters)
                print(same_context)
                return same_context

        #Create serch text
        def create_search_text(summary,input_txt):
            p_messages = [{'role': 'system', 'content' : "You are typing into a search engine"},
                          {'role': 'user', 'content' : "convert the text into one concise sentance which would work well in a search engine.\nNot a list.\n" + summary + "\n" + input_txt}]

            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.1,'max_tokens':1000}

            search_txt = call_gpt(p_messages,p_parameters)
            return search_txt

        #Search and return relevant docs from the knowledge base
        def search_for_relevant_documents(search_txt,list_ids):
            df_docs = K_BOT(search_txt,list_ids)
            knowledge = 'ID\tmanufacturer\toperating system\tproduct\ttopic'
            counter = 0
            knowledge_ids_as_list = []
            for index, row in df_docs.iterrows():
                if counter > 2:
                    break
                knowledge_ids_as_list.append(row['id'])
                counter += 1
                knowledge =  knowledge + '\n' + row['id'] + '\t' + row['manufacturer_label'] + '\t' + row['os_name'] + '\t' + row['product_name']  +  '\t'+ row['topic_name']

            return knowledge, knowledge_ids_as_list

        def respond_to_the_question(knowledge,conversation_summary,input_txt):
            p_messages = [{'role': 'system', 'content' : "You are an expert but friendly Qelp technical support agent. You pay strong attention to the user's question.You never refer to the user in third person. If the user expresses frustration, you use your soft skills to make the user feel understood. If the user states that the solution worked, or that they do not need your help anymore, you end the conversation gracefully without asking any further questions.  You use the context that you are given to answer users' questions.\n###" + knowledge + "###"},
                          {'role': 'user', 'content' : "what do you remember of the conversation so far"},
                          {'role': 'assistant', 'content' : conversation_summary},
                          {'role': 'user', 'content' : input_txt},
                          {'role': 'user', 'content':f"\nIdentify the knowledgebase IDs and topic names that answers the questions. Ensure to verify the manufacturer and product name before giving an answer and if needed, ask clarifying questions like OS version until one answer remains\n "},
                         
                         ]

            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.3,'max_tokens':1000, 'stream': True}

            kbot_reply = call_gpt(p_messages,p_parameters)

            print('\n' + input_txt)
            # print(kbot_reply + '\n')

            return kbot_reply

        def knowledge_ids(prompt,knowledge):
            max_knowledge_tokens = 10000
            p_messages = [{"role": "system", "content" :"You are an expert at identifying knowledgebase id's for the context that you are given that answer a given question\n"  
                       + knowledge[:max_knowledge_tokens]},
                        {"role": "user", "content" : "list only the ID from the knowledgebase which answer the following question or return an empty list if there are none, make sure to give actual id's, DO NOT make them up or modify them\n answer with IDs only" 
                       +prompt
                       +"\nreturn ID as a comma delimited list"}
                      ]
           
            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.0,'max_tokens':5000} 
            listids = call_gpt(p_messages,p_parameters)
            return listids 

        def parallelize_response_ids(knowledge, conversation_summary, input_txt, max_workers=4):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future1 = executor.submit(knowledge_ids, input_txt, knowledge)
                future2 = executor.submit(respond_to_the_question, knowledge, conversation_summary, input_txt)
                listids = future1.result()
                kbot_reply = future2.result()
                return listids, kbot_reply

        def respond_to_the_question_1(knowledge,conversation_summary,input_txt):
            p_messages = [{'role': 'system', 'content' : "You are here as an expert but friendly and helpful Qelp technical support agent, you use the context that you are given to answer users' questions.\n" + knowledge},
                          {'role': 'user', 'content' : "what do you remember of the conversation so far"},
                          {'role': 'assistant', 'content' : conversation_summary},
                          {'role': 'user', 'content' : input_txt},
                          {'role': 'user', 'content':"\nIdentify the knowledgebase IDs and topic names that answers the questions. Ensure to verify the manufacturer and product name before giving an answer and if needed, ask clarifying questions like OS version until one answer remains\n"},
                         
                         ]

            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.3,'max_tokens':1000}

            kbot_reply = call_gpt(p_messages,p_parameters)

            print('\n' + input_txt)
            print(kbot_reply + '\n')

            return kbot_reply

        #Write transaction log to a file
        def transaction_logging(conversation_summary,history):

            transcript ='--- TRANSCRIPT ---\n\n'

            for i in history:
                txt = i['role'] + '\t' + i['content'] +'\n'
                transcript = transcript + txt

            transaction_summary = '--- SUMMARY ---\n\n' + conversation_summary + '\n\n' + transcript

            with open('Transcript_history/log_' + str(round(time.time())) + '.txt', 'w') as f:
                f.write(transaction_summary)


        def build_kb_objects_from_ids(list_ids):
            kb_objects = []
            for kb_id in list_ids:
                new_obj = fetch_and_build_kb_obj(kb_id)
                kb_objects.append(new_obj)

            return kb_objects


        def fetch_and_build_kb_obj(kb_id):
            # get the real data
            # assume phone_support as the default
            kb_obj = {}
            info_url = f"https://horizoncms-251-staging.qelpcare.com/usecases/{kb_id}"
            if self.project == 'tmobile': 
                info_url = f"https://tmobileusa-99-staging.qelpcare.com/usecases/{kb_id}"

            print(f'getting cms data for {info_url}')
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
            
            # be a little paranoid with the kb api, it has incomplete data
            product_slug = ''
            cat_slug = ''
            topic_slug = ''
            product_id = ''
            product_name = ''
            topic_id = ''
            topic_name = ''
            topic_type = ''
            os_id = ''
            os_name = ''
            image_url = ''
            flow = ''
            manufacturer_obj = ''
            manufacturer_name = ''
            manufacturer_obj = info_resp.get('manufacturer')
            if manufacturer_obj and type(manufacturer_obj) == dict:
                manufacturer_name = manufacturer_obj.get('label')
            product_obj = info_resp.get('product')
            if product_obj and type(product_obj) == dict:
                product_slug = product_obj.get('slug')
                product_name = product_obj.get('name')
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
                topic_type = topic_obj.get('type')
            os_obj = info_resp.get('os')
            flow = info_resp.get('flow')
            if os_obj and type(os_obj) == dict:
                os_id = os_obj.get('id')
                os_name = os_obj.get('name')

            if topic_type == 'regular': # its a usecase
                last_segment = f'p5_d{product_id}_t{topic_id}_o{os_id}'
            elif (topic_type in ['flow', 'flow_continued']) and (flow == 'null'):  # its a TroubshootingWizard
                last_segment = f'p14_d{product_id}_t{topic_id}'
            else:  # its an Installation Assistant
                last_segment = f'p15_d{product_id}_t{topic_id}'

            # common build logic
            url_parts = [product_slug]
            url_parts.append(cat_slug)
            url_parts.append(topic_slug)
            url_parts.append(last_segment)
            the_url = '/'.join(s.strip('/') for s in url_parts)
            kb_obj['manufacturer'] = manufacturer_name
            kb_obj['os'] = os_name
            kb_obj['product'] = product_name
            kb_obj['flow'] = flow
            kb_obj['topic_type'] = topic_type 
            kb_obj['tutorial_link'] = the_url
            kb_obj['image_link'] = image_url
            kb_obj['topic_name'] = topic_name

            info_steps = []
            for step_data in info_resp['steps']:
                info_steps.append(step_data['text'])
            kb_obj['steps'] = info_steps 

            return kb_obj

        ###############################################
        #Initialise and reset variables, run this once before starting a new chat session
        global_cost = 0
        search_txt = ''
        transcript = ''
        knowledge = ''
        data = ''
        list_ids = ''
        ###############################################
        #run each time you want to add to the conversation
        history = self.chat_data.get('chat_history', [])
        conversation_summary = self.chat_data.get('conversation_summary', '')
        question_summary = self.chat_data.get('question_summary', '')

        #Take the users side of the conversation and summarise into a coherent question (as the chat evolves)
        input_txt = self.input_txt
        history.append({"role": "user", "content" :input_txt}) 


        input_txt = self.input_txt
        context, search_txt = context_and_summarization(conversation_summary,input_txt)
        conversation_summary = create_summary_text(conversation_summary,input_txt) #returns conversation_summary
        search_txt = create_search_text(conversation_summary,input_txt)                 #returns search_txt
        knowledge, knowledge_ids_as_list = search_for_relevant_documents(search_txt,list_ids) #returns knowledge
        list_ids, kbot_reply = parallelize_response_ids(knowledge=knowledge, conversation_summary=conversation_summary, input_txt=input_txt)
        history.append({"role":"user: ", "content":input_txt}) 
        history.append({"role":"assistant: ", "content":kbot_reply})
        kb_objects = build_kb_objects_from_ids(knowledge_ids_as_list)

        #summarise transcription for question answer function (this is after the results to reduce wait time)
        self.chat_data['conversation_summary'] = conversation_summary
        self.chat_data['question_summary'] = question_summary
        self.chat_data['chat_history'] = history
        self.chat_data['latest_kb_items'] = kb_objects 
        self.chat_data['transcript'] = transcript

        return {
            'message': kbot_reply,
            'kb_items': kb_objects,
        }
