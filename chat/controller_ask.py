import logging
import json
from io import StringIO 
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
            df_knowledge["combined"] = (
              "Manufacturer: " + df_knowledge.manufacturer_label.str.strip() + "; Product: " + df_knowledge.product_name.str.strip()+ "; Topic: " + df_knowledge.topic_name.str.strip()+ "; OS: " + df_knowledge.os_name.str.strip()
            )
            embeddings_concatlist = embedding_list(df_knowledge["combined"])
            # Option to save embeddings if no change rather than re calc everytime
            np.save(title_path, np.array(embeddings_title))
            np.save(content_path, np.array(embeddings_Content))
            np.save(concatlist_path, np.array(embeddings_concatlist))
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
        def K_BOT(input_question,list_ids,manufacturer,product):
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
            df_answers = df_knowledge[['id','language_name','manufacturer_label','manufacturer_id','os_name','os_id','product_name','product_id', 'product_slug', 'topic_name','flow','topic_type','topic_id','topic_slug','category_id','category_slug','steps_text', 'imgaURL', 'cos_sim_max','cos_sim_log',]].sort_values(by=['cos_sim_max'], ascending = False).head(len(df_outliers['index']))
            
            df_answers = df_answers[df_answers['language_name'] == self.language_name]
            if manufacturer is not None:
                df_answers = df_answers[df_answers['manufacturer_label'] == manufacturer]
            if product is not None:
                df_answers = df_answers[df_answers['product_name'] == product] 

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
        ##check the manufacturer##
        def check_manufacturer_in_dataframe(input_txt, df_knowledge, column_tosearch):
            
            manufacturer_exists= matching_keyword = next((keyword for keyword in df_knowledge[column_tosearch].values if str(keyword).lower() in input_txt.lower()), None)
            
            return manufacturer_exists 
        ##check product##
        def check_product_in_dataframe(input_txt, df_knowledge, column_tosearchp):
            
            Product_exists= matching_keyword = next((keyword for keyword in df_knowledge[column_tosearchp].values if str(keyword).lower() in input_txt.lower()), None)
            
            return Product_exists
        def check_os_in_dataframe(input_txt, df_knowledge, column_tosearchos):
            
            Os_exists= matching_keyword = next((keyword for keyword in df_knowledge[column_tosearchos].values if str(keyword).lower() in input_txt.lower()), None)
            
            return Os_exists  
        ###############################################

        key_path = f"keys/openai_{self.project}.txt"
        with open(key_path,"r") as f:
            my_API_key = f.read().strip()
        openai.api_key = my_API_key
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
            p_messages = [{'role': 'system', 'content' : "This is the conversation so far, this is only to guide yourself, do not refer to it explicitly with the user."},
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

        #Search and return relevant docs from the knowledge base
        def search_for_relevant_documents(search_txt,list_ids,manufacturer,product):
            df_docs = K_BOT(search_txt,list_ids,manufacturer,product)
            
            knowledge = 'ID\tmanufacturer\toperating system\tproduct\ttopic'
            for index, row in df_docs.iterrows():
                knowledge =  knowledge + '\n' + row['id'] + '\t' + row['manufacturer_label'] + '\t' + row['os_name'] + '\t' + row['product_name']  +  '\t'+ row['topic_name']
            #print(knowledge)
            return knowledge

        def respond_to_the_question(knowledge,conversation_summary,input_txt):
            
            p_messages = [{'role': 'system', 'content' : "You're a knowledgeable yet amicable Qelp tech support agent. If the context is blank or empty, advise the user to contact support over the phone, if not, after identifying the specific item in context by checking the manufacturer, device, and OS first, provide a positive response, for instance, 'Here are the relevant knowledgebase articles for your query'. A separate process will disclose the knowledgebase id's, not you. Always focus on the user's question and avoid referring to them in third person. Display empathy if a user expresses frustration. If they mention that a solution worked or they no longer require assistance, conclude the conversation politely without pressing for more information. Use the provided context to respond to user's questions.\nContext: ###" + knowledge + "###, \nPrevious Conversation: ###" + conversation_summary + "###"},
                          {'role': 'user', 'content': "If a generic question is made, make sure to ask for the manufacturer, OS and product name verification before responding and ask clarifying questions until only one answer remains. In case of absent context, don't mention your inability to access knowledgebase ID's or articles, instead advise the user to contact support over the phone. This will give you the necessary context to correctly answer the question. Avoid providing step-by-step guidance in your responses, but nevertheless, provide a positive answer AFTER ensuring that you have the manufacturer, device and OS that answers the user's question AND that is contained in the context. You might say: 'Here are the relevant knowledgebase articles for your query'. Keep the ID, Manufacturer, product, or OS undisclosed in responses. DO NOT list out the articles' names or ID's. Answer this question: ###"+input_txt+"### "}]
            
            

            p_parameters = {'model':'gpt-3.5-turbo-16k', 'temperature':0.1,'max_tokens':1000, 'stream': True}

            kbot_reply = call_gpt(p_messages,p_parameters)

            # print('\n' + input_txt)
            # print(kbot_reply + '\n')

            return kbot_reply

        def knowledge_ids(prompt,knowledge):
            max_knowledge_tokens = 10000
            p_messages = [
                {
                "role": "system",
                "content" :"You have been programmed to expertly identify knowledgebase id's from contextual information in response to a given query. You must provide the exact, unmodified id's that correlate to the question. If no id's are found applicable, an empty list should be returned. You will not communicate unavailability of a knowledgebase and you will NOT create or make up fictitious id's.\n###Knowledge: " + knowledge[:max_knowledge_tokens] + "###\n"
                },
                {
                "role": "user",
                "content" : "Extract and list the exact corresponding ID(s) from the previous knowledge context that answer the following question. If there are none, return an empty list and nothing else:\n" + prompt + "\nOutput only the three most relevant IDs in a comma-delimited format, it is of the utmost importance that you make sure to sort them in order of relevance."
                }
                ]
           
            p_parameters = {'model':'gpt-4', 'temperature':0.0,'max_tokens':5000} 
            listids = call_gpt(p_messages,p_parameters)
            return listids 

        def parallelize_response_ids(knowledge, conversation_summary, input_txt, max_workers=4):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future1 = executor.submit(knowledge_ids, input_txt, knowledge)
                future2 = executor.submit(respond_to_the_question, knowledge, conversation_summary, input_txt)
                listids = future1.result()
                kbot_reply = future2.result()
                #print(kbot_reply)
                print("\n\nList of ID's: ", listids)
                return listids, kbot_reply



        ##added new function to get the details 11/29/2023
        def getdetails(listIds, df_knowledge):
            data_info = []

            # Iterate over rows in listIds
            for index, row in listIds.iterrows():
                id = row['ID']

                # Query the knowledge DataFrame for the specific ID and select specific columns
                result = df_knowledge.loc[df_knowledge['id'] == id, ['id', 'manufacturer_label', 'product_name', 'product_id','product_slug', 'os_name', 'os_id', 'topic_name', 'topic_type', 'topic_id', 'topic_slug', 'category_id', 'category_slug', 'flow', 'steps_text', 'imgaURL']]

                # Check if the result is not empty
                if not result.empty:
                    # Extract necessary information for URL construction
                    product_slug = result['product_slug'].iloc[0]
                    cat_slug = result['category_slug'].iloc[0]
                    topic_slug = result['topic_slug'].iloc[0]
                    product_id = result['product_id'].iloc[0]
                    topic_id = result['topic_id'].iloc[0]
                    os_id = result['os_id'].iloc[0]
                    topic_type = result['topic_type'].iloc[0]
                    flow = result['flow'].iloc[0]

                    # URL construction logic
                    if topic_type == 'regular':
                        last_segment = f'p5_d{product_id}_t{topic_id}_o{os_id}'
                    elif (topic_type in ['flow', 'flow_continued']) and (flow == 'null'):
                        last_segment = f'p14_d{product_id}_t{topic_id}'
                    else:
                        last_segment = f'p15_d{product_id}_t{topic_id}'

                    url_parts = [product_slug,cat_slug, topic_slug, last_segment]
                    the_url = '/'.join(s.strip('/') for s in url_parts)

                    # Append the result and URL to data_info
                    result_dict = result.iloc[0].to_dict()
                    result_dict['tutorial_link'] = the_url
                    data_info.append(result_dict)

            # Create a DataFrame from data_info
            df_data = pd.DataFrame(data_info)

            if not df_data.empty:
                # Convert the DataFrame to a JSON-formatted string
                df_data.rename(columns={'imgaURL': 'image_link'}, inplace=True)
                df_data.rename(columns={'manufacturer_label': 'manufacturer'}, inplace=True)
                df_data.rename(columns={'os_name': 'os'}, inplace=True)
                df_data.rename(columns={'product_name': 'product'}, inplace=True)
                df_data.rename(columns={'steps_text': 'steps'}, inplace=True)
                df_data = df_data[['id','manufacturer','os','product','flow','topic_type','tutorial_link','image_link','topic_name','steps']]
                json_data = df_data.to_json()
                return json_data
            else:
                return None
            
        def fix_output_data(input_string):
            # This is a hack.  The data from the earlier stuff is in a form that does not conform to the contract
            #   with the Qelp UI group.  It's not even a dict or list, it's a string. the keys are all different too
            return_obj = []
            json_data = json.loads(input_string)
            print(f'JSON DATA KEYS {json_data.keys()}')
            for kb_index in json_data.get('id', {}).keys():
                kb_id = json_data['id'][kb_index]
                build_dict = {}
                for kb_field_name in json_data.keys():
                    build_dict[kb_field_name] = json_data[kb_field_name][kb_index]
                return_obj.append(build_dict)
           
            return return_obj

        ###############################################
        #Initialise and reset variables, run this once before starting a new chat session
        search_txt = ''
        knowledge = ''
        data = ''
        list_ids = ''
        ###############################################
        #run each time you want to add to the conversation
        history = self.chat_data.get('chat_history', [])
        conversation_summary = self.chat_data.get('conversation_summary', '')
        kbot_reply = 'Hello, how can I help?'
        column_tosearch = 'manufacturer_label'
        column_tosearchp = 'product_name'
        column_tosearchos = 'os_name'
        manufacturer = None
        product = None
        df_answers=''

        input_txt = self.input_txt

        context, search_txt = context_and_summarization(conversation_summary,input_txt)
        conversation_summary = create_summary_text(conversation_summary,input_txt) #returns conversation_summary
          
        knowledge = search_for_relevant_documents(search_txt,list_ids,manufacturer,product)                         #returns knowledge
        list_ids, kbot_reply = parallelize_response_ids(knowledge=knowledge, conversation_summary=conversation_summary, input_txt=input_txt)
        history.append({"role":"user: ", "content":input_txt}) 
        history.append({"role":"assistant: ", "content":kbot_reply})
        manufacturer=check_manufacturer_in_dataframe(input_txt, df_knowledge, column_tosearch)
        product=check_product_in_dataframe(input_txt, df_knowledge, column_tosearchp)
        osname=check_product_in_dataframe(input_txt, df_knowledge, column_tosearchos)
        df_convert = pd.read_csv(StringIO(knowledge), sep='\t')
        df_convert['ID'] = df_convert['ID'].str.strip()
        cleaned_list_ids = [id.strip() for id in list_ids.split(',')]
        filtered_df = df_convert[df_convert['ID'].str.lower().isin([id.lower() for id in cleaned_list_ids])]
        jsonresult=getdetails(filtered_df,df_knowledge)
        jsonresult = fix_output_data(jsonresult)  # this is a hack 

        self.chat_data['conversation_summary'] = conversation_summary
        self.chat_data['chat_history'] = history
        self.chat_data['latest_kb_items'] = jsonresult

        return {
            'message': kbot_reply,
            'kb_items': jsonresult,
        }
