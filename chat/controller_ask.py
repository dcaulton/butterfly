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
            response = self.ask_qelp2()
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

        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import euclidean_distances
        from sentence_transformers import SentenceTransformer, util

        ###############################################
        #Load knowledgebase data
        df_knowledge = pd.read_csv("data/dataset_qelp_phone_support.csv")
        df_knowledge = df_knowledge.fillna('none')
        df_knowledge.dropna(inplace=True)
        df_knowledge.reset_index(level=0, inplace=True)

        ###############################################
        # Load embedding model

        # https://www.sbert.net/docs/pretrained_models.html
        # https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1

        emb_model=SentenceTransformer(
            #"sentence-transformers/multi-qa-distilbert-cos-v1"
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
        content_path = os.path.join('embeddings', 'phone_support', 'embeddings_Content.npy')
        title_path = os.path.join('embeddings', 'phone_support', 'embeddings_title.npy')
        embeddings_title = None
        embeddings_Content = None
        if not os.path.exists(content_path) or not os.path.exists(title_path):
            print('calculating embeddings')
            #Create embeddings for each column we want to compare our text with
            embeddings_title   = embedding_list(df_knowledge['topic_name'])
            embeddings_Content = embedding_list(df_knowledge['steps_text'])
            # Option to save embeddings if no change rather than re calc everytime
            np.save('embeddings/phone_support/embeddings_title.npy', np.array(embeddings_title))
            np.save('embeddings/phone_support/embeddings_Content.npy', np.array(embeddings_Content))
        else:
            # Option to load saved embeddings if no change rather than re calc everytime
            embeddings_title = np.load('embeddings/phone_support/embeddings_title.npy', allow_pickle= True).tolist()
            embeddings_Content = np.load('embeddings/phone_support/embeddings_Content.npy', allow_pickle= True).tolist()
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
            
            #Create df of potential answers
            df_answers = df_knowledge[['id','language_name','manufacturer_label','os_name','product_name','topic_name','steps_text','cos_sim_max','cos_sim_log',]].sort_values(by=['cos_sim_max'], 
                                                                                ascending = False).head(len(df_outliers['index']))
            
            df_answers = df_answers[df_answers['language_name'] == language_name]

            df_answers['steps_text'] = df_answers['steps_text'].str.replace('<[^<]+?>', '')
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("[", "")
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("]", "")
            df_answers['steps_text'] = df_answers['steps_text'].str.replace("*", "")
            #search_results = []

            #If GPT has compiled a list of relevant IDs (after initial user question) filter using this list, save tokens
            if len(list_ids.split(',')) > 0:
                df_answers[df_answers.id.isin(list_ids.split(','))]

            return df_answers
        ###############################################
        with open("keys/openai_phone_support.txt","r") as f:
            my_API_key = f.read()

        openai.api_key = my_API_key
        ###############################################
        # Same topic function
        def same_context(previous_answer, question):
          #prompt_tokens = len(prompt)
          #knowledge_tokens = len(knowledge)
          #summary_tokens = len(summary)
          #max_knowledge_tokens = 10000

          messages = [{"role": "system", "content" : previous_answer + "\n\nIs the following text a continuation of the previous conversation, [yes] or [no]\n"},

                      {"role": "user", "content" : question}
                      #{"role": "assistant", "content" :"if [yes] say '0'/nif [no] say '1'"},             
                      ]
          
          completion = openai.ChatCompletion.create(
            model="gpt-4", 
            temperature = 0.0,
            max_tokens=1,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            #stop=["."],
            messages = messages
                        )
          
          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name
          #Display token info (or not)
          #print(token_usage) 

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)        
        ###############################################
        # Function to summarise the user sequence into a concise string of key words for searching the KB
        def summarise_question(questions):

          messages = [{"role": "system", "content" : "Return search criteria"},
                      {"role": "user", "content" : "convert the text into one concise search criteria which would work well in a search engine\n" 
                       + questions},
                      {"role": "assistant", "content" :"my search query"}
                        ]
          
          completion = openai.ChatCompletion.create(
            model="gpt-4", 
            temperature = 0.1,
            max_tokens  = 500,
            top_p=1,
            frequency_penalty = 1.5,
            presence_penalty  = 0.0,
            #stop=["."],
            messages = messages
                        )
          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name
          #Display token info (or not)
          #print(token_usage) 

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ###############################################
        # Create a summary of the converstion so far to retain context of the conversation (understand back references from the user)
        def summarise_history_3_5(transcript):
          messages = [#{"role": "system", "content" : "you are tasks with remembering the key facts only"},
                      {"role": "user", "content" : "summarise the following conversation in as few words as possible\n" +
                                                    transcript},
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
          #Display token info (or not)
          #print(token_usage) 

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.03)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.06)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ###############################################
        # This is the function which produces a response to the users question
        def run_prompt_3_5(prompt,knowledge,summary):
          max_knowledge_tokens = 10000
          messages = [{"role": "system", "content" :"you are friendly and helpful qelp technical support, this is your knowledgebase\n"  
                       + knowledge[:max_knowledge_tokens]
                       + "\nuse this knowledgebase to ask funelling questions until only one answer remains\n if it's a greetings just simply greet back\n"},
                      {"role": "user", "content" : "Identify the relevant knowledge base topic to answer the question, and provide a brief response, don't include the step by step procedure. Ensure to verify the manufacturer and product name if applicable. If the question is a greeting, respond with a simple greeting" 
                       +prompt}
                      ]
          #list the ids from the knowledgebase which might answer this question\nreturn ids as a comma delimited list
          completion = openai.ChatCompletion.create(
            model="gpt-4", #3.5-turbo-16k
            temperature = 0.7,
            max_tokens=5000,
            top_p=1.0,
            frequency_penalty=0.9,
            presence_penalty=0.5,
            #stop=["."],
            messages = messages
                        )
          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name
          #Display token info (or not)
          #print(token_usage) 

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.003)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.004)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ###############################################
        # This is the function which identifies the relevant item IDs
        def knowledge_ids(prompt,knowledge,summary):
          max_knowledge_tokens = 10000
          messages = [{"role": "system", "content" :"you are friendly and helpful qelp technical support, this is your knowledgebase\n"  
                       + knowledge[:max_knowledge_tokens]},
                      {"role": "user", "content" : "list the ids from the knowledgebase which answer this question or return an empty list if there are none\n" 
                       +prompt
                       +"\nreturn ids as a comma delimited list"}
                      ]
          #
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
          #Extract info for tokens used
          token_usage = completion.usage
          token_usage["function"] = inspect.currentframe().f_code.co_name
          #Display token info (or not)
          #print(token_usage) 

          global global_cost
          in_cost = (completion.usage['prompt_tokens'] * 0.003)/1000
          out_cost = (completion.usage['completion_tokens'] * 0.004)/1000
          global_cost = in_cost + out_cost

          return ''.join(completion.choices[0].message.content)
        ###############################################
        def get_knowledgebase_details(filtered_df):

            data_info = []
            new_data=[]
            for index, row in filtered_df.iterrows():
                id = row['id']
                url = f'https://horizoncms-251-staging.qelpcare.com/usecases/{id}'
                #url = f'https://c2-api-staging.customersaas.com/usecases/{id}'
                response = requests.get(url)
                data = response.json()
                data_info.append(data)

            pd.set_option('display.max_colwidth', 5000)
            df_data = pd.DataFrame(data_info)
            ids = []
            manufacturer_labels = []
            product_names = []
            os_names = []
            steps_texts = []
            imageURLs =[]
            if not df_data.empty:
                    for index, row in df_data.iterrows():
                        id = row['id']
                        manufacturer_label = row['manufacturer']['label']
                        product_name = row['product']['name']
                        imageURL = row["product"]["image"]
                        os_name = (row['os']['name'] if 'os' in row and row['os'] is not None and 'name' in row['os'] else 'UNKNOWN')
                        steps_text = [step['text'] for step in row['steps']]
                        first_three_steps = steps_text[:3]
                # Append the extracted values to the respective lists
                    #ids.append(id)
                    #manufacturer_labels.append(manufacturer_label)
                    #product_names.append(product_name)
                    #os_names.append(os_name)
                    #steps_texts.append(first_three_steps)
                    #imageURLs.append(imageURL)
                        data_dict  = {
                        'id': id,
                        'manufacturer': manufacturer_label,
                        'product': product_name,
                        'os': os_name,
                        'steps': first_three_steps,
                        'tutorial_link':"http://qelp-qc5-client-staging.s3-website.eu-west-1.amazonaws.com/qc5/qelp_test/en_UK/?page=samsung-galaxy-s9-android-9-%28pie%29%2Fapps%2Fhow-to-set-up-the-app-store%2Fp5_d3_t12070_o3",
                        'imgURL': imageURL
                
                        }
                        new_data.append(data_dict)
                    new_df = pd.DataFrame(new_data)
                    json_data = new_df.to_json(orient='index')
                    parsed_data = json.loads(json_data)
                    formatted_json = json.dumps(parsed_data, indent=4)
            #print(formatted_json)    
                    return formatted_json
            else:
                return  None     
        ###############################################
        def getdetails(listofids):
            file_path = 'source_data\dataset_qelp.csv'  # Replace with the actual path to your CSV file
            #target_ids = []  # Replace with the actual IDs you want to search for

            data_for_target_ids = read_csv_data(file_path, listofids)
            json_data = convert_to_json(data_for_target_ids)

            print(json_data)
        ###############################################
        def read_csv_data(file_path, target_ids):
            data_for_target_ids = []

            with open(file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)
                for row in csv_reader:
                    if row[0] in target_ids:  # Assuming the ID is in the first column
                        data_dict = {
                            "id":column[0],
                            "manufacturer_label": column[2],  # Change column1 to an appropriate name
                            "product_name": column[5],  # Change column2 to an appropriate name
                            "imageURL":"https://horizon-cms.s3.eu-central-1.amazonaws.com/image-service/18383fd9c650223dfc8a3882d848c1ae.png",
                            "os_name": column[9],
                            "steps_text":column[10]
                            # Add more columns as needed
                        }
                        data_for_target_ids.append(data_dict)

            return data_for_target_ids
        ###############################################
        def convert_to_json(data):
            if data:
                return json.dumps(data, indent=4)
            else:
                return "No data found for the target IDs."
        ###############################################
        #Initialise and reset variables, run this once before starting a new chat session
        global_cost = 0
        question_summary = ''
        history = []
        conversation_summary = ''
        transcript = ''
        knowledge = ''
        data = ''
        language_name = 'en_UK'
        list_ids = ''
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################







        return {
            'response_text': max(df_knowledge['index']),
            'ids': self.list_ids,
        }
