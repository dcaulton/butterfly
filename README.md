# moth
chatbot api 

#installation instructions
1. make a virtualenv with python 3
2. pip install -r requirements.txt
3. for each project, make a keys/openai_XXX.txt file, where the contents of the file are the API key to openai, and XXX is replaced by the name of the project
4. for each project, populate the data/XXX directory with the raw (pre-embedded) knowledge base date
5. for each project, either a) populate the embeddings/XXX directory with the knowledge base embeddings, so two serialized numpy arrays or b) run the the calc_embeddings management command like so: python manage.py calc_embeddings XXX


