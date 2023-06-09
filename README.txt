All software is written in Python 3.6.8. 

Code is partially written by Maria Trusca and Vlad Miron:
https://github.com/mtrusca/HAABSA_PLUS_PLUS
https://github.com/VladMiron00/SI-ABSA


First, the following code has to be run
1. create_embeddings.py to generate embeddings using the uncased BERT base model.
2. main.py to train and test the model, which will result in a graph of the model that we can use to make new predictions.

Using the graph of the model, SHAP model 1 and SHAP model 2 can be run using Jupyter notebook. 

SHAP model 1 needs the following:
- Data file containing three lines per instance: embedded context sentence (words + tokens), embedded target phrase (words + tokens), and the sentiment.
- The word embedding vectors generated in step 1.
- The model graph generated in step 2.

SHAP model 2 needs the following:
- Data file containing three lines per instance: context sentence (words only), target phrase (words only), and the sentiment.
- The model graph generated in step 2.
