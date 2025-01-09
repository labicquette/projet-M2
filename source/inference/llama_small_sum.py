import os
from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm
from bs4 import BeautifulSoup
import json

def get_model(**parameters):
    client = Client(host='http://localhost:11434')
    return client

def get_tokenizer(**parameters):
    return "tok"
# Fonction pour nettoyer les balises HTML
def remove_html_tags(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

"""
def mapped(ex, client, save_path, dataset, parameters, run,i):
  #ex_no_html = remove_html_tags(ex)
  response = client.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M', 
         options={"num_ctx": 128000, "num_gpu":60},
        messages=[
            {
                'role': 'user',
                'content': "You are an expert at writing short summaries. Your job is to summarize the text below in exactly " +
                            "3 SHORT SENTENCES. Focus only on the most important information and leave out everything extra in order " +
                            "to make 3 SHORT SENTENCES. Here's the text: " + ex["examples"] +
                            "make sure to make 3 SHORT SENTENCES.Not more",
            },
        ])
  #print(response['message']['content'])
  file = get_file(dataset[run][i], parameters["dataset_name"])
  with open(save_path + file, "w") as f:
    f.write(response['message']['content'])
"""

def inference(dataset, **parameters):
    run = parameters["run"]
    client = parameters["model"]
    examples = parameters["examples"]
    print(dataset)
    # Ensure save_path directory exists
    save_path = "/content/drive/MyDrive/projet_m2/data/small_sum/"
    os.makedirs(save_path, exist_ok=True)

    for i, ex in enumerate(tqdm(dataset["train"][examples])):
    #dataset["train"].map(lambda ex,i : mapped(ex, client, save_path, dataset, parameters, run,i), with_indices=True)
      ex_no_html = remove_html_tags(ex)
      response = client.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M', 
      options={"num_ctx": 128000, "num_gpu":60},
      messages=[
          {
              'role': 'user',
              'content': ("You are an expert at writing short summaries. Your job is to summarize the text below in exactly " +
                          "3 SHORT SENTENCES. Focus only on the most important information and leave out everything extra in order " +
                          "to make 3 SHORT SENTENCES. Here's the text: " + ex_no_html +
                          "make sure to make 3 SHORT SENTENCES.Not more"),
          },
      ])
      file = get_file(dataset[run][i], parameters["dataset_name"])
      with open(save_path + file, "w") as f:
          f.write(response['message']['content'])
  
