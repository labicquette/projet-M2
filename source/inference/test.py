import numpy as np
import os
import re
from tqdm import tqdm
from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_model(**parameters):
    client = Client(host='http://localhost:11434')
    return client

def get_tokenizer(**parameters):
    return "tok"

def remove_html_tags(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def inference(dataset, **parameters):

    run = parameters["run"]
    client = parameters["model"]

    examples = parameters["examples"]

    file_path = "/content/drive/MyDrive/projet_m2/data/small_sum_final"

    save_path = parameters["save_path"]

    #train_links_dict = {link: idx for idx, link in enumerate(dataset["train"]["justia_link"])}
    #validation_links_dict = {link: idx for idx, link in enumerate(dataset["validation"]["justia_link"])}
    # Process each example in the dataset

    for i, ex in enumerate(tqdm(dataset[run][examples])):
        
          file = get_file(dataset[run][i], parameters["dataset_name"])
          if os.path.exists(save_path+file):
                  continue
        
          # Updated regex patterns for better matching
          regex_1 = r'<a href="/cases/federal/us/\d+/\d+/">.*?</a>'
          regex_2 = r'<a\s+href\s*=\s*["\']\/cases\/federal\/us\/\d+\/\d+\/["\']>.*?</a>'
          
          matches_1 = re.findall(regex_1, ex)
          matches_2 = [m for m in re.findall(regex_2, ex) if m not in matches_1]
          
          # Combine both matches
          all_matches = matches_1 + matches_2

          for match in all_matches:
              # Updated link extraction regex to handle whitespace
              link_match = re.search(r'href\s*=\s*["\']([^"\']+)["\']', match)
              if link_match:
                  link = link_match.group(1)
                  # Prepend the base URL if it's a relative link
                  if not link.startswith("https://"):
                      link = "https://supreme.justia.com" + link
                  
                  # Check if link is in the train split
                  if link in dataset["train"]["justia_link"]:
                      print(True)
                      train_index = np.where(np.array(dataset["train"]["justia_link"]) == link)[0][0]
                      file = get_file(dataset["train"][int(train_index)], "SCOTUS")
                      # Read the content of the corresponding file
                      file_path = os.path.join(file_path, file)
                      with open(file_path, "r") as content_file:
                          content = content_file.read()
                          ex = ex.replace(match, content)
                          #print(f"Content of the file for link {link}:")
                          #print(content)


                  # Check if link is in the validation split
                  if link in dataset["validation"]["justia_link"]:
                      print(True)
                      validation_index = np.where(np.array(dataset["validation"]["justia_link"]) == link)[0][0]
                      file = get_file(dataset["validation"][int(validation_index)], "SCOTUS")
                      # Read the content of the corresponding file
                      file_path = os.path.join(file_path, file)
                      with open(file_path, "r") as content_file:
                          content = content_file.read()
                          ex = ex.replace(match, content)
                          #print(f"Content of the file for link {link} in validation split:")
                          #print(content)
            
          ex_no_html = remove_html_tags(ex)
          # Send the modified text to the chat model
          response = client.chat(
              model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
              messages=[
                  {
                      'role': 'user',
                      'content': "Resume the content of the following text while keeping all the information: " + ex_no_html,
                  },
              ],
              options={"num_ctx": 128000, "num_gpu": 60}
          )
          
          # Print the model's response
          #print(response['message']['content'])
          
          # Define the output file name and save
          file = get_file(dataset[run][i], parameters["dataset_name"])
          with open(os.path.join(save_path, file), "w") as f:
              f.write(response['message']['content'])

































          '''
          for match in all_matches:
              link_match = re.search(r'href\s*=\s*["\']([^"\']+)["\']', match)
              if link_match:
                  link = link_match.group(1)
                  if not link.startswith("https://"):
                      link = "https://supreme.justia.com" + link

                  # Check in train links
                  if link in train_links_dict:
                      train_index = train_links_dict[link]
                      file = get_file(dataset["train"][train_index], "SCOTUS")
                      file_path = os.path.join(save_path, file)
                      with open(file_path, "r") as content_file:
                          content = content_file.read()
                          ex = ex.replace(match, content)
                          print(f"Content of the file for link {link} (train):")
                          print(content)

                  # Check in validation links
                  elif link in validation_links_dict:
                      validation_index = validation_links_dict[link]
                      file = get_file(dataset["validation"][validation_index], "SCOTUS")
                      file_path = os.path.join(save_path, file)
                      with open(file_path, "r") as content_file:
                          content = content_file.read()
                          ex = ex.replace(match, content)
                          print(f"Content of the file for link {link} (validation):")
                  
                          print(content)
          '''              
