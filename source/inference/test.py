from datasets import load_dataset
import numpy as np
import os
import re
from tqdm import tqdm
from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm


def get_model(**parameters):
    client = Client(host='http://localhost:11434')
    return client

def get_tokenizer(**parameters):
    return "tok"

def inference(dataset, **parameters):

    run = parameters["run"]
    client = parameters["model"]

    examples = parameters["examples"]
    # Load dataset
    ds = load_dataset("/content/projet-M2/source/load_SCOTUS.py", trust_remote_code=True)
    
    save_path = "/content/drive/MyDrive/projet_m2/data/small_sum_final"
    # Load train and validation datasets
    dstrain = ds["train"]
    #print(dstrain["justia_link"])
    dsval = ds["validation"]

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
                
                # Check if link is in the lists dataset[run]["justia_link"] then:
                #if link in dataset["train"]["justia_link"]:
                #index = np.where(np.array(dsval["justia_link"]) == link)[0][0]
                
                file = get_file(dataset[run][np.where(np.array(dstrain["justia_link"]) == link)[0][0]], "SCOTUS")
                # Read the content of the corresponding file
                file_path = os.path.join(save_path, file)
                with open(file_path, "r") as content_file:
                    content = content_file.read()
                print(f"Content of the file for link {link}:")
                print(content)
