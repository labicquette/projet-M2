import os
from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm
from bs4 import BeautifulSoup


import re
from tqdm import tqdm

def remove_html_tags(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def find_case_name_in_directory(directory_path, case_dict):
    # List all files in the directory
    files_in_directory = os.listdir(directory_path)

    # Create a mapping of lowercase filenames to their original names for case-insensitive matching
    file_mapping = {file.lower(): file for file in files_in_directory}

    # Find case names that match filenames
    matching_cases = {}
    for case_name in case_dict:
        # Replace spaces in case names with underscores and construct expected filename
        expected_file = f"{case_name.lower()}.txt"
        print(expected_file)
        if expected_file in file_mapping:
            matching_cases[case_name] = file_mapping[expected_file]

    return matching_cases

def inference(dataset, **parameters):
    run = parameters["run"]
    client = parameters["model"]
    examples = parameters["examples"]
    file_directory = "/content/drive/MyDrive/projet_m2/data/small_sum_validation/" 
    save_path = parameters["save_path"]
    
    justia_links = dataset[run]["justia_link"]
    case_names = dataset[run]["case_name"]
    case_dict = dict(zip(case_names, justia_links))
    
    # Find matching filenames in the directory
    matching_cases = find_case_name_in_directory(file_directory, case_dict)
    
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
                
                # Check if the link exists in case_dict
                case_name = next((name for name, justia_link in case_dict.items() 
                                if justia_link == link), None)
                
                if case_name:
                    # Check if the case name has a corresponding file
                    if case_name in matching_cases:
                        file_path = os.path.join(file_directory, matching_cases[case_name])
                        try:
                            with open(file_path, "r") as file:
                                file_content = file.read()
                        except FileNotFoundError:
                            file_content = "[FILE NOT FOUND]"
                        # Replace the link in the example with the file content
                        ex = ex.replace(match, file_content)
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
        print(response['message']['content'])
        
        # Define the output file name and save
        file = get_file(dataset[run][i], parameters["dataset_name"])
        with open(os.path.join(save_path, file), "w") as f:
            f.write(response['message']['content'])