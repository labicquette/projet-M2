import os
from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm
from bs4 import BeautifulSoup


import re
from tqdm import tqdm

def inference(dataset, **parameters):
    run = parameters["run"]
    client = parameters["model"]
    examples = parameters["examples"]
    file_directory = parameters["file_directory"]  # Directory where case files are stored
    save_path = parameters["save_path"]  # Directory to save the output files
    dataset_name = parameters["dataset_name"]  # Dataset name for file naming

    # Extract the lists
    justia_links = dataset[run]["justia_link"]
    case_names = dataset[run]["case_name"]

    # Create a dictionary mapping case names to Justia links
    case_dict = dict(zip(case_names, justia_links))

    # Function to read the content of a file given the case name
    def get_file_content(case_name):
        file_path = f"{file_directory}/{case_name}.txt"  # Assuming files are named based on case_name
        try:
            with open(file_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            return "[FILE NOT FOUND]"

    # Process each example in the dataset
    for i, ex in enumerate(tqdm(dataset[run][examples])):

        # Replace links in the text with corresponding file contents
        for case_name, justia_link in case_dict.items():

            # Escape the Justia link for regex matching
            escaped_link = re.escape(justia_link)

            # Define regex patterns for regular and escaped quotes
            regular_pattern = f'<a href="/cases/federal/us/\d+/\d+/">.*?</a>'
            escaped_pattern = f'<a href=\s*\\?"/cases/federal/us/\d+/\d+/\\?">.*?</a>'

            # Get the content of the file associated with the case name
            file_content = get_file_content(case_name)

            # Replace the matched links with the file content
            ex = re.sub(regular_pattern, file_content, ex)
            ex = re.sub(escaped_pattern, file_content, ex)

        # Send the modified text to the chat model
        response = client.chat(
            model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
            messages=[
                {
                    'role': 'user',
                    'content': "Resume the content of the following text while keeping all the information: " + ex,
                },
            ],
            options={"num_ctx": 128000, "num_gpu": 60}
        )

        # Print the model's response
        print(response['message']['content'])
        
        # Define the output file name
        file = get_file(dataset[run][i], parameters["dataset_name"])
        with open(parameters["save_path"]+file, "w") as f:
            f.write(response['message']['content'])
            f.close()

