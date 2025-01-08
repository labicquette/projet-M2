from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm
import re
import os  # Needed for file operations and creating directories

def get_model(**parameters):
    client = Client(host='http://localhost:11434')
    return client

def get_tokenizer(**parameters):
    return "tok"
'''
def extract_and_summarize_citations(dataset, **parameters):
    base_url = "https://supreme.justia.com"  # Set the base URL for constructing absolute links
    run = parameters["run"]  # Use a parameter to dynamically select the dataset subset
    client = parameters["model"]  # Make the model client configurable for flexibility
    examples = parameters["examples"]  # Allow selection of a subset of examples for processing
    save_path = parameters["save_path"]  # Allow the user to define where summaries are saved
    link_pattern = r'<a href="(/cases/federal/us/\d+/\d+/)".*?>.*?</a>'  # Use a regex to extract links for compatibility with HTML

    if not os.path.exists(save_path):  # Ensure the directory exists before saving files
        os.makedirs(save_path)  # Create the directory if it doesn’t exist

    for ex in tqdm(dataset[run][examples]):  # Iterate over the dataset for batch processing
        
        links = re.findall(link_pattern, ex)  # Handle both strings and dictionaries
        full_links = [base_url + link for link in links]  # Convert relative links to absolute for uniformity

        for full_link in full_links:  # Process each extracted link separately for better granularity
            linked_entry = next(
                (item for item in dataset[run] if isinstance(item, dict) and item.get("justia_link") == full_link), 
                None
            )  # Find the corresponding entry in the dataset for link integrity

            if linked_entry:  # Ensure that a corresponding entry exists before proceeding
                linked_text = linked_entry["examples"]  # Retrieve the text to summarize it
                summary_response = client.chat(  # Use the model to generate a summary for scalability
                    model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
                    options={"num_ctx": 115000},  # Allow specification of model for flexibility
                    messages=[
                        {
                            'role': 'user',
                            'content': (
                                "You are an expert at writing short summaries. Your job is to summarize the text below in exactly "
                                "3 SHORT SENTENCES. Focus only on the most important information and leave out everything extra in order "
                                "to make 3 SHORT SENTENCES. Here's the text: " + linked_text+ "make sure to make 3 SHORT SENTENCES.Not more"
                            )
                        },
                    ]
                )
                small_summary = summary_response['message']['content']  # Extract the model output for reuse
                print(small_summary)
                # Save the small summary to a file
                file_name = f"{full_link.replace('/', '_').replace('.', '_')}.txt"
                with open(os.path.join(save_path, file_name), "w") as f:  # Open file in write mode to save summary
                    f.write(small_summary)  # Save the summary for reuse in subsequent steps
'''

def extract_and_summarize_citations(dataset, **parameters):
    base_url = "https://supreme.justia.com"  # Set the base URL for constructing absolute links
    run = parameters["run"]  # Use a parameter to dynamically select the dataset subset
    client = parameters["model"]  # Make the model client configurable for flexibility
    examples = parameters["examples"]  # Allow selection of a subset of examples for processing
    save_path = parameters["save_path"]  # Allow the user to define where summaries are saved
    link_pattern = r'<a href="(/cases/federal/us/\d+/\d+/)".*?>.*?</a>'  # Use a regex to extract links for compatibility with HTML

    if not os.path.exists(save_path):  # Ensure the directory exists before saving files
        os.makedirs(save_path)  # Create the directory if it doesn’t exist

    for ex in tqdm(dataset[run][examples][175:]):  # Iterate over the dataset for batch processing            
        links = re.findall(link_pattern, ex)  # Handle both strings and dictionaries
        full_links = [base_url + link for link in links]  # Convert relative links to absolute for uniformity

        for full_link in full_links:  # Process each extracted link separately for better granularity
            linked_entry = next(
                (item for item in dataset[run] if isinstance(item, dict) and item.get("justia_link") == full_link), 
                None
            )  # Find the corresponding entry in the dataset for link integrity

            if linked_entry:  # Ensure that a corresponding entry exists before proceeding
                linked_text = re.sub(r'<[^>]+>', '', linked_entry["examples"])  # Remove all HTML tags from linked text except links
                summary_response = client.chat(  # Use the model to generate a summary for scalability
                    model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
                    options={"num_ctx": 115000},  # Allow specification of model for flexibility
                    messages=[
                        {
                            'role': 'user',
                            'content': (
                                "You are an expert at writing short summaries. Your job is to summarize the text below in exactly "
                                "3 SHORT SENTENCES. Focus only on the most important information and leave out everything extra in order "
                                "to make 3 SHORT SENTENCES. Here's the text: " + linked_text+ "make sure to make 3 SHORT SENTENCES.Not more"
                            )
                        },
                    ]
                )
                small_summary = summary_response['message']['content']  # Extract the model output for reuse
                # Save the small summary to a file
                file_name = f"{full_link.replace('/', '_').replace('.', '_')}.txt"
                with open(os.path.join(save_path, file_name), "w") as f:  # Open file in write mode to save summary
                    f.write(small_summary)  # Save the summary for reuse in subsequent steps



def generate_final_summaries(dataset, **parameters):
    base_url = "https://supreme.justia.com"  # Base URL is needed to match full links
    run = parameters["run"]  # Dynamically select which dataset subset to process
    client = parameters["model"]  # Use a configurable model client for flexibility
    examples = parameters["examples"]  # Allow specifying which examples to process
    save_path = parameters["save_path"]  # Define where the final summaries are saved
    summary_files_path = "/content/drive/MyDrive/projet_m2/runs/Summary/llama_more/SCOTUS/validation/small_sum_2/"# Specify location of small summaries

    if not os.path.exists(save_path):  # Ensure the output directory exists before saving
        os.makedirs(save_path)  # Create the output directory if it doesn’t exist

    for i, ex in enumerate(tqdm(dataset[run][examples][98:])):  # Use progress tracking for transparency
        links = re.findall(r'<a href="(/cases/federal/us/\d+/\d+/)".*?>.*?</a>',ex)  # Extract links for processing
        full_links = [base_url + link for link in links]  # Convert relative links to absolute for uniformity
        processed_text = ex  # Initialize processed text to modify it in-place

        for full_link in full_links:  # Iterate over links for individual replacement
            summary_file_name = f"{full_link.replace('/', '_').replace('.', '_')}.txt"  # Generate the corresponding file name
            summary_file_path = os.path.join(summary_files_path, summary_file_name)  # Build the full file path

            if os.path.exists(summary_file_path):  # Ensure the summary file exists before using it
                with open(summary_file_path, "r") as f:  # Open the file to read its contents
                    small_summary = f.read()  # Read the summary for replacing the link

                processed_text = processed_text.replace(  # Replace the link with the small summary
                    f'<a href="{full_link}">link</a>',
                    small_summary
                )
        no_html = re.sub(r'<[^>]+>', '', processed_text)
        # Generate the final summary
        final_summary_response = client.chat(  # Use the model for creating the final summary
            model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
            options= {"num_ctx":115000},
            messages=[
                {
                    'role': 'user',
                    'content': (
                        "You are a lawyer and you need to make a summary of the following legal text while keeping all the information: " + no_html + 
                        "make sure to include the information about the case opinions and decisions taken "
                    ),
                },
            ]
        )
        final_summary = final_summary_response['message']['content']  # Extract the final summary from the model

        # Save the final summary to a file
        file_name =get_file(dataset[run][i], parameters["dataset_name"])  # Use sequential naming for clarity and uniqueness
        with open(os.path.join(save_path, file_name), "w") as f:  # Open file in write mode to save the final summary
            f.write(final_summary)  # Save the summary for future use or analysis

def inference(dataset, **parameters):
  # Check if the user wants to run the first function
        #print("Executing extract_and_summarize_citations...")
        #extract_and_summarize_citations(dataset, **parameters)  # Call the first function

        print("Executing generate_final_summaries...")
        generate_final_summaries(dataset, **parameters)  # Call the second function



         
        
#TODO foreach prompt we need to separate text and citations with regex
#TODO put text in one list and regex on another
#TODO cocatenate text and regex in one prompt 
#TODO can you dataset[run] of i to have more info on dataset 
#TODO if rerun shows error clean data in runs directory