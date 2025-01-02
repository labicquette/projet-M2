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
"""
# Regex pattern to extract links from text
link_pattern = r'<a href="(/cases/federal/us/\d+/\d+/)".*?>.*?</a>'


def inference(dataset, **parameters):
    base_url = "https://supreme.justia.com"  # Base URL for constructing full links
    run = parameters["run"]  # Specifies which part of the dataset to process
    client = parameters["model"]  # AI model object for text summarization
    examples = parameters["examples"]  # Subset of examples to process
    save_path = parameters["save_path"]  # Path to save final processed summaries

    for i, ex in enumerate(tqdm(dataset[run][examples])):  # Iterate over the dataset entries
        # Check if `ex` is a string or dictionary
        if isinstance(ex, dict):
            original_text = ex["examples"]  # Extract the original text for processing
            justia_link = ex.get("justia_link")  # Extract the link if available
        elif isinstance(ex, str):
            original_text = ex  # If `ex` is just a string, treat it as the input text
            justia_link = None  # No link available in this case
        else:
            raise TypeError("Unexpected structure for dataset entry: Must be a dict or string.")

        links = re.findall(link_pattern, original_text)  # Extract relative links from the text
        full_links = [base_url + link for link in links]  # Convert relative links to full URLs
        processed_text = original_text  # Initialize the processed text with the original content

        for full_link in full_links:  # Iterate over each extracted link
            linked_entry = next((item for item in dataset[run] if isinstance(item, dict) and item.get("justia_link") == full_link), None)  # Find matching link in the dataset
            if linked_entry:  # If a matching entry is found
                linked_text = linked_entry["examples"]  # Retrieve the text associated with the link
                # Summarize the linked text using the AI model
                summary_response = client.chat(
                    model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
                    messages=[
                        {
                            'role': 'user',
                            'content': "You are an expert at writing short summaries. Your job is to summarize the text below in exactly 3 SHORT SENTENCES. Focus only on the most important information and leave out everything extra in order to make 3 SHORT SENTENCES.Make sure it is clear, simple, and easy to understand with 3 SHORT SENTENCES. Do not add unnecessary details, and do not make it longer than 3 SHORT SENTENCES.Here's the text: " + linked_text +" Start: with this text name is [fill blank] its core message is ...",
                        },
                    ]
                )
                summary = summary_response['message']['content']  # Extract the summary from the AI response
                print("--- BEGIN SMALL SUMMARY--- \n"+ summary +"\n --- END SMALL SUMMARY--- ") # print small summary
                # Replace the link in the text with the summarized content
                processed_text = processed_text.replace(
                    f'<a href="{full_link}">link</a>',
                    summary
                )

        # Generate the final summary for the processed text using the AI model
        final_summary_response = client.chat(
            model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
            messages=[
                {
                    'role': 'user',
                    'content': "Resume the content of the following text while keeping all the information: " + processed_text,
                },
            ]
        )
        final_summary = final_summary_response['message']['content']  # Extract the final summary
        print(final_summary)  # Print the final summary

        # Save the final summary to a file
        file_name = get_file(ex, parameters["dataset_name"]) if isinstance(ex, dict) else f"entry_{i}.txt"  # Generate a unique file name
        with open(save_path + file_name, "w") as f:  # Open the file in write mode
            f.write(final_summary)  # Write the final summary to the file
            f.close()  # Close the file
"""

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
        if isinstance(ex, dict):  # Handle dictionary entries for flexibility in dataset structure
            original_text = ex["examples"]  # Extract the text directly for easier processing
            justia_link = ex.get("justia_link")  # Get the link if it exists, to name output files
        elif isinstance(ex, str):  # Handle string entries for uniform processing
            original_text = ex  # Treat the string as the main text to process
            justia_link = None  # No link associated with plain strings
        else:
            raise TypeError("Unexpected structure for dataset entry: Must be a dict or string.")  # Error handling for robustness

        links = re.findall(link_pattern, original_text)  # Extract all links from the text for summarization
        full_links = [base_url + link for link in links]  # Convert relative links to absolute for uniformity

        for full_link in full_links:  # Process each extracted link separately for better granularity
            linked_entry = next(
                (item for item in dataset[run] if isinstance(item, dict) and item.get("justia_link") == full_link), 
                None
            )  # Find the corresponding entry in the dataset for link integrity

            if linked_entry:  # Ensure that a corresponding entry exists before proceeding
                linked_text = linked_entry["examples"]  # Retrieve the text to summarize it
                summary_response = client.chat(  # Use the model to generate a summary for scalability
                    model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',  # Allow specification of model for flexibility
                    messages=[
                        {
                            'role': 'user',
                            'content': (
                                "You are an expert at writing short summaries. Your job is to summarize the text below in exactly "
                                "3 SHORT SENTENCES. Focus only on the most important information and leave out everything extra in order "
                                "to make 3 SHORT SENTENCES. Here's the text: " + linked_text
                            )
                        },
                    ]
                )
                small_summary = summary_response['message']['content']  # Extract the model output for reuse

                # Save the small summary to a file
                file_name = f"{justia_link.replace('/', '_')}.txt" if justia_link else f"link_summary_{hash(full_link)}.txt"  
                # Use `justia_link` to create unique and recognizable file names
                with open(os.path.join(save_path, file_name), "w") as f:  # Open file in write mode to save summary
                    f.write(small_summary)  # Save the summary for reuse in subsequent steps


def generate_final_summaries(dataset, **parameters):
    base_url = "https://supreme.justia.com"  # Base URL is needed to match full links
    run = parameters["run"]  # Dynamically select which dataset subset to process
    client = parameters["model"]  # Use a configurable model client for flexibility
    examples = parameters["examples"]  # Allow specifying which examples to process
    save_path = parameters["save_path"]  # Define where the final summaries are saved
    summary_files_path = parameters["summary_files_path"]  # Specify location of small summaries

    if not os.path.exists(save_path):  # Ensure the output directory exists before saving
        os.makedirs(save_path)  # Create the output directory if it doesn’t exist

    for i, ex in enumerate(tqdm(dataset[run][examples])):  # Use progress tracking for transparency
        if isinstance(ex, dict):  # Handle dictionary entries for dataset flexibility
            original_text = ex["examples"]  # Extract the main text for processing
        elif isinstance(ex, str):  # Handle string entries for uniformity
            original_text = ex  # Treat the string as the main text
        else:
            raise TypeError("Unexpected structure for dataset entry: Must be a dict or string.")  # Error handling for robustness

        links = re.findall(r'<a href="(/cases/federal/us/\d+/\d+/)".*?>.*?</a>', original_text)  # Extract links for processing
        full_links = [base_url + link for link in links]  # Convert relative links to absolute for uniformity
        processed_text = original_text  # Initialize processed text to modify it in-place

        for full_link in full_links:  # Iterate over links for individual replacement
            summary_file_name = f"{full_link.replace('/', '_')}.txt"  # Generate the corresponding file name
            summary_file_path = os.path.join(summary_files_path, summary_file_name)  # Build the full file path

            if os.path.exists(summary_file_path):  # Ensure the summary file exists before using it
                with open(summary_file_path, "r") as f:  # Open the file to read its contents
                    small_summary = f.read()  # Read the summary for replacing the link

                processed_text = processed_text.replace(  # Replace the link with the small summary
                    f'<a href="{full_link}">link</a>',
                    small_summary
                )

        # Generate the final summary
        final_summary_response = client.chat(  # Use the model for creating the final summary
            model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',  # Allow model configuration
            messages=[
                {
                    'role': 'user',
                    'content': (
                        "Resume the content of the following text while keeping all the information: " + processed_text
                    ),
                },
            ]
        )
        final_summary = final_summary_response['message']['content']  # Extract the final summary from the model

        # Save the final summary to a file
        file_name = f"Summary_opi_{i + 1}.txt"  # Use sequential naming for clarity and uniqueness
        with open(os.path.join(save_path, file_name), "w") as f:  # Open file in write mode to save the final summary
            f.write(final_summary)  # Save the summary for future use or analysis

def inference(dataset, **parameters):
  # Check if the user wants to run the first function
        print("Executing extract_and_summarize_citations...")
        extract_and_summarize_citations(dataset, **parameters)  # Call the first function

        #print("Executing generate_final_summaries...")
       # generate_final_summaries(dataset, **parameters)  # Call the second function



         
        
#TODO foreach prompt we need to separate text and citations with regex
#TODO put text in one list and regex on another
#TODO cocatenate text and regex in one prompt 
#TODO can you dataset[run] of i to have more info on dataset 
#TODO if rerun shows error clean data in runs directory