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

    # Extract the lists
    justia_links = dataset[run]["justia_link"]
    case_names = dataset[run]["case_name"]

    # Create a dictionary mapping case names to justia links
    case_dict = dict(zip(case_names, justia_links))


    # Print the resulting dictionary
    print(case_dict)
    for i,ex in enumerate(tqdm(dataset["train"][examples])):    
          response = client.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M', messages=[
                  {
                      'role': 'user',
                      'content': "Resume the content of the following text while keeping all the information" + ex,
                  },
                  ],
                  options={"num_ctx":128000,"num_gpu":60})
          print(response['message']['content'])

          file = get_file(dataset[run][i], parameters["dataset_name"])
          with open(parameters["save_path"]+file, "w") as f:
              f.write(response['message']['content'])
              f.close()

