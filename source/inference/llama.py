from source.utils import get_file
import ollama
from ollama import Client
from tqdm import tqdm
import os
import torch
from transformers import pipeline, AutoTokenizer

def get_model(**parameters):
    client = Client(host='http://localhost:11434')
    return client

def get_tokenizer(**parameters):
    return "tok"

def mapped(ex, i, client, dataset, run, parameters):
  file = get_file(dataset[run][i], parameters["dataset_name"])
  if os.path.exists(parameters["save_path"]+file):
    return
  response = client.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M', messages=[
                {
                    'role': 'user',
                    'content': "Resume the content of the following text while keeping all the information" + ex,
                },
                ],
                options={"num_ctx":128000, "num_gpu":60})
  #print(response['message']['content'])
  file = get_file(dataset[run][i], parameters["dataset_name"])
  with open(parameters["save_path"]+file, "w") as f:
            f.write(response['message']['content'])
            f.close()

def inference(dataset, **parameters):

    run = parameters["run"]
    client = parameters["model"]

    examples = parameters["examples"]
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline("text-generation",
                    model=model_id,
                    tokenizer = tokenizer,
                    token=parameters["hf_token"],
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    max_new_tokens=20,
                    return_full_text =False,
                    add_special_tokens=True
                    )
    print(dataset)
    prompts = [ "Resume the content of the following text while keeping all the information" + t for t in dataset[run][examples]]
    #dataset[run].map(lambda ex, i : mapped(ex[examples], i, client, dataset, run, parameters), with_indices=True)
    #print(prompts)
    for out in tqdm(pipe(prompts,do_sample=True,temperature=0.6,top_p=0.9)):
        text = out[0]['generated_text']
        print(text)




    for i,ex in enumerate(tqdm(dataset[run][examples])):
        file = get_file(dataset[run][i], parameters["dataset_name"])
        if os.path.exists(parameters["save_path"]+file):
            continue
        response = client.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M', messages=[
                {
                    'role': 'user',
                    'content': "Resume the content of the following text while keeping all the information" + ex,
                },
                ],
                options={"num_ctx":128000, "num_gpu":60})
        print(response['message']['content'])
        file = get_file(dataset[run][i], parameters["dataset_name"])
        with open(parameters["save_path"]+file, "w") as f:
            f.write(response['message']['content'])
            f.close()
