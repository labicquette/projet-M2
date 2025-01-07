# from source.inference import bart, llama, lsa, legal_pegasus
from source.inference import rag_llama

from datasets import load_dataset, disable_caching
import time
import os
import torch

def main():
    #disable_caching()
    models = [#bart,
              #llama,
              #lsa,
              #legal_pegasus,
              rag_llama,
            #Llama
            ]

    #TODO include CNN/DailyMail and Multi-LexSum
    datasets = [
                #load_dataset("abisee/cnn_dailymail","3.0.0",trust_remote_code=True),
                load_dataset("./source/load_SCOTUS.py",trust_remote_code=True)
            ]
    datasets_names = [
                      #"cnn_dailymail", 
                      "SCOTUS"
                      ]

    #print(len(datasets[0]['validation'][215]["opinion_texts_source"][3]))
    #print(list(datasets[0]['train'][0]))
    examples_dict = {
                    # "cnn_dailymail":"article",
                    "SCOTUS":"examples"
                    }

    labels_dict = {
                    # "cnn_dailymail":"highlights",
                    "SCOTUS":"labels"
                    }

    parameters = {
        "run":"validation"
        }
    
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    curr_time = "Summary"
    if not os.path.exists("./runs/"+curr_time):
        os.makedirs("./runs/"+curr_time)

    #TODO add LSA,CaseSumm,Pegasus

    for model in models:
        parameters["save_path"] = "./runs/"+"Summary"+"/"+model.__name__.split(".")[-1]
        if not os.path.exists("./runs/"+curr_time+"/"+model.__name__.split(".")[-1]):
            os.makedirs("./runs/"+curr_time+"/"+model.__name__.split(".")[-1])
        
        # parameters["model"] = model.get_model(**parameters)
        # parameters["tokenizer"] = model.get_tokenizer(**parameters)
        
        for dataset, name in zip(datasets, datasets_names):
            
            parameters["dataset_name"] = name
            parameters["examples"] = examples_dict[name]
            parameters["labels"] = labels_dict[name]
            
            if not os.path.exists("./runs/"+curr_time+"/"+model.__name__.split(".")[-1]+"/"+name):
                os.makedirs("./runs/"+curr_time+"/"+model.__name__.split(".")[-1]+"/"+name)
            for run in list(dataset.keys()):
                if run == "train" or run == "test":
                    continue
                if not os.path.exists("./runs/"+curr_time+"/"+model.__name__.split(".")[-1]+"/"+name+"/"+run):
                    os.makedirs("./runs/"+curr_time+"/"+model.__name__.split(".")[-1]+"/"+name+"/"+run)
                
                parameters["run"] = run
                print("Launching : ", model.__name__.split(".")[-1], name)
                parameters["save_path"] = "./runs/"+curr_time+"/"+model.__name__.split(".")[-1]+"/"+name+"/"+run+"/"
                model.inference(dataset, **parameters)                   

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main()