from glob import glob
import os
from datasets import load_dataset
from source.utils import get_file, labels_dict
import evaluate
from source.sari import Sari
#rouge = evaluate.load("rouge")
#sari = evaluate.load("sari")

dataset_paths = {"SCOTUS": "./source/load_SCOTUS.py",
                 "cnn_dailymail" :"abisee/cnn_dailymail",}
examples_dict = {
                    "cnn_dailymail":"article",
                    "SCOTUS":"examples"
                    }


def evaluation():
    rouge = evaluate.load("rouge")
    models = dirs  = [ f.path for f in os.scandir("./runs/Summary") if f.is_dir() ]
    for m in models:
        dirs  = [ f.path for f in os.scandir(m) if f.is_dir() ]
        for d in dirs:
            d_splits = d.split("/")
            name = d.split("/")[-1]
            model = d_splits[-2]
            ds = d_splits[-1]
            #print(model, ds)
            splits  = [ f.path for f in os.scandir(d) if f.is_dir() ]
            for split in splits :
                print(split)
                if "cnn" in split:
                    continue
                    dataset = load_dataset("abisee/cnn_dailymail","3.0.0",trust_remote_code=True) 
                else:
                    dataset = load_dataset(dataset_paths[name],trust_remote_code=True)
                files = glob(split + "/*")
                column = []
                for row in dataset[split.split("/")[-1]] :
                    with open(split+"/"+get_file(row, name), "r")as f:
                        column += [f.read()]
                dataset[split.split("/")[-1]].add_column("inferred", column)
                sources = dataset[split.split("/")[-1]][examples_dict[name]]
                references = dataset[split.split("/")[-1]][labels_dict[name]]
                
                result = rouge.compute(predictions=column, references=references, use_stemmer=True)
                print("ROUGE RESULTS : ", result)
                res = "\n" + model + ds + split.split("/")[-1] + str(result["rouge1"]) + "," + str(result["rouge2"]) +  "," + str(result["rougeL"]) +  "," + str(result["rougeLsum"])
                
                with open("./results.csv", "a") as file:
                    file.write(res)
    

    
    print("break")


    return 0

if __name__ == '__main__':
    with open("./results.csv", "w") as file:
        file.write("model,dataset,run,rouge1,rouge2,rougeL,rougeLsum")
    evaluation()