from source.inference import bart
from datasets import load_dataset

def main():
    runs = [bart,
            #pegasus,
            #Llama
            ]


    datasets = [
                load_dataset("./source/load_SCOTUS.py")
            ]

    parameters = {}

    for run in runs:
        for dataset in datasets:
            run.inference(dataset, **parameters)

if __name__ == '__main__':
    main()