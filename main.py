from source.inference import bart


def main():
    runs = [bart,
            #pegasus,
            #Llama
            ]


    datasets = [
                #GLUE
            ]
    
    parameters = {}

    for run in runs:
        for dataset in datasets:
            run.inference(dataset, **parameters)

if __name__ == '__main__':
    main()