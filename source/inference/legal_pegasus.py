
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from source.utils import get_file

def get_model(**parameters):
    model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to("cuda")
    return model

def get_tokenizer(**parameters):
    tok = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
    return tok

def inference(dataset, **parameters):

    run = parameters["run"]
    model = parameters["model"]
    tok = parameters["tokenizer"]

    examples = parameters["examples"]

    for i,ex in enumerate(tqdm(dataset[run][examples])):
        #id_case = dataset[run][i]["justia_link"].split("/")[-3:-1]
        #file = dataset[run][i]["case_name"].replace("/","")+ "-"+ id_case[0]+ "-"+ id_case[1]
        #print(dataset[run][i])

        input_tokenized = tok.encode(ex, return_tensors='pt',max_length=1024,truncation=True).to("cuda")
        summary_ids = model.generate(input_tokenized,
                                  num_beams=9,
                                  no_repeat_ngram_size=3,
                                  length_penalty=2.0,
                                  min_length=150,
                                  max_length=250,
                                  early_stopping=True)
        summary = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        ### Summary Output

        print(summary)

        file = get_file(dataset[run][i], parameters["dataset_name"])
        with open(parameters["save_path"]+file, "w") as f:
            f.write(summary)
            f.close()