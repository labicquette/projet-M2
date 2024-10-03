from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


def get_model(**parameters):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", max_length=130,min_length=30,forced_bos_token_id=0)
    return model

def get_tokenizer(**parameters):
    tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn",max_length=130,min_length=30,truncation=True, model_max_length=142)
    return tok

def inference(dataset, **parameters):

    run = parameters["run"]
    model = parameters["model"]
    tok = parameters["tokenizer"]

    pipe = pipeline("summarization",
                    model=model,
                    tokenizer=tok,
                    do_sample=False,
                    )

    examples = parameters["examples"]

    for i,res in tqdm(enumerate(pipe(KeyDataset(dataset[run], examples), truncation=True, max_length =130, min_length=30,do_sample=False, batch_size=4))):
        id_case = dataset[run][i]["justia_link"].split("/")[-3:-1]
        file = dataset[run][i]["case_name"].replace("/","")+ "-"+ id_case[0]+ "-"+ id_case[1]

        with open(parameters["save_path"]+file, "w") as f:
            f.write(res[0]["summary_text"])
            f.close()