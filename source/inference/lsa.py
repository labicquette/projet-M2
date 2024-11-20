from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from tqdm import tqdm
from source.utils import get_file
from sumy.utils import get_stop_words

def get_model(**parameters):
    stemmer = Stemmer("english")
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    return summarizer

def get_tokenizer(**parameters):
    return Tokenizer("english")

def inference(dataset, **parameters):

    run = parameters["run"]
    model = parameters["model"]
    tok = parameters["tokenizer"]

    examples = parameters["examples"]

    nb_sentences = 5

    for i,ex in enumerate(tqdm(dataset[run][examples])):
        summary = model(PlaintextParser(ex,tok).document, nb_sentences)
        temp_summ = ""
        for s in summary:
            temp_summ += s._text + "\n"
        file = get_file(dataset[run][i], parameters["dataset_name"])
        with open(parameters["save_path"]+file, "w") as f:
            f.write(temp_summ)
            f.close()