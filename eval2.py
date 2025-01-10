from glob import glob
import os
from datasets import load_dataset
from source.utils import get_file, labels_dict
import evaluate

# Dictionnaires des datasets
dataset_paths = {"SCOTUS": "./source/load_SCOTUS.py"}
examples_dict = {"SCOTUS": "examples"}

# Fonction d'évaluation
def evaluation():
    # Chargement des métriques
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    sari = evaluate.load("sari")  # Ajout de SARI

    # Liste des modèles
    models = [f.path for f in os.scandir("./runs/Summary") if f.is_dir()]

    for m in models:
        dirs = [f.path for f in os.scandir(m) if f.is_dir()]

        for d in dirs:
            d_splits = d.split("/")
            name = d.split("/")[-1]
            model = d_splits[-2]
            ds = d_splits[-1]

            splits = [f.path for f in os.scandir(d) if f.is_dir()]
            for split in splits:
                # Chargement du dataset
                if "cnn" in split:
                    continue
                    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_paths[name], trust_remote_code=True)

                files = glob(split + "/*")

                predictions = []
                sources = []
                references = []
                if split.split("/")[-1] == ".ipynb_checkpoints":
                    continue
                for row in dataset[split.split("/")[-1]]:
                    
                    with open(split + "/" + get_file(row, name), "r") as f:
                        predictions.append(f.read())

                    sources.append(row[examples_dict[name]])
                    references.append(row[labels_dict[name]])

                # Calcul des scores ROUGE
                rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

                # Calcul des scores BERTScore (moyenne sur tout le dataset)
                bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")

                # Calcul des scores SARI
                sari_result = sari.compute(sources=sources, predictions=predictions, references=[[ref] for ref in references])

                # Affichage des résultats
                print(f"Model: {model}, Dataset: {ds}, Split: {split.split('/')[-1]}")
                print("ROUGE :", rouge_result)
                print("BERTScore (F1):", sum(bertscore_result['f1']) / len(bertscore_result['f1']))  # Moyenne des F1
                print("SARI :", sari_result)

    print("break")
    return 0

# Exécution du script
if __name__ == '__main__':
    evaluation() 