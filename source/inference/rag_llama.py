from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from source.utils import get_file
from langchain_community.embeddings import HuggingFaceEmbeddings

def inference(dataset, **parameters):
    """
    Effectue l'inférence sur un dataset en utilisant ChatOllama avec une approche RAG.

    Args:
        dataset (dict): Le dataset contenant les données d'entrée.
        **parameters: Paramètres pour la configuration.

    Returns:
        None. Les résultats sont enregistrés dans des fichiers.
    """


    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Extraire les paramètres
    run = parameters["run"]
    examples = parameters["examples"]
    save_path = parameters["save_path"]
    dataset_name = parameters["dataset_name"]

    # Initialiser la base de vecteurs
    vector_store = Chroma(persist_directory="/content/projet-M2/data/chroma_db", embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # Construire le template du prompt
    prompt_template = """
    Vous êtes un expert en résumé de texte légal dans le cadre de procès pour prendre des décisions de justice.
    Rédigez un résumé clair et concis adapté à un professionnel du droit.
    Utilisez un langage précis et formel, en capturant les points clés du texte.

    Contexte : {context}
    Question : {question}
    Réponse
    """
    generation_prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialiser ChatOllama
    llm = ChatOllama(
        model="llama3.2",
        temperature=0  # Ajustez selon vos besoins
    )

    # Boucle d'inférence
    for i, ex in enumerate(tqdm(dataset[run][examples], desc="Processing examples")):
        # Générer la requête utilisateur
        user_query = f"résumer le cas juridique suivant : {ex}"

        # Récupérer les documents pertinents à partir de la base de vecteurs
        relevant_chunks = retriever.invoke(user_query)
        context_text = "\n\n---\n\n".join([result.page_content for result in relevant_chunks])

        # Construire le prompt complet avec le contexte
        prompt = generation_prompt.format(context=context_text, question=user_query)

        # Envoyer le prompt au modèle ChatOllama
        response = llm.predict(prompt)

        # Formater la réponse et sauvegarder
        # sources = list(set([result.metadata.get("source", None) for result in relevant_chunks]))
        # formatted_response = f"Réponse: {response}\n\nSources: {sources}"
        # print(formatted_response)

        # Sauvegarder la réponse dans un fichier
        file = get_file(dataset[run][i], dataset_name)
        with open(save_path + file, "w") as f:
            f.write(response)
