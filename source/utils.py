

labels_dict = {"cnn_dailymail":"highlights",
               "SCOTUS":"labels"}


def get_file(dataset, name):
    if name == "SCOTUS":
        id_case = dataset["justia_link"].split("/")[-3:-1]
        file = dataset["case_name"].replace("/","")+ "-"+ id_case[0]+ "-"+ id_case[1]

    if name == "cnn_dailymail":
        file = dataset["id"] # to set 
    return file
