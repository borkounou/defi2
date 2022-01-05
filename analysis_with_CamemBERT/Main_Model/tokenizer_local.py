from transformers import CamembertTokenizer
import torch 
from classifier import path_modele

# The below function tokenizes the data and it has two arguments: data and max_length
def tokenization(data, max_length):

    '''
    args:
        - data: a pandas dataframe
        - max_length: int
    
    return:
        - input_ids: torch
        - attention_mask: torch

    '''
    print("Starting tokenization...")

    tokenizer = CamembertTokenizer.from_pretrained(path_modele)

    input_ids = []
    attention_mask = []
    # Encoding every sentence
    for element in data:
        encoded_element = tokenizer.encode_plus(str(element), add_special_tokens=True, 
                                                truncation=True, max_length=max_length,
                                                padding='max_length', return_tensors='pt')
        input_ids.append(encoded_element["input_ids"])
        attention_mask.append(encoded_element["attention_mask"])
    print("Tokenization finished !")

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    # return the input_ids and attention_mask
    return input_ids, attention_mask