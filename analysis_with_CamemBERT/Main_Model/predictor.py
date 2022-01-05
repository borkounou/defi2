'''
Important packages and librairies
'''
from numpy.core.numeric import False_
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
# local packages
from classifier import MainClassifier, path_modele

from tokenizer_local import tokenization
from parameters import MAX_LENGTH, BATCH_SIZE, save_model, path_test,path_result, device_GPU_CPU

'''
Check if the cuda GPU is avaible. If so, set de device to cuda otherwise the it is set to CPU
'''

device = device_GPU_CPU()

'''
The below function takes inputs args:
        - model
        - test_dataloader
and outputs:
    - probs: array of float
'''

def predict_test(model, test_dataloader):
 #   model.eval()

    all_logits = [] # an empty list

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        # Add the predicted outputs of the batch to the list
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to compute probabilities 
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs

'''
test_processing: is the function which deals with processing of the test data.
args:
    - path_test: String==> the path of the dataset

return:
    - df: a pandas dataframe
    - dataloader_test: a dataloader compatible with the format

'''
def test_processing(path_test):
    # Dataframe
    df = pd.read_csv(path_test, sep=',', encoding='utf-8')
    df.Commentaire = df.Commentaire.fillna('-')
    df = df[['Review_id','Commentaire']]
    df = df.dropna()
  
    X_test = df['Commentaire']
    # Convert into numpy array
    X_test = np.array(X_test)
    # Tokenizer 
    input_ids_test, attention_masks_test = tokenization(X_test, MAX_LENGTH)
    # Load into tensorDataset
    dataset_test= TensorDataset(input_ids_test, attention_masks_test)
    # Dataloader
    dataloader_test =  DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=BATCH_SIZE)
    return df, dataloader_test


'''
The below function takes model and the path of the test data as input arguments.
it saves and display the predicted result as text file
'''

def test_executable(model, path_test):
    '''
    args: 
        - model: classifier
        - path_test: String==> the path of the dataset
    
    '''

    labels_index ={'0,5':0, '1,0':1, '1,5':2, '2,0':3, '2,5':4, '3,0':5, '3,5':6, '4,0':7, '4,5':8, '5,0':9}

    reversed_dict= {}
    dataframe, dataloader_test = test_processing(path_test)
    probs = predict_test(model, dataloader_test)
    preds=  np.argmax(probs, axis=1)
    dataframe['predict'] = preds
    for k in labels_index:
        reversed_dict[labels_index[k]]=k

    dataframe['predict'] = dataframe['predict'].replace(reversed_dict)
    # Saving the dataframe as a text file compatible with the required format of test data
    np.savetxt(path_result, dataframe[['Review_id','predict']] ,delimiter=" ", fmt='%s')
    print(f"The file of predicted result is saved in this path: {path_result}. Check it out!")
  
    print("Test completed!!!!")


# =======================================
#          Computation 
# =======================================
def predictor_local():

    model = MainClassifier()
    model.to(device)
    model.load_state_dict(torch.load(save_model))
    test_executable(model,path_test)


# if __name__ == "__main__":
#     predictor_local()


# =======================================
# =================END===================
# =======================================


