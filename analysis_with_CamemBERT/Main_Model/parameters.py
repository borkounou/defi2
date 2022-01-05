import torch 
'''
In this script, we are considering all constants and parameters in one block 
'''

MAX_LENGTH = 130
BATCH_SIZE = 8



# Path of train data
path_train = 'train.csv'
# Path of evaluation data
path_val = "dev.csv"

# Path of test data
path_test = "test.csv"

# path where we save the final predicted result: it is a text file

path_result = "simple_test.txt"


# Path of saved model
save_model = 'borkounouBERT.h5'

# set the cuda cpu if available

def device_GPU_CPU():
    # Configuration de GPU
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device
