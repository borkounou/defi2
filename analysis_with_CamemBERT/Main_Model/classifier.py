import torch.nn as nn
import torch 
from transformers import CamembertModel
from transformers import AdamW, get_linear_schedule_with_warmup

# We use the pretrained model camembert for finetuning
path_modele ="camembert-base"
# Classifer

class MainClassifier(nn.Module):
    def __init__(self):
        super(MainClassifier, self).__init__()
        # Pretrained model
        self.camembert = CamembertModel.from_pretrained(path_modele)
        # Hidden layer
        self.fc1 = nn.Linear(self.camembert.config.hidden_size, 200)
        # Dropout regularizer
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm = nn.BatchNorm1d(self.camembert.config.hidden_size)
        # Output of ten labels
        self.out = nn.Linear(200, 10)

    def forward(self, input_ids, input_masks):
        camembert = self.camembert(input_ids, input_masks)
        camembert = camembert[0]
        x = camembert[:, 0, :]
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        output = self.out(x)

        return output




# Initialization of the model 

def initialize_model(dataloader_train, device,epochs=3):
    # Instantiate camembert Classifier
    classifier = MainClassifier()

    # Tell PyTorch to run the model on GPU
    classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(dataloader_train) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return classifier, optimizer, scheduler
        

    