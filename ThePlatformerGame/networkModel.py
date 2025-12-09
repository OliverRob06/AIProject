import torch.nn as nn  
import torch.nn.functional as nnf   

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # 7 is the input dimension, ie number of state space variables
        # 128 is the number of hidden layers
        # There are six discrete actions available, so 6 is the size of the output layer
        # Dropout will be set to 0.2, depending on underfitting or overfitting this may be altered.

        self.layer1 = nn.Linear(7, 128)
        self.layer2 = nn.Linear(128, 6)
        self.dropout = nn.Dropout(0.2)

    # Top level traversing data through NN
    # x is the current snapshot of the environment. Initially holding the inputs
    # This is by default called when the model is called: policy(observation)
    def forward(self, x):
        x = self.layer1(x) # Move the data from the input layer to the hidden layer
        x = self.dropout(x) # Applies dropout to nodes
        x = nnf.relu(x) # Introduces non-linearity. Forces -ve values to be exactly 0.
        x = self.layer2(x) # Move the data from the hidden layer (128) to the output layer (6).
        x = nnf.softmax(x, dim = -1) # Turns the output layer (6) into (6) probabilities all adding up to 1
        return x # returns 6 probabilities.

