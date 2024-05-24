import snntorch as snn
from snntorch import spikegen
import torch
from torch import nn
import torch.nn.functional as F
from snntorch import surrogate
from snntorch import utils



class SMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMLP, self).__init__()
        """ This is an NLP model based on spikie neurons:
            -  """
        # Initialize layers
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.99, # Decay factor Beta
                            threshold=1, # U_{thr} threshold to generate a spike
                            learn_beta=False, # We don't want to learn the decay factor, just a hyperparameter
                            reset_mechanism='subtract', # We will substract the threshold voltage when a spike is generated (-R*U_{thr})
                            spike_grad=spike_grad)  # We will use a fast sigmoid as a surrogate gradient
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.99, threshold=1, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad,
                              output=True) # We want to get the output of the LIF neuron which is the membrane potential 

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x, mem1 = self.lif1(x, mem1) #
        x = self.fc2(x)
        x, mem2 = self.lif2(x, mem2)
        return x, mem2