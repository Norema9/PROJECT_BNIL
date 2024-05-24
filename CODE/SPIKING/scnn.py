import snntorch as snn
from snntorch import spikegen
import torch
from torch import nn
import torch.nn.functional as F
from snntorch import surrogate
from snntorch import utils



class SimpleCNN(nn.Module):
    def __init__(self, in_channels = 1, 
                        kernel_size = 3, 
                        hidden_channels = 32, 
                        out_channels = 64, 
                        output_size = 10, 
                        strike = 1):
        super(SimpleCNN, self).__init__()
        """ This is an NLP model based on spikie neurons:
            -  """
        # Initialize layers
        spike_grad = surrogate.fast_sigmoid(slope = 25)
        # self.cl1 = nn.Linear(input_size, hidden_size)
        self.cl1 = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = kernel_size, padding = strike)
        self.lif1 = snn.Leaky(beta=0.99, # Decay factor Beta
                            threshold=1, # U_{thr} threshold to generate a spike
                            learn_beta=False, # We don't want to learn the decay factor, just a hyperparameter
                            reset_mechanism='subtract', # We will substract the threshold voltage when a spike is generated (-R*U_{thr})
                            spike_grad=spike_grad)  # We will use a fast sigmoid as a surrogate gradient
        self.max_pool1 = nn.MaxPool2d(kernel_size = kernel_size)
        self.cl2 = nn.Conv2d(in_channels = hidden_channels, out_channels = out_channels, kernel_size = kernel_size, padding = strike)
        self.lif2 = snn.Leaky(beta=0.99, threshold=1, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad,
                              output=True) # We want to get the output of the LIF neuron which is the membrane potential 
        self.max_pool2 = nn.MaxPool2d(kernel_size = kernel_size)

        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(out_channels*3**2, output_size)
        self.lif3 = snn.Leaky(beta=0.99, threshold=1, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad,
                              output=True) # We want to get the output of the LIF neuron which is the membrane potential 
        

    def forward_cnn(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        x = self.cl1(x)
        x, mem1 = self.lif1(x, mem1)
        x = self.max_pool1(x)
        x = self.cl2(x)
        x, mem2 = self.lif2(x, mem2)
        x = self.max_pool2(x)
        x = self.flat(x)
        x = self.fc3(x)
        x, mem3 = self.lif3(x, mem3)

        return x, mem3

    def forward(self, data, num_steps=25):
        mem_rec = []
        spk_rec = []
        for step in range(num_steps):
            spk, mem = self.forward_cnn(data[step])
            mem_rec.append(mem)
            spk_rec.append(spk)
        return torch.stack(spk_rec), torch.stack(mem_rec) 
        # return spk_rec, mem_rec