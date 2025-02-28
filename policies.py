import numba
import torch
import torch.nn as nn
import numpy as np
from numba.experimental import jitclass
from numba.typed import List

class MLP_heb(nn.Module):
    "MLP, no bias"

    def __init__(self, input_space, action_space):
        super(MLP_heb, self).__init__()

        self.fc1 = nn.Linear(input_space, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, action_space, bias=False)

    def forward(self, ob):
        state = torch.as_tensor(ob[0]).float().detach()

        x1 = torch.tanh(self.fc1(state))
        x2 = torch.tanh(self.fc2(x1))
        o = self.fc3(x2)

        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  


class CNN_heb(nn.Module):
    "CNN+MLP with n=input_channels frames as input. Non-activated last layer's output"

    def __init__(self, input_channels, action_space_dim):
        super(CNN_heb, self).__init__()
        # print("output    ", action_space_dim)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)

        self.linear1 = nn.Linear(648, 128, bias=False)
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.out = nn.Linear(64, action_space_dim, bias=False)

    def forward(self, ob):
        state = torch.as_tensor(ob.copy())
        state = state.float()

        x1 = self.pool(torch.tanh(self.conv1(state)))
        x2 = self.pool(torch.tanh(self.conv2(x1)))

        x3 = x2.view(-1)

        x4 = torch.tanh(self.linear1(x3))
        x5 = torch.tanh(self.linear2(x4))

        o = self.out(x5)

        return x3, x4, x5, o
        # return self.pool(self.conv2(x1)).view(-1), self.linear1(x3), self.linear2(x4), o


class NN():
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.activations = [[0 for i in range(node)] for node in nodes]
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1]+nodes[1]*nodes[2]+nodes[2]*nodes[3]

        self.weights = [[] for _ in range(len(self.nodes) - 1)]

    def activate(self, inputs):
        self.activations[0] = [np.tanh(x) for x in inputs]
        for i in range(1, len(self.nodes)):
            self.activations[i] = [0. for _ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                sum = 0  # self.weights[i - 1][j][0]
                for k in range(self.nodes[i - 1]):
                    sum += self.activations[i - 1][k - 1] * self.weights[i - 1][j][k]
                self.activations[i][j] = np.tanh(sum)
        return np.array(self.activations[-1])

    def set_weights(self, weights):
        # self.weights = [[] for _ in range(len(self.nodes) - 1)]
        c = 0
        for i in range(1, len(self.nodes)):
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    self.weights[i - 1][j][k] = weights[c]
                    c += 1
        # print(c)

    def get_list_weights(self):
        wghts = []
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    wghts.append(np.abs(self.weights[i - 1][j][k]))
        return wghts


#  )

tmp = List()
tmp.append(0)

@jitclass(
    [('nodes', numba.typeof(tmp)), ("nnodes", numba.typeof(0)), ("activations", numba.typeof(np.zeros((2,1) ,dtype=float))),
     ('hrules',numba.typeof( np.zeros((2,1) ,dtype=float))), ('nparams', numba.typeof(0))])
class WLNHNN():
    def __init__(self, nodes):
        self.nodes = nodes
        self.nnodes = self.sa(self.nodes)
        self.activations = np.zeros((self.nnodes, 1))
        self.hrules = np.zeros((self.nnodes, 7))
        self.nparams = (self.nnodes * 5) - nodes[0] - nodes[-1]

    def sa(self, arr):
        s = 0
        for e in arr:
            s += e
        return s

    def __call__(self, inputs):
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor(self.forward(inputs[0].tolist()))

    def call(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        for i in range(len(inputs)):
            self.activations[i] = np.tanh(inputs[i])
        for i in range(len(inputs)):
            self.hrules[i,5] = self.hrules[i,6]
            self.hrules[i,6] = self.activations[i,0]
        offset = self.nodes[0]
        for l in range(1, len(self.nodes)):

            for o in range(self.nodes[l]):
                act = 0  # self.weights[i - 1][j][0]
                for i in range(self.nodes[l - 1]):
                    dw0 = self.cdw(l, i, o, 5)
                    dw1 = self.cdw(l, i, o, 6)
                    act += (dw0 + dw1) * self.activations[self.sa(self.nodes[:l - 1]) + i,0]
                # print(l,i,o,offset)
                self.activations[offset + o] = np.tanh(act)
                self.hrules[offset + o,5] = self.hrules[offset + o,6]
                self.hrules[offset + o,6] = self.activations[offset + o,0]
            offset += self.nodes[l]

        return self.activations[self.sa(self.nodes) - self.nodes[-1]:]

    def cdw(self, l, i, o, t):
        offsetI = self.sa(self.nodes[:l - 1])
        offsetO = self.sa(self.nodes[:l])

        dw = (
                self.hrules[offsetI + i,2] * self.hrules[offsetO + o,2] * self.hrules[offsetI + i,t] *
                self.hrules[offsetO + o,t] +  # both
                self.hrules[offsetO + o,1] * self.hrules[offsetO + o,t] +  # post
                self.hrules[offsetI + i,0] * self.hrules[offsetI + i,t] +  # pre
                self.hrules[offsetO + o,3] * self.hrules[offsetI + i,3])
        eta = 0.5 * (self.hrules[offsetO + o,4] + self.hrules[offsetI + i,4])
        return eta * dw


    def set_hrules(self, hrules):
        c = 0
        start = 0
        for layer in range(len(self.nodes)):
            for node in range(self.nodes[layer]):
                if layer == 0:  # input
                    self.hrules[start + node][0] = hrules[c]
                    self.hrules[start + node][1] = 0  # hrules[c + 1]
                    self.hrules[start + node][2] = hrules[c + 1]  # hrules[c + 1]
                    self.hrules[start + node][3] = hrules[c + 2]  # hrules[c + 1]
                    self.hrules[start + node][4] = hrules[c + 3]
                    c += 4

                elif layer == (len(self.nodes) - 1):  # output
                    self.hrules[start + node][0] = 0
                    self.hrules[start + node][1] = hrules[c]  # hrules[c + 1]
                    self.hrules[start + node][2] = hrules[c + 1]  # hrules[c + 1]
                    self.hrules[start + node][3] = hrules[c + 2]  # hrules[c + 1]
                    self.hrules[start + node][4] = hrules[c + 3]
                    c += 4

                else:
                    self.hrules[start + node][0] = hrules[c]
                    self.hrules[start + node][1] = hrules[c + 1]  # hrules[c + 1]
                    self.hrules[start + node][2] = hrules[c + 2]  # hrules[c + 1]
                    self.hrules[start + node][3] = hrules[c + 3]  # hrules[c + 1]
                    self.hrules[start + node][4] = hrules[c + 4]
                    c += 5
            start += self.nodes[layer]

        # print(self.hrules)

if __name__ =='__main__':
    nodes = List()
    nodes.append(1)
    nodes.append(1)
    nodes.append(1)
    fka = WLNHNN(nodes)