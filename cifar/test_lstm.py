# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


torch.manual_seed(1)

  # Input dim is 3, output dim is 3
# inputs = [autograd.Variable(torch.randn((1, 3)))
#           for _ in range(5)]  # make a sequence of length 5
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(len(sentence), 4, -1), self.hidden)
        # print("hidden", self.hidden)
        # tensor.view == reshape(tensor)
        # print("lstm out", lstm_out.shape)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


ngradients = np.array([[0.1,0.2,0.1,0.2],[0.1,0.2,0.1,0.2]])
nloss = np.array([[1.,0.8,0.5,0.2],[0.4,0.2,0.1,0.05]])
nlr = np.array([[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]])
bestlr = np.array([[2,2,3,3],[1,1,2,3]])

model = LSTMTagger(3,6,5)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(3000):
    if epoch %1000 == 0:
        print("####### epoch : %d #######"%(epoch))
    for i in range(2):
        sentence = np.zeros((4, 3))
        sentence[:, 0] = ngradients[i]
        sentence[:, 1] = nloss[i]
        sentence[:, 2] = nlr[i]

        v_sentence = autograd.Variable(torch.FloatTensor(sentence))
        v_targets = autograd.Variable(torch.LongTensor(bestlr[i]))
        # print(targets)

        model.zero_grad()
        model.hidden = model.init_hidden()  # lstm initialize hidden layer
        logits = model(v_sentence)
        # print(logits.shape)
        # print(v_targets.shape)
        loss = loss_function(logits, v_targets)
        loss.backward()
        optimizer.step()
        # sentence


## predict
for i in range(2):
    sentence = np.zeros((4, 3))
    sentence[:, 0] = ngradients[i]
    sentence[:, 1] = nloss[i]
    sentence[:, 2] = nlr[i]

    v_sentence = autograd.Variable(torch.FloatTensor(sentence))
    v_targets = autograd.Variable(torch.FloatTensor(bestlr[i]))
    # print(targets)

    logits = model(v_sentence)
    print("$resuts$ : ", logits)


# initialize the hidden state.
# hidden = (autograd.Variable(torch.randn(1, 1, 3)),
#           autograd.Variable(torch.randn((1, 1, 3))))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(
#     torch.randn((1, 1, 3))))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)

