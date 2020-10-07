For batching I’ve used pad_sequence(from torch.nn.utils.rnn) and
DataLoader(from torch.utils.data) to pad and batch the padded data after sending the training data in
the process sequence. For ‘PAD’ I’ve used the padding value as Zero.
For Irish data set (training and test) by using SGD optimizer, with a batch size of 32. I’ve gotten an
accuracy of 43%. One of the reasons for dropping accuracy is that the training data is getting padded. As
the neural network does not understand that the pads are insignificant.
But with Adam optimizer and a batch size of 32, I’ve gotten an accuracy of 85.11%.
Yes, the training speed does increase significantly when used batching.
The model is saved at model_save_Batch.pth.
References: https://discuss.pytorch.org/t/facing-an-issue-while-using-batching-in-lstm/95263
