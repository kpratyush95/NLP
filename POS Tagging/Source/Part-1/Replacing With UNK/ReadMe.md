For this I’ve changed the prepare_sequence function to handle the words which are not in the training
data and replace them ‘UNK’ which are not in training data. By training the model on the Irish(Testing
and Training files). For 10 epochs, I’ve got an accuracy of 79.5 %.
The model is saved in model_save.pth file.
The second part where we are supposed to be substitute the training_data with “UNK” I have created 2
functions
1) Substitute_with_UNK() // Parameters: data and n. The data is training data and the value of n is
used to weed out the words which are less than n.
2) Rare_words_to_UNK() // replaces the less frequent words with ‘UNK’ in the training data.
By training the model to replace the less frequent words with ‘UNK’ on Irish(Testing and Training files),
For 10 epochs, I’ve got an accuracy of 81.5%.
The model is saved in model_save_UNK.pth.
References: https://stackoverflow.com/questions/36656870/replace-rare-word-tokens-python
