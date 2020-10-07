 I have applied Early stopping criteria on Irish (Training, Testing and
development) data set. Used Patience as 8 (I,e.) watch the model till 8 epochs if the validation loss does
not improve for 8 consecutive epochs. Restore the previous model.
For Irish dataset without replacing with UNK. I’ve gotten an accuracy of of 80.02 %. The model has
reached an Early stopping at Epoch 21 (Restoring the model which is saved at Epoch 13).
Model is saved at model_save_Early_stop.



Early stopping with UNK: After replacing the less frequent words as UNK in the training set and
applying Early stopping criteria on Irish(Training, Testing and development) data set.
For Irish dataset with replacing with UNK. I’ve gotten an accuracy of 83.36%. The model has reached an
Early stopping at Epoch 28 (Restoring the model which is saved at Epoch 20).
The model is saved at model_save_Early_stop_UNK.pth.
