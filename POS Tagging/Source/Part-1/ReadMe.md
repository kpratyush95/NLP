For running the code I’ve used google collab. I am providing the .ipnb files. For this you need to upload
the irish.test, irish.dev, irish.train for every .ipnb notebook. Further I’ve added the .py file for each
module as well. Each module is placed in a separate file.
1.a [1 credit] Model Read/Write:
For this part I’ve used inbuilt torch.save() function. The save function takes 2 arguments
1) model_state which contains the state and optimizer
2) Path: where to save the file.
To load the saved model, I’ve used torch.load() function which takes in 1 argument, path, to retrieve
saved model.
The model is stored in .pth file. Here I’ve saved my model in model_save.pth.
