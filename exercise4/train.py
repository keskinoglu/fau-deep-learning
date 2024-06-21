import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model as m
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = None

#Hyperparameters
BATCH_SIZE = 96
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPOCHS = 150
EARLY_STOPPING = 30
L2_REG = 1e-4
EPS = 1e-8 # 1.0 or 0.1 via TensorFlow documentation, 1e-8 default


try:
    import os
    PATH = os.path.dirname(os.path.abspath(__file__))
    #print("Path set to: " + PATH)
except:
    print("Could not determine path. Please set it manually if needed.")
    pass

def main():
    # load the data from the csv file and perform a train-test-split
    # this can be accomplished using the already imported pandas and sklearn.model_selection modules
    # TODO
    data = pd.read_csv(PATH + '/data.csv', sep=';')
    #data = pd.read_csv('data.csv')
    train_data, val_data = train_test_split(data, test_size=0.1)

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    # TODO
    ### GPU OPTIMIZATIONS ###
    train_loader = t.utils.data.DataLoader(ChallengeDataset(train_data, mode="train"), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = t.utils.data.DataLoader(ChallengeDataset(val_data, mode="val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    ### NO GPU OPTIMIZATIONS ###
    #train_loader = t.utils.data.DataLoader(ChallengeDataset(train_data, mode="train"), batch_size=32, shuffle=True)
    #val_loader = t.utils.data.DataLoader(ChallengeDataset(val_data, mode="val"), batch_size=32, shuffle=False)

    # create an instance of our ResNet model
    # TODO
    model = m.ResNet()

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    # set up the optimizer (see t.optim)
    # create an object of type Trainer and set its early stopping criterion
    # TODO
    #pos_weight = t.tensor([3.5, 15.4]).cuda() # !!!!!!!!!!! assume CUDA for now !!!!!!!!!!!!!
    #criterion = t.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #weight=t.tensor([1.0, 3.6]).cuda()
    criterion = t.nn.BCELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG, eps=EPS, betas=BETAS)
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, early_stopping_patience=EARLY_STOPPING)

    # go, go, go... call fit on trainer
    res = trainer.fit(epochs=EPOCHS) #TODO

    # plot the results
    plt.clf()
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    # plot the first element in each tuple in res[1]
    plt.plot(np.arange(len(res[1])), [x[0] for x in res[1]], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(PATH + '/losses.png')

    plt.clf()
    plt.plot(np.arange(len(res[1])), [x[1] for x in res[1]], label='f1')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(PATH + '/f1 score.png')


if __name__ == '__main__':
    import time
    start = time.time()
    print("Starting training...")
    main()
    print("Training took: " + str(int((time.time() - start) / 60)) + " minutes and " + str(int((time.time() - start) % 60)) + " seconds.")