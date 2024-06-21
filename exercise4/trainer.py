import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit                      # Loss criterion
        self._optim = optim                    # Optimizer
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        try:
            t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
            print("Checkpoint saved in: /checkpoints/checkpoint_{:03d}.ckp".format(epoch))
        except RuntimeError:
            import os
            PATH = os.path.dirname(os.path.abspath(__file__))
            t.save({'state_dict': self._model.state_dict()}, PATH + '/checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
            print("Checkpoint saved in: " + PATH + "/checkpoints/checkpoint_{:03d}.ckp".format(epoch))
        except:
            print("Could not save checkpoint. Please set the path manually if needed.")
            pass
    
    def restore_checkpoint(self, epoch_n):
        try:
            ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
            self._model.load_state_dict(ckp['state_dict'])
        except FileNotFoundError:
            import os
            PATH = os.path.dirname(os.path.abspath(__file__))
            ckp = t.load(PATH + '\\checkpoints\\checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
            self._model.load_state_dict(ckp['state_dict'])
        except:
            print("Could not restore checkpoint. Please set the path manually if needed.")
            pass
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        try:
            import os
            PATH = os.path.dirname(os.path.abspath(__file__))
            fn = PATH + "/" + fn
        except:
            pass
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              #opset_version=11,          # CHANGED FOR LOCAL DEV
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
        print("Model saved to {}".format(fn))

    def update_pos_weight_for_BCE_loss(self, y):
        # calculate the positive weight for the loss function
        pos_weight = t.empty(y.size()).cuda()

        #pos_crack = 0# sum(crack for crack, inactive in y)
        #pos_inactive = 0# sum(inactive for crack, inactive in y)

        #for crack, inactive in y:
        #    pos_crack += crack
        #    pos_inactive += inactive

        #neg_crack = len(y) - pos_crack
        #neg_inactive = len(y) - pos_inactive

        pos_crack_ratio = 3.514673 #neg_crack / pos_crack
        pos_inactive_ratio = 15.39344 #neg_inactive / pos_inactive

        for i in range(len(y)):
            crack_weight = 1
            inactive_weight = 1

            if y[i][0] == 1: # crack
                crack_weight = pos_crack_ratio
            if y[i][1] == 1: # inactive
                inactive_weight = pos_inactive_ratio

            pos_weight[i] = t.tensor([crack_weight, inactive_weight]).cuda()

        self._crit.weight = pos_weight

        return

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        if self._cuda:
            x = x.cuda()
            y = y.cuda()
        predictions = self._model(x)

        try:
            self.update_pos_weight_for_BCE_loss(y)
            # might not be working
            scaler = t.cuda.amp.GradScaler()
            with t.cuda.amp.autocast():
                loss = self._crit(predictions.float(), y.float())
            scaler.scale(loss).backward()
            scaler.step(self._optim)
            scaler.update()
        except:
            #print("No AMP support. Using normal training.")
            self.update_pos_weight_for_BCE_loss(y)
            loss = self._crit(predictions.float(), y.float()) # Loss criterion / loss function # weird bug with dtype Long / Float
            loss.backward()
            self._optim.step()
        return loss.item()

    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO

        if self._cuda:
            x = x.cuda()
            y = y.cuda()
        with t.no_grad():
            predictions = self._model(x)
            self.update_pos_weight_for_BCE_loss(y)
            loss = self._crit(predictions, y.to(dtype=t.float32))
        return loss.item(), predictions.detach().cpu()
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO

        self._model.train()
        avg_loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            avg_loss += self.train_step(x, y)
        avg_loss /= len(self._train_dl)
        return avg_loss

    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()
        avg_loss = 0
        avg_f1 = 0
        with t.no_grad():
            for x, y_true in self._val_test_dl:
                if self._cuda:
                    x, y_true = x.cuda(), y_true.cuda()
                loss, prediction = self.val_test_step(x, y_true)
                avg_loss += loss
                #avg_f1 += f1_score(y.cpu().numpy().argmax(axis=1), prediction.argmax(axis=1), average='macro')
                binary_prediction = (prediction > 0.5).to(t.int64)
                avg_f1 += f1_score(y_true.cpu(), binary_prediction.cpu(), average='micro') 
        avg_loss /= len(self._val_test_dl)
        avg_f1 /= len(self._val_test_dl)
        return avg_loss, avg_f1
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        train_losses = []
        val_losses = []
        epoch = 0
        best_val_loss = float("inf")
        best_f1 = 0
        best_epoch = 0
        early_stopping_counter = 0

        #while True:
      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
        #TODO

        while True:
            if epochs > 0 and epoch >= epochs:
                self.restore_checkpoint(best_epoch) # load the best model
                # save onnx with name of the best f1 score
                self.save_onnx("best_f1_{}_epoch_{}.onnx".format(best_f1, best_epoch))
                break

            self._model.train()
            train_losses.append(self.train_epoch())

            self._model.eval()
            val_loss = self.val_test()
            val_losses.append(val_loss)

            # save checkpoint if f1 improved
            if val_loss[1] > best_f1:
                best_epoch = epoch
                best_f1 = val_loss[1]
                self.save_checkpoint(epoch)
                early_stopping_counter = 0

            # save checkpoint if val improved
            if val_loss[0] < best_val_loss:
                best_val_loss = val_loss[0]
                #self.save_checkpoint(epoch)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if self._early_stopping_patience > 0 and early_stopping_counter >= self._early_stopping_patience:
                    self.restore_checkpoint(best_epoch) # load the best model
                    # save onnx with name of the best f1 score
                    self.save_onnx("best_f1_{}_epoch_{}.onnx".format(best_f1, best_epoch))
                    break

            epoch += 1

            print("Epoch: {}, Train Loss: {}, Val Loss: {}, Val F1: {}".format(epoch - 1, train_losses[-1], val_loss[0], val_loss[1]))

            ### EXPERIMENT
            ### Increase LR per epoch ###
            #if epoch % 10 == 0:
            #    for param_group in self._optim.param_groups:
            #        param_group['lr'] *= 5
            #    print("LR decreased to {}".format(param_group['lr']))

            ### Results; EXP 2 0.000125 is best
                        #EXP 1 0.00064 is best

        return train_losses, val_losses

