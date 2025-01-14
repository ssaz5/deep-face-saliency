#Based on solver.py from DL4CV class
import numpy as np
import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.is_cuda = torch.cuda.is_available()
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, reset_hist=True):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        #optim = self.optim(model.parameters(), **self.optim_args)
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        
        if reset_hist:
            self._reset_histories()
        
        
        iter_per_epoch = len(train_loader)

        if self.is_cuda:
             model.cuda()


        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        for epoch in range(num_epochs):
            # TRAINING
            count = 0
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs), Variable(targets)
                if self.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs, mu, logvar = model(inputs)
                if self.is_cuda:
                    loss = self.loss_func(outputs, targets.type(torch.cuda.FloatTensor), mu, logvar)
                else:
                    loss = self.loss_func(outputs, targets.type(torch.FloatTensor), mu, logvar)

                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss * 1000))
                count +=1
                if count % 5000 == 0:
                    model.save('tempsave_VAE_model_epoch_'+'.pth')
                    #VALIDATION
                    val_losses = []
                    val_scores = []
                    model.eval()
                    
                    for inputs, targets in val_loader:
                        inputs, targets = Variable(inputs), Variable(targets)
                        if self.is_cuda:
                            inputs, targets = inputs.cuda(), targets.cuda()

                        outputs, mu, logvar = model.forward(inputs)
                        loss = self.loss_func(outputs, targets.type(torch.cuda.FloatTensor), mu, logvar)
                        val_losses.append(loss.data.cpu().numpy())
#                         self.val_loss_history.append(loss.data.cpu().numpy())
                        #_, preds = torch.max(outputs, 1)
                        preds = outputs

                        scores = np.mean((preds.type(torch.cuda.FloatTensor) == targets.type(torch.cuda.FloatTensor)).data.cpu().numpy())
                        val_scores.append(scores)
        #                 count +=1
#                         if count == 100:
#                             break

                    model.train()
                    val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
                    self.val_acc_history.append(val_acc)
                    self.val_loss_history.append(val_loss)
                    if log_nth:
                        print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                           num_epochs,
                                                                           val_acc,
                                                                           val_loss * 1000))
            #print(outputs.size())
            #_, preds = torch.max(outputs, 1)
            preds = outputs
            #print(preds)
            # Only allow images/pixels with label >= 0 e.g. for segmentation

            if model.is_cuda:
                train_acc = np.mean((preds.type(torch.cuda.FloatTensor) == targets.type(torch.cuda.FloatTensor)).data.cpu().numpy())
            else:
                train_acc = np.mean((preds.type(torch.cuda.FloatTensor) == targets.type(torch.FloatTensor)).data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss * 1000))
            #VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            count = 0
            for inputs, targets in val_loader:
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs, mu, logvar = model.forward(inputs)
                loss = self.loss_func(outputs, targets.type(torch.cuda.FloatTensor), mu, logvar)
                val_losses.append(loss.data.cpu().numpy())
#                 self.val_loss_history.append(loss.data.cpu().numpy())
                #_, preds = torch.max(outputs, 1)
                preds = outputs

                scores = np.mean((preds.type(torch.cuda.FloatTensor) == targets.type(torch.cuda.FloatTensor)).data.cpu().numpy())
                val_scores.append(scores)
#                 count +=1
                if count == 100:
                    break

            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss * 1000))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
print('FINISH.')