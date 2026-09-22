# Trainer class
import numpy as np
import torch
import torch.nn as nn
class Trainer:
    @staticmethod
    def train_epoch(dataloader_train, model_1, model_2, optimizer_1, 
                    optimizer_2, loss_fn, device):
        model_1.train() 
        model_2.train()
        train_loss_1 = []
        train_loss_2 = []
        for step, batch in enumerate(dataloader_train):
            x_tr, y_tr, s_tr = batch
            x_tr, y_tr, s_tr = x_tr.to(device), y_tr.to(device), s_tr.to(device)
            #print(f'X: {x_tr.shape}, y: {y_tr.shape}')
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # Model 1 forward + loss
            _, z, x_nsp = model_1(x_tr)
            #print(f"[Trainer] Model 1 output shape: {x_nsp.shape}")
            #print(f"[Trainer] Model 1 latent shape: {z.shape}")
            loss_1 = loss_fn(x_nsp.unsqueeze(1), x_tr) # autoencoder loss

            # Model 1 backward
            loss_1.backward()
            optimizer_1.step()
            train_loss_1.append(loss_1.item())

            # Model 2 forward + backward
            z = z.detach() # detach the two graphs
            y_hat = model_2(z.squeeze(1), s_tr)
            #print(f"[Trainer] Model 2 output shape: {y_hat.shape}")
            loss_2 = loss_fn(y_hat, y_tr)
            loss_2.backward()
            optimizer_2.step()
            train_loss_2.append(loss_2.item())

        return np.mean(train_loss_1), np.mean(train_loss_2)

    @staticmethod
    def test_epoch(dataloader_val, model_1, model_2, loss_fn, device):
        model_1.eval() # Set the eval mode for model
        model_2.eval()
        test_loss_1 = []
        test_loss_2 = []
        with torch.no_grad(): # No need to track the gradients
            for step, batch in enumerate(dataloader_val):
                x_val, y_val, s_val = batch
                x_val, y_val, s_val = x_val.to(device), y_val.to(device), s_val.to(device)

                # Model 1 forward
                _, z, x_nsp = model_1(x_val)
                loss_1 = loss_fn(x_nsp.unsqueeze(1), x_val)
                test_loss_1.append(loss_1.item())
                
                # Model 2 forward
                y_hat_2 = model_2(z.squeeze(1), s_val)
                loss_2 = loss_fn(y_hat_2, y_val)
                test_loss_2.append(loss_2.item())

        return np.mean(test_loss_1), np.mean(test_loss_2)
    
    # Testing
    @staticmethod
    def testing(dataloader_test, model_1, model_2, loss_fn, device):
            model_1.eval() # Set the eval mode for model
            model_2.eval()
            mse_loss_1, mse_loss_2 = [], []
            x_org, x_pred, y_org, y_pred = [], [], [], []
            with torch.no_grad(): # No need to track the gradients
                for _, batch in enumerate(dataloader_test):
                    x_test, y_test, s_test = batch
                    x_test, y_test, s_test = x_test.to(device), y_test.to(device), s_test.to(device)

                    # Model 1 forward
                    _, z, x_nsp = model_1(x_test)
                    loss_1 = loss_fn(x_nsp.unsqueeze(1), x_test)
                    mse_loss_1.append(loss_1.item())
                    
                    # Model 2 forward
                    y_hat_2 = model_2(z.squeeze(1), s_test)
                    loss_2 = loss_fn(y_hat_2, y_test)
                    mse_loss_2.append(loss_2.item())

                    # Collect original and predicted samples
                    x_org.append(x_test.cpu().numpy())
                    x_pred.append(x_nsp.unsqueeze(1).cpu().numpy())
                    y_org.append(y_test.cpu().numpy())
                    y_pred.append(y_hat_2.cpu().numpy())

            return mse_loss_1, mse_loss_2, x_org, x_pred, y_org, y_pred