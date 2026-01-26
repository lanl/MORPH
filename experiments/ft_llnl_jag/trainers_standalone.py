import torch
import numpy as np

class Trainer2:
    @staticmethod
    def train_epoch_ss(dataloader_train, model, optimizer, loss_fn, device):
        model.train() 
        train_loss = []
        for step, batch in enumerate(dataloader_train):
            x_tr, y_tr, s_tr = batch
            x_tr, y_tr, s_tr = x_tr.to(device), y_tr.to(device), s_tr.to(device)
            # print(f'X: {x_tr.shape}, y: {y_tr.shape}')
            # reshape x_tr
            x_tr = x_tr[:, 0, :, :, 0, :, :].reshape(-1, 4, 64, 64)  # (B,T,F,C,D,H,W) -> (B, 4, 64, 64)
            # print(f'X: {x_tr.shape}, y: {y_tr.shape}')

            optimizer.zero_grad()

            # Model 1 forward + loss
            y_hat = model(x_tr, s_tr)
            loss = loss_fn(y_hat, y_tr) 

            # Model 1 backward
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        return np.mean(train_loss)

    @staticmethod
    def test_epoch_ss(dataloader_val, model, loss_fn, device):
        model.eval() # Set the eval mode for model
        test_loss = []
        with torch.no_grad(): # No need to track the gradients
            for step, batch in enumerate(dataloader_val):
                x_val, y_val, s_val = batch
                x_val, y_val, s_val = x_val.to(device), y_val.to(device), s_val.to(device)

                # print(f'X: {x_tr.shape}, y: {y_tr.shape}')
                # reshape x_tr
                x_val = x_val[:, 0, :, :, 0, :, :].reshape(-1, 4, 64, 64)  # (B,T,F,C,D,H,W) -> (B, 4, 64, 64)
                # print(f'X: {x_val.shape}, y: {y_val.shape}')

                # Modelforward
                y_hat = model(x_val, s_val)
                loss = loss_fn(y_hat, y_val)
                test_loss.append(loss.item())

        return np.mean(test_loss)