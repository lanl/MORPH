# Trainer class
import numpy as np
import torch
import torch.nn as nn
import experiments.lansce.dataloading_lansce as dataloading
class Trainer:
    @staticmethod
    def train_epoch(dataloader_train, model, optimizer, device):
        model.train() 
        train_loss = []
        for step, batch in enumerate(dataloader_train):
            x_tr, y_tr = batch
            x_tr, y_tr = x_tr.to(device), y_tr.to(device)
            #print(f'X: {x_tr.shape}, y: {y_tr.shape}')
            optimizer.zero_grad()

            # Model 1 forward + loss
            _, _, x_nsp = model(x_tr)
            #print(f"[Trainer] Model output shape: {x_nsp.shape}")
            loss = nn.MSELoss()(x_nsp, y_tr) 

            # Model 1 backward
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        return np.mean(train_loss)
    
    @staticmethod
    def test_epoch(dataloader_val, model, device):
        model.eval() # Set the eval mode for model
        test_loss = []
        with torch.no_grad(): 
            for step, batch in enumerate(dataloader_val):
                x_val, y_val = batch
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Model 1 forward
                _, _, x_nsp = model(x_val)
                loss = nn.MSELoss()(x_nsp, y_val)
                test_loss.append(loss.item())

        return np.mean(test_loss)
    
    # Testing
    @staticmethod
    def testing_nsp(inputs, targets, model, device):
        inputs = torch.from_numpy(inputs).to(device)
        targets = torch.from_numpy(targets).to(device)

        model.eval()
        mse_loss = []

        with torch.no_grad():
            for snaps in range(inputs.shape[0]):
                x_tr = inputs[snaps].unsqueeze(0)   # shape: (1,1,2,1,1,5736)
                y_tr = targets[snaps].unsqueeze(0)  # shape: (1,2,1,1,5736)
                #print(f"[testing_nsp] Input shape: {x_tr.shape}, Target shape: {y_tr.shape}")

                _, _, x_nsp = model(x_tr)      # shape: (1,2,1,1,5736)
                #print(f"[testing_nsp] Model output shape: {x_nsp.shape}")
                loss = nn.MSELoss()(x_nsp, y_tr)
                mse_loss.append(loss.item())

        return mse_loss
    
    # Testing
    @staticmethod
    def testing_ro(psp_input_np, model, horizon=47, device='cpu'): #input frame: (F=C,P)
        input_frame_uptf7 = psp_input_np[None, None, :, None, None, None, :]  # (1, 1, F=C, 1, 1, 1, P)
        cur = torch.from_numpy(input_frame_uptf7).to(device)
        init_frame = cur.detach().cpu().numpy().copy()
        model.eval()
        pred_rollouts = []

        with torch.no_grad():
            for _ in range(horizon):
                _, _, dam_nsp = model(cur)      # (B=1, F=C, 1, 1, H, W)
                cur = dam_nsp.unsqueeze(1)      # (B=1, T=1, F=C, 1, 1, H, W)
                pred_rollouts.append(cur.detach().cpu().numpy())

        pred_rollouts = np.concatenate(pred_rollouts, axis=1)
        pred_full_traj_uptf7 = np.concatenate([init_frame, pred_rollouts], axis=1) # (1, T=401, F=C, 1, 1, 1, P)
        pred_full_traj = pred_full_traj_uptf7[0, :, :, 0, 0, 0, :]  # (T, F, P)

        return pred_full_traj # Return shape: (T, F, P)