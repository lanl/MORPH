# Trainer class
import numpy as np
import torch
import torch.nn as nn

class Trainer:
    @staticmethod
    def masked_sst_loss(pred, target):
        # pred: (B, F, C, D, H, W) and target: (B, F, C, D, H, W)
        pred_sst = pred[:, 0]
        pred_mask = pred[:, 1]

        true_sst = target[:, 0]
        ocean_mask = target[:, 1]

        ocean_mask = ocean_mask.float()

        # Main loss: SST only, ocean/padding masked
        sst_loss = ((pred_sst - true_sst) ** 2 * ocean_mask).sum()
        sst_loss = sst_loss / (ocean_mask.sum() + 1e-8)

        return sst_loss
    
    @staticmethod
    def relative_sst_L2_error(pred, target):
        """
        pred, target: (B, 2, H, W)

        channel 0 = SST
        channel 1 = mask
        """

        pred_sst = pred[:, 0]
        true_sst = target[:, 0]
        ocean_mask = target[:, 1].float()

        diff = (pred_sst - true_sst) * ocean_mask
        ref = true_sst * ocean_mask

        numerator = torch.sqrt(torch.sum(diff ** 2))
        denominator = torch.sqrt(torch.sum(ref ** 2)) + 1e-8

        return numerator / denominator
    
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
            loss = Trainer.masked_sst_loss(x_nsp, y_tr)
            #print(f"[Trainer] Step {step}, Loss: {loss.item():.6f}, {loss.dtype}")

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
                
                loss = Trainer.masked_sst_loss(x_nsp, y_val)

                test_loss.append(loss.item())

        return np.mean(test_loss)
    
    @staticmethod
    def test_epoch_nsp(dataloader_val, model, sst_max, sst_min,device):
        scale = sst_max - sst_min
        model.eval() # Set the eval mode for model
        test_loss_list, relative_loss_list = [], []
        input_frames, target_frames, pred_frames = [], [], []
        with torch.no_grad(): 
            for step, batch in enumerate(dataloader_val):
                x_val, y_val = batch
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Model 1 forward
                _, _, x_nsp = model(x_val)

                # un-normalize only SST/value channels
                x_val[:, :, 0] = x_val[:, :, 0] * scale + sst_min
                y_val[:, 0] = y_val[:, 0] * scale + sst_min
                x_nsp[:, 0] = x_nsp[:, 0] * scale + sst_min
                
                loss = Trainer.masked_sst_loss(x_nsp, y_val)
                relative_loss = Trainer.relative_sst_L2_error(x_nsp, y_val)

                test_loss_list.append(loss.item())
                relative_loss_list.append(relative_loss.item())
                input_frames.append(x_val.cpu().numpy())
                target_frames.append(y_val.cpu().numpy())
                pred_frames.append(x_nsp.cpu().numpy())

        return (np.concatenate(input_frames, axis=0), 
                np.concatenate(target_frames, axis=0), 
                np.concatenate(pred_frames, axis=0), 
                test_loss_list, relative_loss_list)