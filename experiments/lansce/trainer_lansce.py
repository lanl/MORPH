# Trainer class
import numpy as np
import torch
import torch.nn as nn
import experiments.lansce.dataloading_lansce as dataloading
class Trainer:
    @staticmethod
    def to_log_space(x):
        x = x.float()
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def to_raw_space(x_log):
        # Model outputs can be slightly negative even though true log1p targets are >= 0.
        # Clamp after expm1 so raw density/counts are nonnegative.
        x_raw = torch.expm1(x_log)
        x_raw = torch.clamp(x_raw, min=0.0)
        return x_raw
    
    @staticmethod
    def train_epoch(dataloader_train, model, optimizer, device, log_scale = True):
        model.train()
        train_loss = []

        criterion = nn.MSELoss()

        for step, batch in enumerate(dataloader_train):
            x_tr, y_tr = batch
            x_tr = x_tr.to(device)
            y_tr = y_tr.to(device)

            if log_scale:
                # raw -> log
                x_tr_log = Trainer.to_log_space(x_tr)
                y_tr_log = Trainer.to_log_space(y_tr)
            else:
                x_tr_log = x_tr
                y_tr_log = y_tr

            optimizer.zero_grad()

            _, _, x_nsp_log = model(x_tr_log)

            # loss in log-space
            loss = criterion(x_nsp_log, y_tr_log)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        return np.mean(train_loss)
    
    @staticmethod
    def test_epoch(dataloader_val, model, device, log_scale = True):
        model.eval()

        log_mse = []

        criterion = nn.MSELoss()

        with torch.no_grad():
            for step, batch in enumerate(dataloader_val):
                x_val, y_val_raw = batch
                x_val = x_val.to(device)
                y_val_raw = y_val_raw.to(device)

                if log_scale:
                    x_val_log = Trainer.to_log_space(x_val)
                    y_val_log = Trainer.to_log_space(y_val_raw)
                else:
                    x_val_log = x_val
                    y_val_log = y_val_raw

                _, _, x_nsp_log = model(x_val_log)

                # log-space metric
                loss_log = criterion(x_nsp_log, y_val_log)

                # raw-space metric
                x_nsp_raw = Trainer.to_raw_space(x_nsp_log)
                loss_raw = criterion(x_nsp_raw, y_val_raw.float())

                log_mse.append(loss_log.item())

        return np.mean(log_mse)
    
    # Testing
    @staticmethod
    def testing_nsp(inputs, targets, model, device):
        inputs = torch.from_numpy(inputs).to(device)
        targets = torch.from_numpy(targets).to(device)

        model.eval()

        mse_raw = []

        criterion = nn.MSELoss()

        with torch.no_grad():
            for snaps in range(inputs.shape[0]):
                x_raw = inputs[snaps]       # shape: (F, H, W)
                y_raw = targets[snaps]      # shape: (F, H, W)

                x_log = Trainer.to_log_space(x_raw)
                y_log = Trainer.to_log_space(y_raw)

                x_input = x_log[None, None, :, None, None]
                y_target_raw = y_raw.float()[None, :, None, None]

                _, _, pred_log = model(x_input)

                pred_raw = Trainer.to_raw_space(pred_log)
                loss_raw = criterion(pred_raw, y_target_raw)
                mse_raw.append(loss_raw.item())

        return mse_raw
        
    # Testing
    @staticmethod
    def testing_ro(psp_input_np, model, horizon=47, device='cpu'):
        cur_raw = torch.from_numpy(psp_input_np).to(device)

        # Save initial frame in raw space
        init_frame = cur_raw.detach().cpu().numpy().copy()

        # Model input is log-space
        cur_log = Trainer.to_log_space(cur_raw)

        model.eval()
        pred_rollouts_raw = []

        with torch.no_grad():
            for _ in range(horizon):
                _, _, pred_log = model(cur_log)

                # Convert prediction to raw space for saving / plotting / metrics
                pred_raw = Trainer.to_raw_space(pred_log)

                pred_rollouts_raw.append(
                    pred_raw.unsqueeze(1).detach().cpu().numpy()
                )

                # Feed next step back in log-space.
                # Clamp to keep recursive input consistent with log1p(raw) >= 0.
                pred_log_for_next = torch.clamp(pred_log, min=0.0)
                cur_log = pred_log_for_next.unsqueeze(1)

        pred_rollouts_raw = np.concatenate(pred_rollouts_raw, axis=1)
        pred_full_traj = np.concatenate([init_frame, pred_rollouts_raw], axis=1)

        pred_full_traj = pred_full_traj[0, :, :, 0, 0, :, :]

        return pred_full_traj