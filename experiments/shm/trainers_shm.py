import numpy as np
import torch
import torch.nn as nn

class Trainer:
    @staticmethod
    def train_epoch(dataloader_train, model_1, model_2, optimizer_1, optimizer_2, device):
        model_1.train()
        model_2.train()

        mse_criterion = nn.MSELoss()
        bce_criterion = nn.BCEWithLogitsLoss()

        train_loss_1 = []
        train_loss_2 = []
        train_correct = 0
        train_total = 0

        for x_tr, y_tr in dataloader_train:
            x_tr = x_tr.to(device)
            y_tr = y_tr.to(device).float().view(-1, 1)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # forward
            _, z, x_nsp = model_1(x_tr)

            # reconstruction loss
            loss_1 = mse_criterion(x_nsp.unsqueeze(1), x_tr)

            # classifier loss on detached latent
            logits = model_2(z.detach().squeeze(1))
            loss_2 = bce_criterion(logits, y_tr)

            # backward
            loss_1.backward()
            loss_2.backward()

            optimizer_1.step()
            optimizer_2.step()

            train_loss_1.append(loss_1.item())
            train_loss_2.append(loss_2.item())

            predicted = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (predicted == y_tr).sum().item()
            train_total += y_tr.numel()

        train_acc = train_correct / train_total

        return np.mean(train_loss_1), np.mean(train_loss_2), train_acc

    @staticmethod
    def test_epoch(dataloader_val, model_1, model_2, device):
        model_1.eval()
        model_2.eval()

        mse_criterion = nn.MSELoss()
        bce_criterion = nn.BCEWithLogitsLoss()

        val_loss_1 = []
        val_loss_2 = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_val, y_val in dataloader_val:
                x_val = x_val.to(device)
                y_val = y_val.to(device).float().view(-1, 1)

                _, z, x_nsp = model_1(x_val)

                loss_1 = mse_criterion(x_nsp.unsqueeze(1), x_val)
                logits = model_2(z.squeeze(1))
                loss_2 = bce_criterion(logits, y_val)

                val_loss_1.append(loss_1.item())
                val_loss_2.append(loss_2.item())

                predicted = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (predicted == y_val).sum().item()
                val_total += y_val.numel()

        val_acc = val_correct / val_total

        return np.mean(val_loss_1), np.mean(val_loss_2), val_acc

    @staticmethod
    def testing(dataloader_test, model_1, model_2, device):
        model_1.eval()
        model_2.eval()

        mse_criterion = nn.MSELoss()
        bce_criterion = nn.BCEWithLogitsLoss()

        loss_1_list, loss_2_list = [], []
        x_org, x_pred = [], []
        y_org, y_prob, y_pred = [], [], []

        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x_test, y_test in dataloader_test:
                x_test = x_test.to(device)
                y_test = y_test.to(device).float().view(-1, 1)

                _, z, x_nsp = model_1(x_test)

                loss_1 = mse_criterion(x_nsp.unsqueeze(1), x_test)
                logits = model_2(z.squeeze(1))
                loss_2 = bce_criterion(logits, y_test)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                loss_1_list.append(loss_1.item())
                loss_2_list.append(loss_2.item())

                x_org.append(x_test.cpu().numpy())
                x_pred.append(x_nsp.unsqueeze(1).cpu().numpy())
                y_org.append(y_test.cpu().numpy())
                y_prob.append(probs.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

                test_correct += (preds == y_test).sum().item()
                test_total += y_test.numel()

        test_acc = test_correct / test_total

        return loss_1_list, loss_2_list, x_org, x_pred, y_org, y_prob, y_pred, test_acc