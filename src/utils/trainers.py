import torch
import random
import time
from src.utils.main_process_ddp import is_main_process
import torch.cuda.amp as amp

class Trainer:
    @staticmethod
    def train_singlestep(model, train_loader, criterion, optimizer, device, 
                         epoch, scheduler, model_path, save_batch_ckpt, save_batch_freq):
        model.train()
        running_loss, window_loss = 0.0, 0.0
        window_size = 250
        save_batch_freq = 1000
        n_batches   = 0
        start_time  = time.time()
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking = True)
            targets = targets.to(device, non_blocking = True)
            
            # abort if your data already has NaNs or Infs
            # if torch.isnan(images).any() or torch.isinf(images).any():
            #     raise RuntimeError("NaN/Inf in input images")
            # if torch.isnan(targets).any() or torch.isinf(targets).any():
            #     raise RuntimeError("NaN/Inf in targets")
    
            # Zero grads once per iteration (before forward)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # collect the losses
            n_batches   += 1
            window_loss += loss.item()
            running_loss += loss.item()
            
            # print only few batch
            if is_main_process():
                if n_batches < 21:
                    elapsed_0 = (time.time() - start_time) / 60
                    print(f'[Batch no. {n_batches}, Cumtime={elapsed_0:.3f}min, Images(N,F,C,D,H,W):' 
                          f'{images.shape}, Targets(N,F,C,D,H,W): {targets.shape}'
                          f' Batch loss: {loss.item():.5f}')
            
                # if torch.isnan(loss) or torch.isinf(loss):
                #     raise RuntimeError("NaN/Inf in loss")
            
                if n_batches % window_size == 0:
                    elapsed = (time.time() - start_time) / 60
                    avg_all = running_loss / n_batches
                    avg_win = window_loss  / window_size
                    window_loss = 0.0
        
                    print(f"Cumtime={elapsed:.2f} min, Batches={n_batches} | "
                          f"Overall Avg Loss={avg_all:.5f} | "
                         f"Window Avg Loss={avg_win:.5f}")
                
            # -- Optional batch checkpoint every B batches (don't save when no improvement) ---
            if save_batch_ckpt and (n_batches % save_batch_freq) == 0 and is_main_process():
                checkpoint = {"epoch": epoch, # save with last epoch name
                              "model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "scheduler_state_dict": scheduler.state_dict()}
                ckpt_path = f"{model_path}_ep{epoch}_latestbatch.pth"
                torch.save(checkpoint, ckpt_path)
                print(f" Saved (overwrite) new batch checkpoint (old epoch name): {ckpt_path}")
        
        return running_loss / n_batches
    
    @staticmethod
    def train_singlestep_accumulate(model, train_loader, criterion, optimizer, device, accum_steps=16):
        '''
        Use this trainer for a batch_size of 1. Similar to MPP paper
        '''
        model.train()
        running_loss, window_loss = 0.0, 0.0
        window_size = 500
        n_batches   = 0
        start_time  = time.time()
        
        # zero out grads before we start accumulating
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking = True)
            targets = targets.to(device, non_blocking = True)
            outputs = model(images)
            
            # scale the loss so its gradient is equivalent to a batch of size=accum_steps
            loss = criterion(outputs, targets) / accum_steps  
            loss.backward()

            # every accum_steps micro-batches, do an optimizer step
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
             
            # collect the losses
            n_batches   += 1
            window_loss += loss.item()
            running_loss += loss.item()
            
            # print only few batch
            if is_main_process():
                if n_batches < 40:
                    elapsed_1 = (time.time() - start_time) / 60
                    print(f'[Batch no. {n_batches}, Time={elapsed_1:.2f}, Images(N,T,F,C,D,H,W):' 
                          f'{images.shape}, Targets(N,F,C,D,H,W): {targets.shape}'
                          f' Batch loss: {loss.item():.5f}')
            
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("NaN/Inf in loss")
            
                if n_batches % window_size == 0:
                    elapsed = (time.time() - start_time) / 60
                    avg_all = running_loss / n_batches
                    avg_win = window_loss  / window_size
                    window_loss = 0.0
        
                    print(f"Time={elapsed:.2f} min, Batches={n_batches} | "
                          f"Overall Avg Loss={avg_all:.5f} | "
                         f"Window Avg Loss={avg_win:.5f}")

        # flush leftover if total_batches wasn't divisible by accum_steps
        if len(train_loader) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        return running_loss / n_batches
        
    @staticmethod
    def train_singlestep_cycle(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss, window_loss = 0.0, 0.0
        window_size = 1000
        n_batches   = 0
        start_time  = time.time()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking = True)
            targets = targets.to(device, non_blocking = True)
    
            optimizer.zero_grad()
            # Forward prediction: t -> t+1
            predictions_forward = model(images)
            loss_forward = criterion(predictions_forward, targets)
    
            # Backward prediction: t+1 -> t (reverse input-target pair along time dim)
            reversed_inputs  = targets.unsqueeze(1)  # reverse targets along time
            reversed_targets = images[:, -1]         # reverse inputs along time
            predictions_backward = model(reversed_inputs)
            loss_backward = criterion(predictions_backward, reversed_targets[:, -1])
    
            # Combined loss
            loss = (loss_forward + loss_backward) / 2
            loss.backward()
            optimizer.step()
            
            # collect the losses
            n_batches   += 1
            window_loss += loss.item()
            running_loss += loss.item()
            
            # print only few batch
            if is_main_process():
                if n_batches < 40:
                    print(f'[Batch no. {n_batches}, Images(N,T,F,C,D,H,W):' 
                          f'{images.shape}, Targets(N,F,C,D,H,W): {targets.shape}'
                          f' Batch loss: {loss.item():.5f}')
            
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("NaN/Inf in loss")
            
                if n_batches % window_size == 0:
                    elapsed = (time.time() - start_time) / 60
                    avg_all = running_loss / n_batches
                    avg_win = window_loss  / window_size
                    window_loss = 0.0
        
                    print(f"Time={elapsed:.2f} min, Batches={n_batches} | "
                          f"Overall Avg Loss={avg_all:.5f} | "
                         f"Window Avg Loss={avg_win:.5f}")
                    
        return running_loss / n_batches
    
    @staticmethod
    def train_singlestep_amp(model, train_loader, criterion, optimizer, device, 
                             scaler = None, enable_amp=False):
        model.train()
        if scaler is None: 
            scaler = amp.GradScaler(enabled=enable_amp)
        running_loss, window_loss = 0.0, 0.0
        window_size = 1000
        n_batches   = 0
        start_time  = time.time()
        
        for i, (images, targets) in enumerate(train_loader):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # NaN/Inf checks (good to keep)
            if torch.isnan(images).any() or torch.isinf(images).any():
                raise RuntimeError("NaN/Inf in input images")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                raise RuntimeError("NaN/Inf in targets")
    
            # mixed-precision forward:
            with amp.autocast(enabled=enable_amp, dtype = torch.bfloat16):
                outputs = model(images)
                loss    = criterion(outputs, targets)
            
            # scale → backward → (unscale +) step → update scale
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # bookkeeping
            n_batches   += 1
            window_loss += loss.item()
            running_loss += loss.item()
            
            if is_main_process():
                if n_batches < 40:
                    print(f'[Batch no. {n_batches + 1}, Images(N,T,F,C,D,H,W):' 
                          f'{images.shape}, Targets(N,F,C,D,H,W): {targets.shape}'
                          f' Batch loss: {loss.item():.4f}')
                    
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("NaN/Inf in loss")
                    
                if n_batches % window_size == 0:
                    elapsed = (time.time() - start_time) / 60
                    avg_all = running_loss / n_batches
                    avg_win = window_loss  / window_size
                    window_loss = 0.0
        
                    print(f"Time={elapsed:.2f} min, Batches={n_batches+1} | "
                          f"Overall Avg Loss={avg_all:.4f} | "
                         f"Window Avg Loss={avg_win:.4f}")
        
        return running_loss / n_batches

    @staticmethod
    def validate_singlestep(model, val_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        window_size = 1000
        n_batches = 0
        start_time  = time.time()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking = True)
                targets = targets.to(device, non_blocking = True)
                outputs = model(images)
                running_loss += criterion(outputs, targets).item()
                n_batches   += 1
                
                if is_main_process() and n_batches % window_size == 0:
                    elapsed = (time.time() - start_time) / 60
                    avg_all = running_loss / n_batches

                    print(f"[VAL] Time={elapsed:.2f} min, Batches={n_batches} | "
                          f"Overall Avg Loss={avg_all:.4f}")
                    
            return running_loss / n_batches
        
    @staticmethod
    def train_rollouts(model, loader, criterion, optimizer, device, lambda_rollouts, rollout_horizon):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for init, fut in loader:
            buffer = init.to(device)
            fut = fut.to(device)
            B = buffer.size(0)
            for t in range(min(fut.size(1), rollout_horizon)):
                pred = model(buffer)
                target = fut[:, t]
                loss = lambda_rollouts * criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * B
                n_steps += B
                buffer = torch.cat([buffer[:,1:], pred.detach().unsqueeze(1)], dim=1)
        return total_loss / n_steps

    @staticmethod
    def validate_rollouts(model, loader, criterion, device, lambda_rollouts):
        model.eval()
        total_loss = 0.0
        n_steps = 0
        with torch.no_grad():
            for init, fut in loader:
                buffer = init.to(device)
                fut = fut.to(device)
                B = buffer.size(0)
                for t in range(fut.size(1)):
                    pred = model(buffer)
                    loss = lambda_rollouts * criterion(pred, fut[:, t])
                    total_loss += loss.item() * B
                    n_steps += B
                    buffer = torch.cat([buffer[:,1:], pred.unsqueeze(1)], dim=1)
        return total_loss / n_steps

    @staticmethod
    def train_rollouts_scheduledsampling(model, loader, criterion, optimizer, device, lambda_rollouts, rollout_horizon, eps):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for init, fut in loader:
            buffer = init.to(device)
            fut = fut.to(device)
            B = buffer.size(0)
            for t in range(min(fut.size(1), rollout_horizon)):
                pred = model(buffer)
                target = fut[:, t]
                loss = lambda_rollouts * criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * B
                n_steps += B
                next_in = pred.detach() if random.random() < eps else target
                buffer = torch.cat([buffer[:,1:], next_in.unsqueeze(1)], dim=1)
        return total_loss / n_steps
