import torch
import time

# ----------------------------------------------------------------------
# MOD: Multi-task train/val helpers for CE+NS (2 loaders)
# ----------------------------------------------------------------------
def train_one_epoch_multitask(model, train_loaders, criterion, optimizer,
                              device, epoch, task_probs):
    model.train()
    start_time  = time.time()

    iters = [iter(l) for l in train_loaders]
    steps_per_epoch = max(len(l) for l in train_loaders)
    total_loss = 0.0
    total_steps = 0

    for step in range(steps_per_epoch):
        # --- choose task: 0 = CE, 1 = NS ---
        task_idx = torch.multinomial(task_probs, num_samples=1).item()

        # --- get loader and iterator for chosen task ---
        loader = train_loaders[task_idx]
        it = iters[task_idx]

        # --- get next batch ---
        try:
            batch = next(it)
        except StopIteration: # restart iterator if exhausted
            iters[task_idx] = iter(loader)
            it = iters[task_idx]
            batch = next(it)

        images, targets = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
            
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(images)
        loss = criterion(y_hat, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)

        if total_steps < 20:
            print(f"Cumtime: {time.time() - start_time:.3f}s | Inp: {images.shape}, Out: {targets.shape}, Avg loss: {avg_loss:.6f}")

    avg_loss = total_loss / max(total_steps, 1)

    return avg_loss


@torch.no_grad()
def validate_multitask(model, val_loaders, criterion, device):
    model.eval()
    results = {}
    for name, loader in zip(["ce", "ns"], val_loaders):
        total_loss = torch.tensor(0.0, device=device)
        total_count = torch.tensor(0, device=device, dtype=torch.long)

        for batch in loader:
            x, y = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            bs = x.size(0)

            total_loss += loss * bs
            total_count += bs

        avg_loss = (total_loss / total_count.clamp(min=1)).item()
        results[name] = avg_loss

    if len(results) > 0:
        combined = sum(results.values()) / len(results)
    else:
        combined = 0.0

    return combined, results