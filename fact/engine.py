"""Local and federated training loops."""

import time

import torch


def _cycle(loader):
    while True:
        yield from loader


def evaluate(model, loader, loss_fn, device):
    """Return (accuracy, mean loss) over the whole loader."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_sum += loss_fn(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def train(model, loader, test_loader, loss_fn, optimizer, device, epochs, steps_per_epoch, recorder, phase,
          scheduler=None, fedavg=None, local_steps=None, log_frequency=50):
    """Train for `epochs` epochs of `steps_per_epoch` steps each and return the final test loss.

    With `fedavg`, agents average their models every `local_steps` steps and after every epoch.
    Federated agents must all use the same `steps_per_epoch` so that the collectives line up;
    an agent with less data cycles through it.
    """
    batches = _cycle(loader)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        window_loss = window_correct = window_seen = 0
        for _ in range(steps_per_epoch):
            step += 1
            inputs, labels = next(batches)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            start = time.time()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            comp_time = time.time() - start

            comm_time = fedavg.average(model) if fedavg is not None and step % local_steps == 0 else 0.0

            batch_loss = loss.item()
            batch_correct = (outputs.argmax(1) == labels).sum().item()
            recorder.append(f"{phase}-train-loss", batch_loss)
            recorder.append(f"{phase}-train-acc-top1", batch_correct / labels.size(0))
            recorder.append(f"{phase}-comp-time", comp_time)
            recorder.append(f"{phase}-comm-time", comm_time)

            window_loss += batch_loss * labels.size(0)
            window_correct += batch_correct
            window_seen += labels.size(0)
            if step % log_frequency == 0:
                print(f"[rank {recorder.rank}] {phase} step {step}: loss {window_loss / window_seen:.3f}, "
                      f"accuracy {100 * window_correct / window_seen:.2f}%", flush=True)
                window_loss = window_correct = window_seen = 0

        if scheduler is not None:
            scheduler.step()
        if fedavg is not None:
            fedavg.average(model)
        accuracy, test_loss = evaluate(model, test_loader, loss_fn, device)
        recorder.append(f"{phase}-epoch-acc-top1", accuracy)
        recorder.append(f"{phase}-epoch-loss", test_loss)
        recorder.flush()
        print(f"[rank {recorder.rank}] {phase} epoch {epoch}: test accuracy {100 * accuracy:.2f}%, "
              f"test loss {test_loss:.4f}", flush=True)
    return test_loss
