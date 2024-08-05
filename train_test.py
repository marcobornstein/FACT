from mpi4py import MPI
import torch
import time


def local_training(model, trainloader, testloader, device, loss_fn, optimizer, epochs, log_frequency, recorder,
                   scheduler):
    i = 1
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        total_examples = 0
        correct_prediction = 0
        running_time = 0.0
        model.train()
        for data in trainloader:

            # take one training step
            train_step(i, model, data, loss_fn, optimizer, recorder, None, device, total_examples,
                       correct_prediction, running_loss, running_time, 1, log_frequency, federated=False)

            # update counter
            i += 1

        if scheduler is not None:
            scheduler.step()
        # spit out the final accuracy after training
        if epoch == epochs:
            final_loss = test(model, loss_fn, testloader, device, recorder, epoch, return_loss=True)
            return final_loss
        else:
            test(model, loss_fn, testloader, device, recorder, epoch)

        MPI.COMM_WORLD.Barrier()


def federated_training(model, communicator, trainloader, testloader, device, loss_fn, optimizer, epochs, log_frequency,
                       recorder, scheduler, local_steps=3):
    i = 1
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        total_examples = 0
        correct_prediction = 0
        running_time = 0.0
        model.train()
        for data in trainloader:

            # take one training step
            train_step(i, model, data, loss_fn, optimizer, recorder, communicator, device, total_examples,
                       correct_prediction, running_loss, running_time, local_steps, log_frequency, federated=True)

            # update counter
            i += 1

        if scheduler is not None:
            scheduler.step()
        # spit out the final accuracy after training
        communicator.sync_models(model)
        if epoch == epochs:
            # ensure models are synced so that final test accuracies are all equivalent
            final_loss = test(model, loss_fn, testloader, device, recorder, epoch, return_loss=True, local=False)
            return final_loss
        else:
            test(model, loss_fn, testloader, device, recorder, epoch, local=False)

        MPI.COMM_WORLD.Barrier()


def train_step(i, model, data, loss_fn, optimizer, recorder, communicator, device, total_examples, correct_prediction,
               running_loss, running_time, local_steps, log_frequency, federated=False):

    # get input and true label, place on GPU
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    init_time = time.time()
    # forward + backward + optimize
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    # compute running accuracy
    num_examples = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    num_correct = (predicted == labels).sum().item()
    total_examples += num_examples
    correct_prediction += num_correct

    # print statistics
    loss_val = loss.item()
    running_loss += (loss_val * num_examples)
    comp_time = time.time() - init_time
    running_time += comp_time

    # print every X mini-batches
    if i % log_frequency == 0:
        print(f' [rank {recorder.rank}] step: {i}, loss: {running_loss / total_examples:.3f}, '
              f'accuracy: {100 * correct_prediction / total_examples:.3f}%, time: {running_time / log_frequency:.3f}')
        running_loss = 0.0
        running_time = 0.0
        total_examples = 0
        correct_prediction = 0
        recorder.save_to_file()

    # perform FedAvg/D-SGD every K steps (IF FEDERATED)
    if i % local_steps == 0 and federated:
        comm_time = communicator.communicate(model)
    else:
        comm_time = 0

    recorder.add_new(comp_time, comm_time, num_correct / num_examples, loss_val, local=False)

    return running_loss, running_time, total_examples, correct_prediction


def nonuniform_federated_training(model, communicator, trainloader, testloader, device, loss_fn, optimizer, max_steps,
                                  epochs, log_frequency, recorder, scheduler, local_steps=6):
    i = 1
    total_steps = max_steps * epochs
    while True:
        running_loss = 0.0
        total_examples = 0
        correct_prediction = 0
        running_time = 0.0
        model.train()
        for data in trainloader:

            # take one training step
            train_step(i, model, data, loss_fn, optimizer, recorder, communicator, device, total_examples,
                       correct_prediction, running_loss, running_time, local_steps, log_frequency, federated=True)

            if i % max_steps == 0:
                epoch = i//max_steps
                if scheduler is not None:
                    scheduler.step()
                communicator.sync_models(model)
                if i % total_steps == 0:
                    # spit out the final accuracy after training
                    final_loss = test(model, loss_fn, testloader, device, recorder, epoch, return_loss=True, local=False)
                    return final_loss
                else:
                    test(model, loss_fn, testloader, device, recorder, epoch, local=False)

            # update counter
            i += 1


def test(model, loss_fn, test_dl, device, recorder, epoch_num, test_batches=30,
         epoch=True, return_loss=False, local=True):
    correct = 0
    total = 0
    test_loss = 0
    i = 1
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()
            loss_val = loss_fn(outputs, labels).item()
            test_loss += (loss_val * batch_size)

            # only sample a few batches if doing random sample test
            if i == test_batches and not epoch:
                break

            i += 1

    test_acc = correct / total
    test_loss = test_loss / total
    recorder.add_test_stats(test_acc, test_loss, epoch=epoch, local=local)
    if epoch:
        print(f'[rank {recorder.rank}] epoch {epoch_num}, test accuracy & loss on {total} '
              f'images: {100 * test_acc: .3f}% and {test_loss:}')
    else:
        print(f'[rank {recorder.rank}] test accuracy & loss on {total} images: {100 * test_acc: .3f}% and {test_loss:}')
    if return_loss:
        return test_loss
