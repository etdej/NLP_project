import torch
from torch.autograd import Variable
import numpy as np

# The following function gives batches of vectors and labels,
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["review"])
        labels.append(dict["rating"])
    return vectors, labels

def evaluate(model, data_iter, batch_size, alphabet_size, l0):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])

        vectors = torch.stack(vectors)
        vectors_ = torch.unsqueeze(vectors, 1)
        one_hot = torch.FloatTensor(batch_size, alphabet_size, l0).zero_()
        one_hot.scatter_(1, vectors_, 1)
        vectors = Variable(one_hot)
                
        labels = Variable(torch.stack(labels).squeeze())

        output = model(vectors)
        
        predicted = (output.data > 0.5)
        
        total += len(labels)
        correct += np.equal(np.squeeze(predicted.numpy()), labels.data.numpy()).sum()

    return correct / float(total)


def training_loop(batch_size, total_batches, alphabet_size, l0, num_epochs, model, loss_, optim,
                  training_iter, validation_iter, train_eval_iter, comet_exp=None):
    step = 0
    epoch = 0

    while epoch <= num_epochs:
        model.train()
        vectors, labels = get_batch(next(training_iter))

        vectors = torch.stack(vectors)
        vectors_ = torch.unsqueeze(vectors, 1)
        one_hot = torch.FloatTensor(batch_size, alphabet_size, l0).zero_()
        one_hot.scatter_(1, vectors_, 1)

        vectors = Variable(one_hot) # batch_size, seq_len
        
        labels = torch.stack(labels)

        labels = Variable(labels.squeeze())
        model.zero_grad()

        output = model(vectors)
        
        lossy = loss_(output.squeeze(), labels.float())
        
        if comet_exp:
            comet_exp.log_metric("loss", lossy.data.numpy().mean())
        
        lossy.backward()

        optim.step()

        if step % total_batches == 0:
            val_train = evaluate(model, train_eval_iter, batch_size, alphabet_size, l0)
            val_val = evaluate(model, validation_iter, batch_size, alphabet_size, l0)
            if comet_exp:
                comet_exp.log_metric("val_train_acc", val_train)
                comet_exp.log_metric("val_val_acc", val_val)
            print("Epoch %i; Step %i; Loss %f; Train acc: %f; Dev acc %f"
                      %(epoch, step, lossy.data[0], val_train, val_val))
            epoch += 1
        step += 1
