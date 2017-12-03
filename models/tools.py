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

def evaluate(model, data_iter, batch_size, alphabet_size, l0, cuda=False):
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

	if cuda:
	    vectors = vectors.cuda()

        output = model(vectors)

#        print(vectors.data.numpy().sum(axis=2))
        #_, predicted = torch.max(output, 1)
	if cuda:
            predicted = (output.cpu().data > 0.5)
        else :
	    predicted = (output.data > 0.5)

        total += len(labels)
        correct += np.equal(np.squeeze(predicted.numpy()), labels.data.numpy()).sum()

    return correct / float(total)

def load_checkpoint(model, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['state_dict'])

def training_loop(batch_size, total_batches, alphabet_size, l0, num_epochs, model, loss_, optim,
                  training_iter, validation_iter, train_eval_iter, save_file, comet_exp=None, cuda=False):
    step = 0
    epoch = 0
    best_val = 0

    while epoch <= num_epochs:
        model.train()
        vectors, labels = get_batch(next(training_iter))

        vectors = torch.stack(vectors)
        vectors_ = torch.unsqueeze(vectors, 1)
        one_hot = torch.FloatTensor(batch_size, alphabet_size, l0).zero_()
        one_hot.scatter_(1, vectors_, 1)

        vectors = Variable(one_hot) # batch_size, seq_len

        labels = torch.stack(labels)

        labels = Variable(labels.squeeze()).float()
        model.zero_grad()

	if cuda:
	   vectors = vectors.cuda()
           labels = labels.float().cuda()
        output = model(vectors)

        lossy = loss_(output.squeeze(), labels)
        #print(lossy.data[0])

        if comet_exp:
            comet_exp.log_metric("loss", lossy.cpu().data.numpy().mean())


        lossy.backward()

        optim.step()

        if step % total_batches == 0:
            #if epoch % 5 == 0:
            val_train = evaluate(model, train_eval_iter, batch_size, alphabet_size, l0, cuda)
            val_val = evaluate(model, validation_iter, batch_size, alphabet_size, l0, cuda)
            if comet_exp:
                comet_exp.log_metric("val_train_acc", val_train)
                comet_exp.log_metric("val_val_acc", val_val)
            print("Epoch %i; Step %i; Loss %f; Train acc: %f; Dev acc %f"
                      %(epoch, step, lossy.cpu().data[0], val_train, val_val))
            if val_val > best_val:
		best_val = val_val
		state = {
      		      'epoch': epoch,
                      'state_dict': model.state_dict(),
            	      'best_prec1': best_val,
                      'optimizer' : optim.state_dict(),
        	}

		torch.save(state, save_file)

            epoch += 1
        step += 1
