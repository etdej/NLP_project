import torch
import torch.nn as nn

class Char_CNN_Small(nn.Module):

    def __init__(self,
                 l0,
                 alphabet_size,
               	 dropout_rate,
		 nb_classes,
                 batch_size,
                 ):
        super(Char_CNN_Small,self).__init__()
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.alphabet_size = alphabet_size
        self.l0 = l0
        self.max_pool = nn.MaxPool1d(3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = nn.Conv1d(self.alphabet_size,256,7)
        self.conv2 = nn.Conv1d(256,256,7)
        self.conv3 = nn.Conv1d(256,256,3)
        self.conv4 = nn.Conv1d(256,256,3)
        self.conv5 = nn.Conv1d(256,256,3)
        self.conv6 = nn.Conv1d(256,256,3)

        l6 = int((l0 - 96)/27)
#        l6 = int((l0 - 6)/9 - 2)
        in_feat = l6*256

        self.linear1 = nn.Linear(in_feat,1024)
        self.linear2 = nn.Linear(1024,1024)


        self.classifier = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        batch_size = x.size(0)

        out = x
        out = self.conv1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = out.view(batch_size, -1)

        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)


        out = self.classifier(out)
        out = self.sigmoid(out)

        return out

    def init_weights(self):

        mean = 0
        std = 0.05
        nn.init.normal(self.conv1.weight, mean=mean, std=std)
        nn.init.normal(self.conv2.weight, mean=mean, std=std)
        nn.init.normal(self.conv3.weight, mean=mean, std=std)
        nn.init.normal(self.conv4.weight, mean=mean, std=std)
        nn.init.normal(self.conv5.weight, mean=mean, std=std)
        nn.init.normal(self.conv6.weight, mean=mean, std=std)

        nn.init.normal(self.linear1.weight, mean=mean, std=std)
        nn.init.normal(self.linear2.weight, mean=mean, std=std)
        nn.init.normal(self.classifier.weight, mean=mean, std=std)


if __name__ == '__main__':
    l0 = 1014
    alphabet_size = 68
    nb_classes = 10
    model = Char_CNN_Small( l0, alphabet_size, nb_classes, batch_size)
