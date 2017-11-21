import torch
import torch.nn as nn

class Char_CNN_Small(nn.Module):

    def __init__(self,
                 fully_layers,
                 l0,
                 alphabet_size,
                 nb_classes,
                 batch_size,
                 ):
        super(Char_CNN_Small,self).__init__()

        self.conv_layers = [
                    [256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
                    ]
        self.fully_layers = fully_layers
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.alphabet_size = alphabet_size
        self.l0 = l0

        self.convs = []
        self.linear = []
        self.max_pool = nn.MaxPool1d(3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        in_feat = alphabet_size
        for out_feat, kernel_size, max_pool in self.conv_layers:
            conv = nn.Conv1d(in_feat, out_feat, kernel_size)
            self.convs.append(conv)
            in_feat = out_feat

        l6 = int((l0 - 96)/27)
        in_feat = l6*out_feat

        for out_feat in fully_layers:
            self.linear.append(nn.Linear(in_feat, out_feat))
            in_feat = out_feat
        
        self.classifier = nn.Linear(in_feat, nb_classes)

        if self.nb_classes == 2:
            self.classifier = nn.Linear(in_feat, 1)
            self.class_non_lin = nn.Sigmoid()
            
        else:
            self.classifier = nn.Linear(in_feat, nb_classes)
            self.class_non_lin = nn.Softmax()


    def forward(self, x):
        out = x
        for conv in self.convs[:2]:
            out = conv(out)
            out = self.relu(out)
            out = self.max_pool(out)
            print(out.data.numpy().max())
            print(out.data.numpy().min())

        for conv in self.convs[2:5]:
            out = conv(out)
            out = self.relu(out)
            print(out.data.numpy().max())
            print(out.data.numpy().min())

        out = self.convs[5](out)
        out = self.relu(out)
        out = self.max_pool(out)
        print(out.data.numpy().max())
        print(out.data.numpy().min())
        

        out = out.view(self.batch_size, -1)
        print(out.data.numpy().max())
        print(out.data.numpy().min())


        for lin in self.linear:
            out = lin(out)
            out = self.relu(out)
            out = self.dropout(out)
            print(out.data.numpy().max())
            print(out.data.numpy().min())

#        print(out.data.numpy().shape)
#        print(out.data.numpy()[:, :10])
        out = self.classifier(out)
       # print(out.data.numpy())
              
        out = self.class_non_lin(out)
        print(out.data.numpy().max())
        print(out.data.numpy().min())
  
        return out

    def init_weights(self):
        for conv in self.convs:
            nn.init.normal(conv.weight, mean=0, std=0.05)
        
        for lin in self.linear:
            nn.init.normal(lin.weight, mean=0, std=0.05)

if __name__ == '__main__':
    fully_layers = [1024, 1024]
    l0 = 1014
    alphabet_size = 68
    nb_classes = 10
    batch_size = 8

    learning_rate = 0.008
    num_epochs = 20
    model = Char_CNN_Small(fully_layers, l0, alphabet_size, nb_classes, batch_size)