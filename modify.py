'''To modify the code for 13 classes, you will need to make changes to several parts of the code.

In the DataGenerator class, you will need to modify the __init__ method to read in the labels for 13 classes instead of just 2. You will also need to modify the __getitem__ method to return the labels for the corresponding 13 classes.

In the train function, you will need to modify the loss function to use a suitable loss function for multi-class classification. For example, you could use nn.CrossEntropyLoss, which combines nn.LogSoftmax and nn.NLLLoss in a single class.

You will also need to modify the output layer of your neural network to have 13 output units instead of just 1.

Here is some sample code that you can use as a reference to modify your existing code for multi-class classification:'''

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 13)  # 13 output units for 13 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


'''In addition, you will need to modify the loss function in the train function:'''

criterion = nn.CrossEntropyLoss()
'''And modify the __getitem__ method in the DataGenerator class to return the labels for the corresponding 13 classes:'''

# label = np.zeros(13)
# for i in self.labels[index]:
#     label[i] = 1
# return image, label


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data, label_path, test=False):
        self.data = data
        self.label_path = label_path
        self.num_classes = None
        self.test = test
        self.__dataset_information()

    def __dataset_information(self):
        with open(self.label_path) as f:
            labels = json.load(f)

        self.numbers_of_data = len(labels)
        self.index_name_dic = dict()
        self.class_count = {}

        for idx, (name, label) in enumerate(labels.items()):
            self.index_name_dic[idx] = [name, label]
            self.class_count[label] = self.class_count.get(label, 0) + 1

        self.num_classes = len(self.class_count)
        output(
            f"Load {self.numbers_of_data} videos with {self.num_classes} classes"
        )
        print(
            f"Load {self.numbers_of_data} videos with {self.num_classes} classes"
        )

    def __len__(self):
        return self.numbers_of_data

    def __getitem__(self, idx):
        ids = self.index_name_dic[idx]
        x, y = self.__data_generation(ids)
        return x, y

    def __data_generation(self, ids):
        name, label = ids
        y = torch.LongTensor([label])

        clips = []
        size = 5 if self.test else 1

        for _ in range(size):
            x = np.load(os.path.join(self.data, f"{name}.mp4.npy"))
            start = x.shape[0] - 16

            if start > 0:
                start = np.random.randint(0, start)
                x = x[start:][:16]
            else:
                start = np.random.randint(0, 1)
                x = np.array(x)[start:]

            x = (x - min_xyz) / (max_xyz - min_xyz)
            pad_x = np.zeros((16, 478, 3))

            if x.shape[0] == 16:
                pad_x = x
            else:
                pad_x[:x.shape[0]] = x

            pad_x = torch.FloatTensor(pad_x)
            clips.append(pad_x)

        clips = torch.stack(clips, 0)
        return clips, y


'''Here are the changes made:

Added a new instance variable num_classes to keep track of the number of classes in the dataset.
Changed the data type of the label y from FloatTensor to LongTensor to support multi-class classification.
Updated the data_generation function to use the new label format and to return a LongTensor instead of a FloatTensor.
Added a class_count dictionary to keep track of the number of samples for each class. This can be useful for data balancing or stratified sampling in future.'''

######

def train(epochs,training_generator,test_generator,file):

    con = []
    net = TemporalModel()
    net.cuda()

    lr = 0.0005
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay= 0.0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[299], gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    start_time = time.time()
    best_accuracy = 0

    for epoch in range(epochs):
        train_loss  = 0
        pred_label = []
        true_label = []
        number_batch = 0
        for x, y in tqdm(training_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):
            if torch.cuda.device_count() > 0:
                x = x.cuda()
                y = y.cuda()

            b,d,t,n,c = x.size()
            x = x.view(-1,t,n,c)
            pred = net(x)
            loss = loss_func(pred,y.argmax(dim=1))
            pred_y = torch.argmax(pred, dim=1)
            pred_label.append(pred_y)
            true_label.append(y.argmax(dim=1))

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            number_batch += 1
            lr = lr_scheduler.get_last_lr()[0]

        lr_scheduler.step()
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        output('Epoch: ' + 'train' + str(epoch) +
              '| train accuracy: ' + str(train_accuracy.item())  +
              '| train  loss: ' + str(train_loss / number_batch))
        print('Epoch: ' + 'train' + str(epoch) +
              '| train accuracy: ' + str(train_accuracy.item())  +
              '| train  loss: ' + str(train_loss / number_batch))

        net.eval()
        pred_label = []
        pred_avg   = []
        true_label = []
        with torch.no_grad():
            for x, y in tqdm(test_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):

                if torch.cuda.device_count() > 0:
                    x = x.cuda()
                    y = y.cuda()

                b,d,t,n,c = x.size()
                x = x.view(-1,t,n,c)
                pred_y    = net(x)
                pred_mean = (pred_y.view(b,d).mean(1,keepdim = True) >= 0.5).float().cpu().detach()
                pred_y    = torch.argmax(pred_y, dim=1)
                pred_label.append(pred_y)
                pred_avg.append(pred_mean)
                true_label.append(y.cpu())

            pred_label = torch.cat(pred_label,0)
            pred_avg   = torch.cat(pred_avg,0)
            true_label = torch.cat(true_label,0)

            test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            test_avg      = torch.sum(pred_avg   == true_label).type(torch.FloatTensor) / true_label.size(0)
            con.append([epoch,test_accuracy])
            output('test accuracy: ' + str(test_accuracy.item()) +
                  '| avg accuracy: '  + str(test_avg.item()))
            print(Fore.GREEN + 'test accuracy: ' + str(test_accuracy.item()) +
                  '| avg accuracy: '  + str(test_avg.item()))

            if test_accuracy > best_accuracy:
                filepath = f"uva/{file}-{epoch:}-{loss}-{test_accuracy}.pt"
                torch.save(net.state_dict(), filepath)
                #   torch.save(net, filepath)
                #   test_frames(f'{test_accuracy}={test_f}')
                best_accuracy = test_accuracy

        net.train()

        output(f"ETA Per Epoch:{(time.time() - start_time) / (epoch + 1)}")
        # print(f"ETA Per Epoch:{(time.time() - start_time) / (epoch + 1)}")

    best_v = max(con,key = lambda x:x[1])
    global perf
    perf += f"best accruacy is {best_v[1]} in epoch {best_v[0]}" + "\n"
    output(perf)

################
class TemporalModel(nn.Module):
    
    def __init__(self):
        super(TemporalModel,self).__init__()
                
        self.encoder  =  CurveNet() # curve aggregation, needed for Point Clouds Shape Analysis. 
        self.downsample = nn.Sequential(
                            nn.Conv1d(478, 32, kernel_size=1, bias=False),
                            nn.BatchNorm1d(32),
                            # nn.Dropout(p=0.25), #* NEW
                            #nn.ReLU(inplace=True),
                            #nn.Conv1d(128, 32, kernel_size=1, bias=False),
                            #nn.BatchNorm1d(32),
                            )
        
        self.transformer = Transformer(256, 6, 4, 256//4, 256 * 2, 0.1)
        self.time = Transformer(256, 3, 4, 256//4, 256 * 2, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,13), #* MODIFIED
            nn.Sigmoid()
        )
        
    def forward(self,x):
        b,t,n,c = x.size()
    
        x = rearrange(x, "b t n c -> (b t) c n")
        x = rearrange(self.dropout(self.encoder(x)), "b c n -> b n c") 
        x = self.downsample(x).view(b,t,32,-1) #b t 32 c
        x = self.transformer(x,swap = True).view(b,t,-1,256).mean(2)
        x = self.time(x).mean(1)
        x = self.mlp_head(x)
        return x

################
class TemporalModel(nn.Module):
    def __init__(self):
        super(TemporalModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64,
                      128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128,
                      256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,
                      512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=256,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 13)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 3, 1, 2)
        b, c, h, w = cnn_out.size()
        cnn_out = cnn_out.view(b, c, h * w)
        cnn_out = cnn_out.permute(0, 2, 1)
        rnn_out, _ = self.rnn(cnn_out)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
################
# Last Modified: 
################
def train(epochs, training_generator, test_generator, file):
    
    con = []      
    net = TemporalModel()
    net.cuda()
    
    lr = 0.0005
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[299], gamma=0.1)
    loss_func = nn.CrossEntropyLoss()  # use CrossEntropyLoss for multi-class classification
    start_time = time.time()
    best_accuracy = 0

    for epoch in range(epochs):
        train_loss = 0
        pred_label = []
        true_label = []
        number_batch = 0
        for x, y in tqdm(training_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):
            if torch.cuda.device_count() > 0:
                x = x.cuda()
                y = y.cuda()
                
            b, d, t, n, c = x.size()
            x = x.view(-1, t, n, c)
            pred = net(x)
            loss = loss_func(pred, y)
            pred_y = torch.argmax(pred, dim=1)  # convert the predicted probabilities to class labels
            pred_label.append(pred_y)
            true_label.append(y)
            
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            number_batch += 1
            lr = lr_scheduler.get_last_lr()[0]

        lr_scheduler.step()
        pred_label = torch.cat(pred_label, 0)
        true_label = torch.cat(true_label, 0)
        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        output('Epoch: ' + 'train' + str(epoch) + 
              '| train accuracy: ' + str(train_accuracy.item())  + 
              '| train  loss: ' + str(train_loss / number_batch))
        print('Epoch: ' + 'train' + str(epoch) + 
              '| train accuracy: ' + str(train_accuracy.item())  + 
              '| train  loss: ' + str(train_loss / number_batch))
        
        net.eval()
        pred_label = []
        true_label = []
        with torch.no_grad():
            for x, y in tqdm(test_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):

                if torch.cuda.device_count() > 0:
                    x = x.cuda()
                    y = y.cuda()

                b, d, t, n, c = x.size()
                x = x.view(-1, t, n, c)
                pred_y = net(x)
                pred_y = torch.argmax(pred_y, dim=1)  # convert the predicted probabilities to class labels
                pred_label.append(pred_y)
                true_label.append(y.cpu())

            pred_label = torch.cat(pred_label, 0)
            true_label = torch.cat(true_label, 0)

            test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            con.append([epoch,test_accuracy])
            output('test accuracy: ' + str(test_accuracy.item()))
            print(Fore.GREEN + 'test accuracy: ' + str(test_accuracy.item()))

            if test_accuracy > best_accuracy:
                filepath = f"uva/{file}-{epoch:}-{loss}-{test_accuracy}.pt"
                torch.save(net.state_dict(), filepath)
                best_accuracy = test_accuracy

        net.train()

        output(f"ETA Per Epoch:{(time.time() - start_time)
