# Assignment

## Business Problem
- 99.4% validation accuracy
- Less than 20k Parameters
- You can use anything from above you want.
- Less than 20 Epochs
- Have used BN, Dropout, a Fully connected layer, have used GAP.
- To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.

## Model Architecture


```
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_b1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(8)
    self.conv_b1_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(16)

    self.conv_b1_drop = nn.Dropout2d(p=0.5)
    self.conv_b1_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
    self.conv_b3_drop = nn.Dropout2d(p=0.25)

    self.conv_b1_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
    
    # [<batch_size>, <channels>, <width>, <height>]
     # so, let's do the spatial averaging: reduce <width> and <height> to 1
    self.fc1 = nn.Linear(in_features=32 * 1 * 1, out_features=10)
    
    # self.pool = nn.AdaptiveAvgPool2d(1)
    # self.fc2 = nn.Linear(in_features=16, out_features=10)

  def forward(self, input_image):
    # input layer
    x = input_image

    ### mninst image handling logic

    # conv1 layer
    x = self.conv_b1_1(x)
    x = self.bn1(x)
    x = F.relu(x)
    # x = F.max_pool2d(x, kernel_size=2, stride=2) # 32, 6, 13, 13
    # print(x.shape)
    
    x =self.conv_b1_drop(x)

    # conv2 layer
    x = self.conv_b1_2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2) # 32, 12, 5, 5
    # print(x.shape)


    # conv3 layer
    x = self.conv_b1_3(x)  # 32, 24, 6, 6
    x = self.conv_b3_drop(x)
    x = F.relu(x)
    
    # x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    # print(x.shape)  
    x = x.reshape(-1, 32 * 1 * 1)
    # print(x.shape)  
    
    # print(x.shape)  
    x = self.fc1(x)
    # x = self.fc2(x)
    x = F.log_softmax(x)
    return x
```
