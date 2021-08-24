# Center Loss
Pytorch Center Loss implementation is based on [pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss).

## A Toy Example
Dataset: MNIST <br>
Network: Modified LeNet <br>
Batch size: 128 <br>
Epochs: 50 <br>
Learning rate: 1e-2 <br>
Optimizer: SGD <br>

### Cross-Entropy Loss
*Epoch 1, 5, 10, 20, 50*

![s1](./images/softmax_epoch_1.png)
![s5](./images/softmax_epoch_5.png)
![s10](./images/softmax_epoch_10.png)
![s20](./images/softmax_epoch_20.png)
![s50](./images/softmax_epoch_50.png)


### Cross-Entropy Loss and Center Loss
*Epoch 1, 5, 10, 20, 50* <br>
**Lambda = 1**

![sc1](./images/softmax_center_epoch_1.png)
![sc5](./images/softmax_center_epoch_5.png)
![sc10](./images/softmax_center_epoch_10.png)
![sc20](./images/softmax_center_epoch_20.png)
![sc50](./images/softmax_center_epoch_50.png)

**Lambda = 0.1**

![sc0.1_1](./images/softmax_center_lambda_0.1_epoch_1.png)
![sc0.1_5](./images/softmax_center_lambda_0.1_epoch_5.png)
![sc0.1_10](./images/softmax_center_lambda_0.1_epoch_10.png)
![sc0.1_20](./images/softmax_center_lambda_0.1_epoch_20.png)
![sc0.1_50](./images/softmax_center_lambda_0.1_epoch_50.png)

**Lambda = 0.01**

![sc0.01_1](./images/softmax_center_lambda_0.01_epoch_1.png)
![sc0.01_5](./images/softmax_center_lambda_0.01_epoch_5.png)
![sc0.01_10](./images/softmax_center_lambda_0.01_epoch_10.png)
![sc0.01_20](./images/softmax_center_lambda_0.01_epoch_20.png)
![sc0.01_50](./images/softmax_center_lambda_0.01_epoch_50.png)

**Lambda = 0.001**

![sc0.001_1](./images/softmax_center_lambda_0.001_epoch_1.png)
![sc0.001_5](./images/softmax_center_lambda_0.001_epoch_5.png)
![sc0.001_10](./images/softmax_center_lambda_0.001_epoch_10.png)
![sc0.001_20](./images/softmax_center_lambda_0.001_epoch_20.png)
![sc0.001_50](./images/softmax_center_lambda_0.001_epoch_50.png)
