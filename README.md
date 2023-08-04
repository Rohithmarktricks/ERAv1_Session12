# ERAv1_Session12

This repository contains the Jupyter notebook and the required python modules to train a custom ResNet model on CIFAR10 dataset (with a few albumentations).

Modules are as follows:


1. ```[data_module.py](data_module.py)```: This contains the source code for the CIFAR10 dataset class ```AlbumDataset``` and a few transformations for train and test datasets respectively.

```Augmentaions on Train Dataset```
```
train_set_transforms = {
    'randomcrop': A.RandomCrop(height=32, width=32, p=0.2),
    'horizontalflip': A.HorizontalFlip(),
    'cutout': A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, fill_value=[0.49139968*255, 0.48215827*255 ,0.44653124*255], mask_fill_value=None),
    'normalize': A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    'standardize': ToTensorV2(),
}

```

```Augmentations on Test Dataset```
```
test_set_transforms = {
    'normalize': A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    'standardize': ToTensorV2()
}
```

2. ```[resnet_custom.py](resnet_custom.py)```: This module contains the source code for ```CustomResnet2``` model and pytorchLigthning framework has been used to build the model. And the model summary/ model skeleton looks as follows

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
    ResidualBlock-14          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         294,912
        MaxPool2d-16            [-1, 256, 8, 8]               0
      BatchNorm2d-17            [-1, 256, 8, 8]             512
             ReLU-18            [-1, 256, 8, 8]               0
           Conv2d-19            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-20            [-1, 512, 4, 4]               0
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
             ReLU-22            [-1, 512, 4, 4]               0
           Conv2d-23            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
             ReLU-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
             ReLU-28            [-1, 512, 4, 4]               0
    ResidualBlock-29            [-1, 512, 4, 4]               0
        MaxPool2d-30            [-1, 512, 1, 1]               0
           Linear-31                   [-1, 10]           5,130
          Softmax-32                   [-1, 10]               0
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.75
Params size (MB): 25.07
Estimated Total Size (MB): 31.84
----------------------------------------------------------------
```

3. ```[utils.py](utils.py)```: This module contains the helper functions to visulize the ```misclassfied_images``` and ```GradCAM``` images of the model predictions.

4. ```[Session_12.ipynb](Session_12.ipynb)```: This Jupyter notebook contains the training, evaluation, logging, saving the checkpoints and tensorboard visualization of the Custom Resnet model.

5. ```Visualization of Misclassified Images```
[Misclassified_images](images/misclassified_images.png)

6. ```Visualizatino of GradCAM output of Misclassified Images```
[Misclassified_images_GRADCAM](images/misclassified_gradCAMs.png)

7. ```Plots of the training/evaluation Logs```
#### Loss Plots
[Training_Loss](images/train_loss.png)

[validation_loss](images/valid_loss.png)

#### Accuracy Metrics Plots
[training_accuracy](images/epoch_train_accuracy.png)

[validation_accuracy](images/epoch_valid_accuracy.png)


8. ```TensorBoard```: To visualize the logs/metrics/loss metrics of both trianing and validation (datasets), you can find the ```logs``` folder in the repository or a subfolder ```version_x``` (x: represents the run) in the ```logs``` folder.

```
# To view this logs on the tensorboard, you can run the following commands in the Jupyter Notebook cell.

%load_ext tensorboard
%tensorboard --logdir logs
```

9. ```Saved Lightning model```
Refer to ```cifar10_resnet_epochs30.ckpt``` checkpoint file for using the model.
```
#the following code snippet can be used to load the checkpoints and using the trained model.

model = CustomResnet2.load_from_checkpoint(checkpoint_path="cifar10_resnet_epochs30.ckpt")
```
