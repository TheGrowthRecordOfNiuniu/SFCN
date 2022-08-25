 ## A feature catcher with excellent deblurring effect 

## News

August 25, 2022: Training codes are released 🔥  

August 24, 2022: Paper accepted at CVPR 2022 🎉  

## Abstract  

Camera shake and object movement are the two prime causes of blurred images. Efficient feature extraction is crucial for deblurring. Although the
existing methods have achieved remarkable achievements in the deblurring task, there is still room for improvement in effects. In this paper, we propose
an efficient architecture called the feature catcher network(FCN). In this multi-stage FCN architecture, the following design allows us to achieve
improvements in performance. Firstly, we propose to apply different calculated trust ratios to the output results of different stages before calculating
losses and then carry out the cumulative evaluation to update parameters for backpropagation. Secondly, we have improved Transformer to create a
query-key mechanism that is effect-friendly to the deblurring task. Thirdly, we propose a multi-stage attention block to make up for the loss of informa-
tion in high-level feature extraction. And the enhanced feature extraction block is employed to capture detailed information to ensure a greater de-
gree of image recovery. Fourthly, besides considering detailed features and high-level features at the same stage, we also construct residual supplements
for blurry images in the raw information mechanism. The experimental results on several datasets demonstrate that our model(FCN) outperforms state-of-the-art methods in terms of deblurring effect.  

## Network Architecture  

![mainFrame](https://user-images.githubusercontent.com/71067558/186557413-18d2f630-e5ce-4316-96b0-16c32fcf337b.png)


## Result  

![image](https://user-images.githubusercontent.com/71067558/186558200-012c0356-056d-452d-a96e-deb80c9a54af.png)


After training our model in the GoPro training set
through the training strategy mentioned in Section 4.2, we test the
effect of the model on the GoPro test set. Similar to the previous
methods, PSNR and SSIM are selected to measure our effective-
ness. As can be seen from the data in the above Table, our model exceeds
recent architectures in both PSNR and SSIM. In particular, we test
Restormer [29 ] architecture proposed this year on the GoPro dataset.
Our architecture (FCN) outperforms Restormer [29 ] 0.400 and 0.011
on PSNR and SSIM, respectively. In particular, the number of model
parameters for the Restormer architecture is 26.097M. However,
the number of model parameters of our architecture FCN is only
20.309M. Therefore, we successfully achieve the current state of
optimal effect in the deblurring field without adding additional
parameters. Some visual comparisons are shown in the following figure.  

![image](https://user-images.githubusercontent.com/71067558/186557698-a1beeca4-8b07-4b4c-9845-8e1d71247eda.png)




## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```


#### Testing on GoPro dataset
- Download [images](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing) of GoPro and place them in `./Datasets/GoPro/test/`
- Run
```
python test.py --dataset GoPro
```

#### Testing on HIDE dataset
- Download [images](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK?usp=sharing) of HIDE and place them in `./Datasets/HIDE/test/`
- Run
```
python test.py --dataset HIDE
```
