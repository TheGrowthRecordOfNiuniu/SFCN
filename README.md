# FCN
A feature catcher with excellent deblurring effect

News
August 25, 2022: Training codes are released 🔥
August 24, 2022: Paper accepted at ICBDT 2022 🎉

Abstract:  Camera shake and object movement are the two prime causes of blurred images. Efficient feature extraction is crucial for deblurring.  Although the existing methods have achieved remarkable achievements in the deblurring task, there is still room for improvement in effects. In this paper, we propose an efficient architecture called the feature catcher network(FCN). In this multi-stage FCN architecture, the following design allows us to achieve improvements in performance. Firstly, we propose to apply different calculated trust ratios to the output results of different stages before calculating losses and then carry out the cumulative evaluation to update parameters for backpropagation. Secondly, we have improved Transformer to create a query-key mechanism that is effect-friendly to the deblurring task. Thirdly, we propose a multi-stage attention block to make up for the loss of information in high-level feature extraction. And the enhanced feature extraction block is employed to capture detailed information to ensure a greater degree of image recovery. Fourthly, besides considering detailed features and high-level features at the same stage, we also construct residual supplements for blurry images in the raw information mechanism. The experimental results on several datasets demonstrate that our model(FCN) outperforms state-of-the-art methods in terms of deblurring effect. The code and models will be available at https://github.com/XinyueZhangqdu/FCN.

Network Architecture

![mainFrame](https://user-images.githubusercontent.com/71067558/186551222-8b87a978-d228-4be2-866e-cbd53a6b6865.png)


Results
![result](https://user-images.githubusercontent.com/71067558/186551584-d10f1e92-d368-4ffb-97b7-748292602a6e.JPG)
As can be seen from the data in the above Table, our model exceeds recent architectures in both PSNR and SSIM. In particular, we test Restormer [29 ] architecture proposed this year on the GoPro dataset. Our architecture (FCN) outperforms Restormer [29 ] 0.400 and 0.011 on PSNR and SSIM, respectively. In particular, the number of model parameters for the Restormer architecture is 26.097M. However, the number of model parameters of our architecture FCN is only 20.309M. Therefore, we successfully achieve the current state of optimal effect in the deblurring field without adding additional parameters. Some visual comparisons are shown in the following figure.
![image](https://user-images.githubusercontent.com/71067558/186551886-1d3c05c0-69eb-4997-b9ba-6efb6f5cc493.png)
