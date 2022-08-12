# Context Encoder Network for 2D Medical Image Segmentation
> [**CE-Net: Context Encoder Network for 2D Medical Image Segmentation**](https://arxiv.org/abs/1903.02740),            
> Zaiwang Gu, Jun Cheng, Huazhu Fu, Kang Zhou, Huaying Hao, Yitian Zhao, Tianyang Zhang, Shenghua Gao, Jiang Liu  
> *arXiv technical report ([arXiv 1903.02740](https://arxiv.org/abs/1903.02740))*         


Contact: [guzw@i2r.a-star.edu.sg](mailto:guzw@i2r.a-star.edu.sg) or [guzaiwang01@gmail.com](mailto:guzaiwang01@gmail.com). Any questions or discussions are welcomed! 

## Abstract 

Medical image segmentation is an important step 
in medical image analysis. With the rapid development of
convolutional neural network in image processing, deep learning
has been used for medical image segmentation, such as optic
disc segmentation, blood vessel detection, lung segmentation, cell
segmentation, etc. Previously, U-net based approaches have been
proposed. However, the consecutive pooling and strided convolutional operations lead to the loss of some spatial information. In
this paper, we propose a context encoder network (referred to as
CE-Net) to capture more high-level information and preserve
spatial information for 2D medical image segmentation. CENet mainly contains three major components: a feature encoder
module, a context extractor and a feature decoder module. We
use pretrained ResNet block as the fixed feature extractor. The
context extractor module is formed by a newly proposed dense
atrous convolution (DAC) block and residual multi-kernel pooling
(RMP) block. We applied the proposed CE-Net to different 2D
medical image segmentation tasks. Comprehensive results show
that the proposed method outperforms the original U-Net method
and other state-of-the-art methods for optic disc segmentation,
vessel detection, lung segmentation, cell contour segmentation
and retinal optical coherence tomography layer segmentation.


## Use CE-Net
Please start up the "visdom" before running the main.py.
Then, run the main.py file.

We have uploaded the DRIVE dataset to run the retinal vessel detection. The other medical datasets will be
uploaded in the next submission.

The submission mainly contains:
1. architecture (called CE-Net) in networks/cenet.py
2. multi-class dice loss in loss.py
3. data augmentation in data.py

Update:
We have modified the loss function. 
The cuda error (or warning) will not occur. 

Update:
The test code has been uploaded. 
Besides, we release a pretrained model, which achieves 0.9819 in the AUC scor in the DRIVE dataset. 

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @article{gu2019net,
      title={Ce-net: Context encoder network for 2d medical image segmentation},
      author={Gu, Zaiwang and Cheng, Jun and Fu, Huazhu and Zhou, Kang and Hao, Huaying and Zhao, Yitian and Zhang, Tianyang and Gao, Shenghua and Liu, Jiang},
      journal={IEEE transactions on medical imaging},
      volume={38},
      number={10},
      pages={2281--2292},
      year={2019},
      publisher={IEEE}
    }
    
The manuscript has been accepted in TMI.


