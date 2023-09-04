# :baby_chick: Official-PyTorch-Implementation-of-FCMEF
This is a PyTorch/GPU implementation of our Information Fusion 2022 paper: [Rethinking multi-exposure image fusion with extreme and diverse exposure levels: A robust framework based on Fourier transform and contrastive learning](https://www.sciencedirect.com/science/article/abs/pii/S1566253522002494).

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure2.png" width="720">
</p>

### For training
* Please refer to the [training code](https://github.com/miccaiif/FCMEF/blob/main/train_clmef_gray.py) for model training.

### For inference
* Please refer to the [testing code](https://github.com/miccaiif/FCMEF/blob/main/CLMEF_fusion2.py) for image fusion.

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure1.png" width="720">
</p>

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure7.png" width="720">
</p>

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure8.png" width="720">
</p>

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure9.png" width="720">
</p>

### For eMEFB and rMEFB Dataset
We construct two new MEF benchmark test sets, eMEFB and rMEFB, which can be used to evaluate MEF algorithms in fusing image pairs with extreme exposure levels and image pairs under diverse combinations of random exposure levels, respectively.
The eMEFB dataset and the rMEFB dataset can be downloaded from this [link](https://drive.google.com/file/d/1t5UVFwyjzfIDLlrdYrKOdZh3tyqE_aFI/view?usp=sharing).

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure2.png" width="720">
</p>

<p align="center">
  <img src="https://github.com/miccaiif/FCMEF/blob/main/figure6.png" width="720">
</p>

### More details
For more details, please refer to the paper and this [repo](https://github.com/miccaiif/TransMEF).


### Citation
If this work is helpful to you, please cite it as:
```
@article{qu2023rethinking,
  title={Rethinking multi-exposure image fusion with extreme and diverse exposure levels: A robust framework based on Fourier transform and contrastive learning},
  author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
  journal={Information Fusion},
  volume={92},
  pages={389--403},
  year={2023},
  publisher={Elsevier}
}
```

### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
