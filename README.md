# Self-supervised Semantic Segmentation: Consistency over Transformation <br> <span style="float: rigth"><sub><sup>ICCV 2023 CVAMD Workshop</sub></sup></span>

[![arXiv](https://img.shields.io/badge/arXiv-2308.13442-b31b1b.svg)](https://arxiv.org/abs/2309.00143) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mindflow-institue/SSCT/blob/main/S3Net_colab.ipynb)


In our recent work, we propose a novel self-supervised algorithm called S<sup>3</sup>-Net for accurate medical image segmentation. The proposed S<sup>3</sup>-Net incorporates the Inception Large Kernel Attention (I-LKA) modules to enhance the network's ability to capture both contextual information and local intricacies, leading to precise semantic segmentation. The paper addresses the challenge of handling deformations commonly observed in medical images by integrating deformable convolutions into the architecture. This allows the network to effectively capture and delineate lesion deformations, leading to better-defined object boundaries. One key aspect of the proposed method is its emphasis on learning invariance to affine transformations, which are frequently encountered in medical scenarios. By focusing on robustness against geometric distortions, the model becomes more capable of accurately representing and managing such distortions. Moreover, to ensure spatial consistency and encourage the grouping of neighboring image pixels with similar features, the paper introduces a spatial consistency loss term. If this code helps with your research please consider citing our paper.

<br>
<p align="center">
  <img src="https://github.com/mindflow-institue/S3Net/assets/61879630/e19a0cb2-aa7c-487b-a4bd-419c689daa99" width="800">
</p>


## Updates
- July 25, 2023: Paper accepted in ICCV CVAMD 2023  
- If you found this paper useful, please consider checking out our previously accepted paper at MIDL 2023 [[Paper](https://openreview.net/forum?id=pp2raGSU3Wx)] [[GitHub](https://github.com/mindflow-institue/MS-Former)]

## Installation

```bash
pip install -r requirements.txt
```

## Run Demo
Put your input images in the ```input_images/image``` folder and just simply run the ```S3Net.ipynb``` notebook ;)

## Experiments
![output](https://github.com/mindflow-institue/S3Net/assets/61879630/dbdc9e16-2f8d-4d37-bbb7-c079f5a91e32)


## Citation
If this code helps with your research, please consider citing the following paper:
</br>

```python
@inproceedings{
  karimijafarbigloo2023self,
  title={Self-supervised Semantic Segmentation: Consistency over Transformation},
  author={Sanaz Karimijafarbigloo and Reza Azad and Amirhossein Kazerouni and Dorit Merhof},
  booktitle={ICCV 2023 Workshop on Computer Vision for Automated Medical Diagnosis},
  year={2023},
}
```
