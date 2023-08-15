# Self-supervised Semantic Segmentation: Consistency over Transformation <br> <span style="float: rigth"><sub><sup>$\text{(\textcolor{teal}{ICCV 2023})}$</sub></sup></span>
[Paper is available]()

In our recent work we propose a novel self-supervised algorithm called S$^3$-Net for accurate medical image segmentation. The proposed S$^3$-Net incorporates the Inception Large Kernel Attention (I-LKA) modules to enhance the network's ability to capture both contextual information and local intricacies, leading to precise semantic segmentation. The paper addresses the challenge of handling deformations commonly observed in medical images by integrating deformable convolutions into the architecture. This allows the network to effectively capture and delineate lesion deformations, leading to better-defined object boundaries. One key aspect of the proposed method is its emphasis on learning invariance to affine transformations, which are frequently encountered in medical scenarios. By focusing on robustness against geometric distortions, the model becomes more capable of accurately representing and managing such distortions. Moreover, to ensure spatial consistency and encourage the grouping of neighboring image pixels with similar features, the paper introduces a spatial consistency loss term. If this code helps with your research please consider citing our paper.


![S3Net](https://github.com/mindflow-institue/MS-Former/assets/61879630/fe8910b5-b9ed-4cf7-be80-50b8398e13b5)


## Updates
- July 25, 2023: Paper accepted in ICCV 2023 workshop 

## Installation

```bash
pip install -r requirements.txt
```

## Run Demo
Put your input images in the ```input/image``` folder and just simply run the ```MSFormer.ipynb``` notebook ;)

## Citation
If this code helps with your research, please consider citing the following paper:
</br>

```python
@inproceedings{
  karimijafarbigloo2023msformer,
  title={{MS}-Former: Multi-Scale Self-Guided Transformer for Medical Image Segmentation},
  author={Sanaz Karimijafarbigloo and Reza Azad and Amirhossein Kazerouni and Dorit Merhof},
  booktitle={Medical Imaging with Deep Learning},
  year={2023},
  url={https://openreview.net/forum?id=pp2raGSU3Wx}
}
```
