# STT-BOX v1.0

Example of visualization codes for the article "A deep learning-based system trained for gastrointestinal stromal tumor screening can identify multiple types of soft tissue tumors" published in *The American Journal of Pathology*.

## Environments and Requirements

* Ubuntu 16.04.6 LTS
* CPU Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
* GPU NVIDIA Tesla T4 16G
* CUDA 10.0.130
* Pytorch 1.2.0
* Python 3.7.6
* Numpy 1.18.1
* Openslide
* Mathplotlib
* PIL
* opencv (cv2)
* skimage

## Implementation
#### 1) Prepare the image waiting for inference
Please prepare the Whole-Slide Image (WSI) for inference. The file format of the image should be ".svs", ".tiff", ".ndpi", and etc. due to th usage of [OpenSlide](https://openslide.org/). Then, modify the parameter for image path in `inference_STT-BOX.py`.
```python
# inference_STT-BOX.py
class Option(object):
    def __init__(self):
        self.img_path = './your_image_name.svs' # example of the image path 
```
#### 2) Pre-procss the trained model
After decompressing the five compressed packages in the folder "model", you will receive a file named "STT_BOX_v1.pth". Please place this file in "`./model/STT_BOX_v1.pth`".

#### 3) Obtain the output heatmap
Run the file `inference_STT-BOX.py`. If a specific GPU is selected (such as No. 3), run `CUDA_VISIBLE_DEVICES=3 python inference_STT-BOX.py`. Finally, you will obtain the heatmap, e.g., "`your_file_name_heatmap.png`". Note that the codes are just examples to obtain visualization, and different outputs will be gained when the parameters are changed.
## Citation
Please cite:
Zhu et al. A deep learning-based system trained for gastrointestinal stromal tumor screening can identify multiple types of soft tissue tumors, in *The American Journal of Pathology*, 2023. (in press)



