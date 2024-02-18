
### Lightweight Single Image Super-Resolution Network Integrating CNN and Transformer

[Zhu Kai](https://github.com/kylechuuuuu)

*Recently, remarkable progress has been achieved in single image super-resolution using methods based on CNN and Transformer architectures. However, existing approaches often construct a substantial number of network layers, leading to a significant increase in computational complexity and memory consumption, thereby limiting the practical deployment and usability of the models. To address this issue, we propose an alternating CNN- Transformer block and Integrative CNN Efficient Transformer for single image super-resolution. We enhance feature extraction efficiency by combining CNN within and between Transformer modules. In addition, we propose two novel structures: multi-branch gated convolution and parallel channel attention, aiming to efficiently extract local spatial information and global channel information from images. Extensive experiments demonstrate that our model achieves high performance while maintaining low model complexity. The proposed model achieves PSNR metrics of 32.32 and 26.25 on the benchmark datasets Set5 and Urban100, respectively, at a scale of ×4. The model parameters and FLOPs are 578K and 26.9G. In comparison to other lightweight super-resolution models, our proposed model outperforms them at lower computational costs.

## How to Use

### 1 Preparation

#### 1.1 Environment

Use the following command to build the Python environment:

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # Mainland China only!
pip install torch torchvision basicsr einops timm matplotlib
```

#### 1.2 Dataset

You can download the datasets you need from our [OneDrive](https://1drv.ms/u/s!AqKlMh-sml1mw362MfEjdr7orzds?e=budrUU) and place the downloaded datasets in the folder `datasets`. To use the YML profile we provide, keep the local folder `datasets` in the same directory tree as the OneDrive folder `datasets`.

| Task      | Dataset  | Relative Path                |
| --------- | -------- | ---------------------------- |
| SISR      | DF2K     | datasets/sr_data/DF2K        |
|           | Set5     | datasets/sr_data/Set5        |
|           | Set14    | datasets/sr_data/Set14       |
|           | BSD100   | datasets/sr_data/BSD100      |
|           | Urban100 | datasets/sr_data/Urban100    |
|           | Manga109 | datasets/sr_data/Manga109    |

>  All datasets have been processed in IMDB format and do not require any additional processing. The processing of the SISR dataset refers to the [BasicSR document](https://basicsr.readthedocs.io/en/latest/api/api_scripts.html), and the processing of the denoising dataset refers to the [NAFNet document](https://github.com/megvii-research/NAFNet/tree/main/docs).

>  To verify the integrity of your download, please refer to `docs/md5.txt`.

### 2 Run
#### 2.1 Train

```shell
python train.py -expe_opt options/repr/TEST/test.yml -task_opt options/task/LSR_x4.yml
python train.py -expe_opt options/repr/TEST/test.yml -task_opt options/task/LSR_x3.yml
```


>  Use the following demo command instead if you prefer to run in CPU mode:
>
> ```shell
> python train.py -expe_opt options/repr/ESWT/ESWT-24-6_LSR.yml -task_opt options/task/LSR_x4.yml --force_yml num_gpu=0
> ```

#### 2.2 Test


```shell
python test.py -expe_opt options/repr/TEST/test.yml -task_opt options/task/LSR_x4.yml
```

#### 2.3 Analyse

This function will analyze the complexity of a specified model on a specified task. Including the following metrics:

- **#Params**: total number of learnable parameters

- **#FLOPs**: abbreviation of floating point operations

- **#Acts**: number of elements of all outputs of convolutional layers

- **#Conv**: number of convolutional layers

- **#Memory**: maximum GPU memory consumption when inferring a dataset

- **#Ave. Time**: average inference time per image in a dataset

```shell
python analyse.py -expe_opt options/repr/TEST/test.yml -task_opt options/task/LSR_x4.yml
```


#### 2.4 Interpret

This function comes from the paper "Interpreting Super-Resolution Networks with Local Attribution Maps". When reconstructing the patches marked with red boxes, a higher DI indicates involving a larger range of contextual information, and a darker color indicates a higher degree of contribution.

```shell
python interpret.py -expe_opt options/repr/TEST/test.yml -task_opt options/task/LSR_x4.yml
```

#### 2.5 Infer

You can use this function to restore your own image.

```shell
python infer.py -expe_opt options/repr/TEST/test.yml -task_opt options/task/LSR_x4.yml
```

## Acknowledgements

This code is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [FriedRiceLab](https://github.com/Fried-Rice-Lab/FriedRiceLab). We thank its developers for creating such a useful toolbox.
