
## <center> Underwater Organism Color Enhancement via Color Code Decomposition, Adaptation and Interpolation

<center> <A HREF="https://xiaofeng-life.github.io/">Xiaofeng Cong</A><sup>+</sup>, 
Jing Zhang<sup>+</sup>, Senior Member, IEEE, 
Yeying Jin, Junming Hou, Yu Zhao, 
Jie Gui<sup>*</sup>, Senior Member, IEEE, 
James Tin-Yau Kwok, Fellow, IEEE, 
Yuan Yan Tang, Life Fellow, IEEE

(<sup>+</sup> equal contributions, * corresponding author)
</center>


🐯 **Abstract**: Underwater images often suffer from quality degradation due to absorption and scattering effects.
Most existing underwater image enhancement algorithms produce a single, fixed-color image, limiting user flexibility and application.
To address this limitation, we propose a method called **ColorCode**, which enhances underwater images while offering a range of controllable color outputs.
Our approach involves recovering an underwater image to a reference enhanced image through supervised training and decomposing it into color and content codes via self-reconstruction and cross-reconstruction. 
The color code is explicitly constrained to follow a Gaussian distribution, allowing for efficient sampling and interpolation during inference.
ColorCode offers three key features: 1) **color enhancement**, producing an enhanced image with a fixed color
; 2) **color adaptation**, enabling controllable adjustments of long-wavelength color components using guidance images
; and 3) **color interpolation**, allowing for the smooth generation of multiple colors through continuous sampling of the color code.
Quantitative and visual evaluations on popular and challenging benchmark datasets demonstrate the superiority of ColorCode over existing methods in providing diverse, controllable, and color-realistic enhancement results.

<figure>
<div align="center">
<img src=figures/motivation.png width="50%">
</div>
</figure>

<div align='center'>

**Fig 1. Motivation of this paper.**
</div>
<br>

<!-- 这是一个注释 -->

## 🐳 Color Enhancement

Color enhancement, producing an enhanced image with a fixed color as follows.

<figure>
<div align="center">
<img src=figures/visual_comparison.png width="100%">
</div>
</figure>

<div align='center'>

**Fig 2. Visual results obtained by various UIE algorithms on UIEB dataset.**
</div>
<br>


<!-- 这是一个注释 -->

## 🐳 Color Adaptation

Color adaptation, enabling controllable adjustments of long-wavelength color components using guidance images as follows.

<figure>
<div align="center">
<img src=figures/color_adaption.png width="100%">
</div>
</figure>

<div align='center'>

**Fig 3. Visualization of ColorCode's color adaptation capability.**
</div>
<br>


<!-- 这是一个注释 -->

## 🐳 Color Interpolation

Color interpolation, allowing for the smooth generation of multiple colors through continuous sampling of the color code as follows.

<figure>
<div align="center">
<img src=figures/more_interpolation_results_1.png width="100%">
</div>
</figure>

<div align='center'>

**Fig 4. Diversification results obtained by using the color interpolation function.**
</div>
<br>

<!-- 这是一个注释 -->

## 🍈 The training and inference processes of ColorCode
The overall pipeline of the proposed ColorCode is shown in Fig. 5, 
which consists of four main processes, namely (i) a supervised training process P1 to learn the deterministic enhanced image, 
(ii) a decomposition process P2 to decompose color and content, 
(iii) a color adaptation process P3 by guidance images with long-wavelength colors, 
and (iv) an interpolating process P4 to obtain underwater organisms with diverse colors by continuous color codes.

<figure>
<div align="center">
<img src=figures/overall_pipeline.png width="100%">
</div>
</figure>

<div align='center'>

**Fig 5. The overall training and inference process of the proposed ColorCode.**
</div>
<br>


<!-- 这是一个注释 -->

## 🍓 The difference between ColorCode and StyleCode

The primary aim of the underwater image enhancement network is to correct distorted colors.
Thus, the color code should directly facilitate enhancement results that closely match reference images. 
This distinguishes the color code from the style code used in style transfer. 
As illustrated in Fig. 6, the color code should align closely with the reference image, whereas the style code does not share this property.

<figure>
<div align="center">
<img src=figures/difference_with_color_code.png width="60%">
</div>
</figure>

<div align='center'>

**Fig 6. Comparisons of results obtained by style code and color code. 
The color blocks represent the main colors of the organisms in the  images. 
The acquisition of color block is achieved by calculating the center of mass of the organism and the surrounding main color.**
</div>
<br>

## 🧨 How to train and test

### 1. Data preparation for Enhancement Task
Download underwater imagery datasets from websites or papers. Follow the organization form below.
```
├── dataset_name
    ├── train
        ├── images
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── labels
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── val
        ├── images
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── labels
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── test
        ├── images
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── labels (if you have)
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── masks (if you have)
            ├── im1.jpg
            ├── im2.jpg
            └── ...
```

### 2. Data preparation color adaption and color interpolation
Note: If you only need to perform color enhancement, you can ignore this step.

**Use the EUVP-scenes if you want fine-tuning color ability!!!**
**The diversity of UIEB and EUVP-imagenet are not enough. 
Besides, the images with less distortion are required. Therefore, the EUVP-scenes is needed !!!**


### 3. Training for color enhancement
Put the config file in folder task_CECF/configs. For example task_CECF/configs/UIEB_3090_dim8_1m_bs6_NoTransBlock_SSIM.yaml

The batch_size must be greater than or equal 4.

```
cd task_ColorCode
python train_CECFPlus_SegDeepLabPretrain_NoMMD.py --config configs/differ_dataset/UIEB_3090_dim8_1m_bs4_Res6_SSIM_GauMean0p5.yaml \
                                                  --device cuda:0 --res_dir ../results/MyCECFPlus/train_CECFPlus_SegDeepLabPretrain_NoMMD/ \
                                                  --grad_acc 1
```


### 4. Training for color adaption and color interpolation

```
cd task_ColorCode 
python train_CECFPlus_SegDeepLabPretrain_MMD.py --config configs/differ_GauMean/UFO120reslected_3090_dim8_1m_bs4_NoTransBlock_SSIM_GauMean0.yaml \
                                                --device cuda:0 --res_dir ../results/MyCECFPlus/train_CECFPlus_SegDeepLabPretrain_MMD/ \
                                                --grad_acc 1 
```



### 5. Test for color enhancement, color adaptation and color interpolation

All pretrained models are place at: https://drive.google.com/drive/folders/1SggrFa6KvSy91OY-qYQZVsymdXMsrsrS?usp=sharing

First, download "DeepLabV3.pth" and put it into "results/MyCECFPlus/".

Then, download the weight files from then put them in "pretrained_models" outside the project. 
+ UIEB_color_enhancement.pt 
+ UFO120_color_adaption.pt
+ UFO120_color_interpolation.pt

Then, run the following test code,

```
cd task_ColorCode 
python test_enhancement.py --config configs/differ_dataset/UIEB_3090_dim8_1m_bs4_Res6_SSIM_GauMean0p5.yaml \
                    --input_folder ../demo_dataset/UIEB/test/images/ \
                    --output_folder ../results/MyCECFPlus/UIEB/ \
                    --checkpoint ../../pretrained_models/UIEB_color_enhancement.pt \
                    --device cuda:0 
```

```
cd task_ColorCode 
python test_adaptation.py --config configs/UFO120reslected_3090_dim8_1m_bs4_NoTransBlock_SSIM.yaml \
                    --input_folder ../results/MyCECFPlus/inputs/UFO120_blur_test/ \
                    --guide_path ../results/MyCECFPlus/inputs/guide_natural_2/ \
                    --output_folder ../results/MyCECFPlus/plot_for_paper/UFO120_natural_test/ \
                    --checkpoint  ../../pretrained_models/UFO120_color_adaption.pt \
                    --subfolder_prefix UFO120reslected_natural_test_ \
                    --device cuda:0 \
                    --have_mask Yes
```

```
cd task_ColorCode 
python test_CECFPlus_Interpolation_SingleImage_Dim2.py --config configs/differ_GauMean/UFO120reslected_3090_dim2_1m_bs4_NoTransBlock_SSIM_GauMean0.yaml \
                    --input_folder ../results/MyCECFPlus/inputs/UFO120_blur_test/ \
                    --guide_path ../results/MyCECFPlus/inputs/guide_natural_2/ \
                    --output_folder ../results/MyCECFPlus/plot_for_paper/UFO120_natural_test_Guassian_SingleImage_Dim2/ \
                    --checkpoint ../../pretrained_models/UFO120_color_interpolation.pt \
                    --subfolder_prefix UFO120reslected_ \
                    --device cuda:0 \
                    --have_mask Yes \
                    --img_name set_u38.jpg
```