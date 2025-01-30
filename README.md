# ViMAEdit: Vision-guided and Mask-enhanced Adaptive Denoising for Prompt-based Image Editing
## 🚀 Getting Started
<span id="getting-started"></span>

### Environment Requirement 🌍
<span id="environment-requirement"></span>

```shell
conda create -n vima python=3.9
conda activate vima
pip install -r requirements.txt
```

### Benchmark Download ⬇️
<span id="benchmark-download"></span>

You can download the benchmark PIE-Bench (Prompt-driven Image Editing Benchmark) [here](https://forms.gle/hVMkTABb4uvZVjme9). The data structure should be like:

```python
|-- data
    |-- annotation_images
        |-- 0_random_140
            |-- 000000000000.jpg
            |-- 000000000001.jpg
            |-- ...
        |-- 1_change_object_80
            |-- 1_artificial
                |-- 1_animal
                        |-- 111000000000.jpg
                        |-- 111000000001.jpg
                        |-- ...
                |-- 2_human
                |-- 3_indoor
                |-- 4_outdoor
            |-- 2_natural
                |-- ...
        |-- ...
    |-- mapping_file.json # the mapping file of PIE-Bench, contains editing text, blended word, and mask annotation
```


## 🏃🏼 Running Scripts
<span id="running-scripts"></span>

### Inference and Evaluation📜
<span id="inference"></span>

**Run the Benchmark**

Run ViMAEdit on PIE-Bench:

```shell
python run_editing_vima.py --anno_file data/PIE-Bench_v1/mapping_file.json --image_dir data/PIE-Bench_v1/annotation_images --sd_model_dir runwayml/stable-diffusion-v1-5 --ip_adapter_dir h94/IP-Adapter --clip_model_dir laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

You can specify --output_dir for output path. 


**Run Any Image**

You can process your own images and editing prompts.
```shell
python run_editing_vima_one_image.py --image_path path/to/image --source_prompt "" --target_prompt "" --sd_model_dir runwayml/stable-diffusion-v1-5 --ip_adapter_dir h94/IP-Adapter --clip_model_dir laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

You can specify --output_path for output path. 

## 🥇 Quantitive Results

On PIE-Bench

| Method | Structure Distance | PSNR      | LPIPS     | MSE       | SSIM      | CLIP-Whole | CLIP-Edited |
|-------|--------------------|-----------|-----------|-----------|-----------|------------|-------------|
| P2P-Zero | 51.13              | 21.23     | 143.87    | 135.00    | 77.23     | 23.36      | 21.03       |
| MasaCtrl | 24.47              | 22.78     | 87.38     | 79.91     | 81.36     | 24.42      | 21.38       |
| PnP | 24.29              | 22.64     | 106.06    | 80.45     | 79.68     | 25.41      | 22.62       |
| P2P | **11.64**          | 27.19     | 54.44     | 33.15     | 84.71     | 25.03      | 22.13       |
| InfEdit | 13.78              | 28.51     | 47.58     | 32.09     | 85.66     | 25.03      | 22.22       |
| TurboEdit | -                  | 29.52     | 44.74     | 26.08     | 91.59     | 25.05      | 22.34       |
| ViMAEdit | 12.43              | 28.49     | **44.39** | 29.12     | 86.38     | 25.97      | 22.85       |
| ViMAEdit-XL | 12.06              | **29.65** | 48.75     | **24.50** | **92.30** | **25.98**  | **22.89**   |

## 🌟 Qualitative Results
![](assets/cases.png)

## 🤝🏼 Citation
If you make use of our work, please cite our paper:
```
@article{wang2024vision,
  title={Vision-guided and Mask-enhanced Adaptive Denoising for Prompt-based Image Editing},
  author={Wang, Kejie and Song, Xuemeng and Liu, Meng and Yuan, Jin and Guan, Weili},
  journal={arXiv preprint arXiv:2410.10496},
  year={2024}
}
```

## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified on the basis of [PnP Inversion](https://github.com/cure-lab/PnPInversion) and [InfEdit](https://github.com/sled-group/InfEdit), thanks to all the contributors!

