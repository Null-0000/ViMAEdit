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



## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified on the basis of [PnP Inversion](https://github.com/cure-lab/PnPInversion), [InfEdit](https://github.com/sled-group/InfEdit), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [StyleDiffusion](https://github.com/sen-mao/StyleDiffusion), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero) , [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play), [Edit Friendly DDPM Noise Space](https://github.com/inbarhub/DDPM_inversion), [Blended Latent Diffusion](https://github.com/omriav/blended-latent-diffusion), [Proximal Guidance](https://github.com/phymhan/prompt-to-prompt), [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix), thanks to all the contributors!

