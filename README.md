<h1 align="center">● Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

Medical SAM 2, or say MedSAM-2, is an advanced segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address both 2D and 3D medical
image segmentation tasks. This method is elaborated on the paper [Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2](https://arxiv.org/abs/2408.00874) and [Medical SAM 2 Webpage](https://supermedintel.github.io/Medical-SAM2/).

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="350" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/framework.png"></div>
 
## 🩻 3D Abdomen Segmentation Visualisation
 <div align="center"><img width="420" height="420" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/example.gif"></div>

## Pre-trained weight

We released our pretrain weight [here](https://huggingface.co/jiayuanz3/MedSAM2_pretrain/tree/main)

## 🧐 Requirement

### GPU Installation (Original)
 Install the environment with CUDA support:

 ``conda env create -f environment.yml``

 ``conda activate medsam2``

### CPU-Only Installation (No NVIDIA GPU required)
 Install the CPU-only environment for systems without NVIDIA graphics cards:

 ``conda env create -f environment_cpu.yml``

 ``conda activate medsam2_cpu``

 You can download SAM2 checkpoint from checkpoints folder:
 
 ``bash download_ckpts.sh``

 **Note:** The system will automatically detect whether GPU is available and use CPU if no CUDA-compatible GPU is found. CPU training will be significantly slower but allows running on any hardware.

 Further Note: We tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Ubuntu 22.04
Conda Version: 23.7.4
Python Version: 3.12.4
```

 ## 🎯 Example Cases
 #### Download REFUGE or BCTV or your own dataset and put in the ``data`` folder, create the folder if it does not exist ⚒️
 
 ### 2D case - REFUGE Optic-cup Segmentation from Fundus Images

**Step1:** Download pre-processed [REFUGE](https://refuge.grand-challenge.org/) dataset manually from [here](https://huggingface.co/datasets/jiayuanz3/REFUGE/tree/main), or using command lines:

 ``wget https://huggingface.co/datasets/jiayuanz3/REFUGE/resolve/main/REFUGE.zip``

 ``unzip REFUGE.zip``

 **Step2:** Run the training and validation by:

**For GPU systems:**
``python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2 -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE``

**For CPU-only systems (no NVIDIA GPU):**
``python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_CPU -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 1 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE -gpu False``

*Note: CPU training uses smaller batch size (-b 1) due to memory constraints and will be significantly slower.*

 ### 3D case - Abdominal Multiple Organs Segmentation
 
 **Step1:** Download pre-processed [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) dataset manually from [here](https://huggingface.co/datasets/jiayuanz3/btcv/tree/main), or using command lines:

 ``wget https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip``

 ``unzip btcv.zip``

**Step2:** Run the training and validation by:

**For GPU systems:**
 ``python train_3d.py -net sam2 -exp_name BTCV_MedSAM2 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv``

**For CPU-only systems (no NVIDIA GPU):**
 ``python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_CPU -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv -gpu False``

*Note: CPU training uses smaller image size (-image_size 512) and will be significantly slower than GPU training.*

## 💻 CPU-Only Support

Medical SAM2 now supports CPU-only execution for systems without NVIDIA graphics cards:

- **Automatic Device Detection**: The system automatically detects GPU availability and falls back to CPU if needed
- **CPU-Optimized Environment**: Use `environment_cpu.yml` for installation without CUDA dependencies
- **Memory Efficient**: Automatically adjusts batch sizes and precision for CPU constraints
- **Cross-Platform**: Works on any system with sufficient RAM (8GB+ recommended)

**Performance Notes:**
- CPU training is 10-50x slower than GPU training depending on the model size
- Recommended to use smaller batch sizes (1-2) and image sizes (512-768) for CPU
- Consider using pre-trained models and fine-tuning rather than training from scratch

## 🚨 News
- 24-12-04. Our Medical SAM 2 paper was updated on Arxiv with new insights and results
- 24-08-05. Our Medical SAM 2 paper **ranked #1 Paper of the day** collected by AK on Hugging Face 🤗
- 24-08-05. Update 3D example details and pre-processed BTCV dataset download link 🔗
- 24-08-05. Update 2D example details and pre-processed REFUGE dataset download link 🔗
- 24-08-05. Our Medical SAM 2 paper was available online 🥳
- 24-08-05. Our Medical SAM 2 code was available on Github 🥳
- 24-07-30. The SAM 2 model was released 🤩

## 📝 Cite
 ~~~
@misc{zhu2024medical,
    title={Medical SAM 2: Segment medical images as video via Segment Anything Model 2},
    author={Jiayuan Zhu and Abdullah Hamdi and Yunli Qi and Yueming Jin and Junde Wu},
    year={2024},
    eprint={2408.00874},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
 ~~~
