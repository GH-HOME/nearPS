# Edge-preserving Near-light Photometric Stereo with Neural Surfaces

Code and data of CVPR submission 3367, **used for CVPR review only**.


## Dependencies
The proposed method is implemented in [PyTorch](https://pytorch.org/) with [SIREN network]((https://github.com/vsitzmann/siren)) as backbone
- Python 3.7
- PyTorch (version = 1.9.1)
- numpy
- scipy
- CUDA-9.0
- Pyvista
- Matplotlib
- opencv-python


## Overview
We provide:
- Datasets and results:
    - Real-captured near-light image data
    - Synthetic near-light image data with ground truth surface normal and depth
- Estimation results from existing methods and ours
- Implementing of our method
    - ``modules.py``: Network structure of our neural surface
    - ``loss_functions.py``: Reconstruction loss with albedo depending on the surface normal and depth
- Code to reproduce the experimental results shown in the paper
    - ``demo.py``


# Get started

- Download the data and result into the `data` folder and unzip
    - Synthetic data
    - Real data

- Check the data and released results from existing methods and ours, e.g.
    - synthetic_data
        - Buddha
            - render_img: record rendered image data
            - render_para: record GT surface normal, depth, 3D mesh, point light positions and radiant parameters
            - Released_Result: save recovered surface normal and depth, reconstructed 3D mesh

-
- Reproduce the experimental results shown in the paper
    ```
    python demo.py
    ```
    The shape estimation results from our method will be saved at
    - ``./data/synthetic_data/objectname/Result_Ours/datetime_cvpr22_submit_experimentname/Recoverd_Shapes``
    - ``./data/real_data/objectname/Result_Ours/datetime_cvpr22_submit_experimentname/Recoverd_Shapes``
