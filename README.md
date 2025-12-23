# segmentation-baseline

It includes six basic models (UNet, ResUNet, AttUNet, UNet++, U^2Net, Swin-UNETR), along with prediction code and metric evaluation code. The evaluation metrics comprise IOU (JC), Dice (DC), HD95, ASD, SSIM (structural similarity assessment), TVratio (structural smoothness assessment), and FOM (boundary assessment). Furthermore, the evaluation module can detect inaccurately annotated images in the dataset by identifying excessively large HD95 metrics.

## Quick Start
### Model Training

To train the models, first grant execution permission (if needed) and run the training script:

`chmod +x run_hard.sh`

`./run_hard.sh`

Most training parameters can be configured directly in the run_hard.shfile. The dataset path needs to be set in train.py.

### Model Prediction
For inference and automated evaluation:

`chmod +x run_predict.sh`

`/run_predict.sh`

Note: The run_predict.shscript includes the evaluation command at the end.

### Standalone Evaluation
To perform metric calculation separately:

`python metrics_calculate.py`
### Dataset Structure

DATASET/
│
├── imgs/               # Contains the original input training/validation images.
├── masks/              # Contains the corresponding ground truth masks for images in `imgs/`.
├── test/               # Contains the held-out test set images.
└── test_masks/         # Contains the ground truth masks for images in `test/`.

## Features
### Supported Models

* UNet: Classical encoder-decoder architecture for biomedical image segmentation

* ResUNet: UNet with residual connections for improved gradient flow

* AttUNet: Attention-guided UNet with attention gates

* UNet++: Nested skip-path connections for enhanced feature fusion

* U²Net: Nested U-structure with depth-wise supervision

* Swin-UNETR: Transformer-based architecture with shifted windows

### Evaluation Metrics

* IOU/Jaccard Coefficient (JC): Region similarity measurement

* Dice Coefficient (DC): Volume overlap evaluation

* HD95: 95th percentile Hausdorff Distance for boundary accuracy

* ASD: Average Symmetric Surface Distance

* SSIM: Structural Similarity Index Assessment

* TV ratio: Structural Smoothness Assessment

* FOM: Figure of Merit for boundary evaluation

The evaluation module can automatically detect inaccurately annotated images in the dataset by identifying abnormally large HD95 metrics.
