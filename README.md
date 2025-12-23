# segmentation-baseline
# Overview
It includes six basic models (UNet, ResUNet, AttUNet, UNet++, U^2Net, Swin-UNETR), along with prediction code and metric evaluation code.  Furthermore, the evaluation module can detect inaccurately annotated images in the dataset by identifying excessively large HD95 metrics.

The model training module is designed to be executed through the run_hard.shshell script. This script handles the main training pipeline, while specific parameters can be configured in different files as described below.
## Basic Execution
Grant execution permission​ to the script (required for the first time):

·chmod +x run_hard.sh·

This command makes the script executable.

Run the training module:

·./run_hard.sh·
