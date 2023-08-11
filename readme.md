# **MAP-monaiclassify**

Lung Nodule Detection module that can be packaged as Monai Application Package. It can be run in isolation and can also be incorporated into Mercure workflow.

<br>

This module is available as a Monai Application Package docker image that can be added to an existing Mercure installation using the docker tag : *mercureimaging/map-monaiclassify*. It will perform lung nodule detection in CT images using the [Lung nodule_ct detection](https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/lung_nodule_ct_detection_v0.5.9.zip) MONAI bundle.

<br>

The code can be modified to deploy other detection models in the MONAI model zoo.

<br>

# Installation

## Add module to existing mercure installation
Follow instructions on [mercure website](https://mercure-imaging.org) on how to add a new module. Use the docker tag *mercureimaging/map-monaiclassify*.

<br>

## Build module for local testing, modification, and development
1. Clone repo.
2. Install the packages and dependencies using `pip install -r requirements.txt`
3. Run the app locally using: `monai-deploy exec lung_app -i <input_path> -o <output_path> -m ./model/lung_model.ts`
4. You can make changes and package this application as your own using MAP. Packaging documentation can be found [here](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/developing_with_sdk/packaging_app.html).

<br>

# Output

 Lung nodule count and raw data are written to a specified output directory:
- `result.json` has the number of nodules count.
- `output_raw.json` has the bounding boxes for the nodules and scores for the predictions.
