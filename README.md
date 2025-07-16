# slo_lr_detection
Detecting left and right eyes in SLO images.

## Example
Here is an example of using the main functions of this package:

### Part 1
Import the required libraries:
```
import cv2
from slo_lr_detection.slo_lr_detection import slo_ml_is_left
```

### Part 2
Specify the required paths and run the model:
```
img = cv2.imread("\path\to\image", 0)
model_path = "\path\to\model"
is_left = slo_ml_is_left(img, model_path, device="cpu")
```

# GCP Guide

This guide provides instructions for mounting Google Cloud Storage, setting up NVIDIA GPU support, and building and pushing a Docker image to Google Artifact Registry. 

## Mount Google Cloud Storage to Local Directory

To mount the Google Cloud Storage bucket named `pm-test4` to a local directory named `gcs`, use `gcsfuse` with flags for implicit directories, directory renaming limit, and maximum connections per host.

```
gcsfuse --implicit-dirs --rename-dir-limit=100 --max-conns-per-host=100 "pm-test4" ./gcs
```

## Install NVIDIA Container Toolkit on local Linux (No need for vertex AI pipelines, already there!)

To enable GPU support for Docker on a local Linux device, first update your package list. Then, install the NVIDIA Container Toolkit, reload the system daemon, and restart Docker to apply the changes.

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## Run Docker Container with GPU Support (No need for vertex AI pipelines, already there!)

With NVIDIA GPU support enabled, run the Docker container using the `--gpus all` flag, and mount the local `gcs` directory to `/gcs` within the container. The Docker image should be named `lr_image:latest`.

```
sudo docker run --gpus all -v /gcs:/gcs/ lr_image:latest
```

## Get the Active Project ID in Google Cloud 

Use Google Cloud SDK’s configuration command to retrieve the active project ID. This will confirm which Google Cloud project you are working in.

```
gcloud config list --format 'value(core.project)'
```

## Define Image URI for Docker (Important)

Set an environment variable named `IMAGE_URI` for your Docker image in Google Artifact Registry. Substitute `$LEFT_RIGHT_PROJECT_ID` with your project ID and `$REPO_NAME` with the repository name. The resulting format should be similar to:

`northamerica-northeast1-docker.pkg.dev/analytical-rain-397718/left-right-detection/lr_image:latest`

## Build the Docker Image

With the image URI set, build the Docker image using a Dockerfile located in the `docker` directory, tagging it with the `IMAGE_URI` specified in the previous step.

```
docker build -t "$IMAGE_URI" -f ./docker/Dockerfile .
```

## Authenticate Docker with Google Cloud Artifact Registry

Use `gcloud` to configure Docker authentication for `northamerica-northeast1-docker.pkg.dev`, allowing you to push the image to Artifact Registry.

```
gclout  auth configure-docker northamerica-northeast1-docker.pkg.dev
```

## Push Docker Image to Google Artifact Registry

Finally, push the Docker image to the registry using the `IMAGE_URI` variable set earlier.

```
docker push "$IMAGE_URI"
```

--- 

## Monitor Model Progress in MLFlow

Monitor the models and metrics.

```
poetry run mlflow ui --backend-store-uri /gcs/left-right-detection/checks/mlruns --port 5001
```

This concludes the setup and deployment process for this project. Ensure each step is followed in order for a successful deployment.
