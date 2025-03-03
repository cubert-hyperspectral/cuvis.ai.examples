# Simple anomaly detection example

## Introduction

In this example, an anomaly detection model is trained outside cuvis.ai and inference is done using cuvis.ai.
Since cuvis.ai was not yet ready to train models with, it had to be done separately.
We chose to go with the [EfficientAD](https://arxiv.org/pdf/2303.14535v3) model since it is an exiting state-of-the-art
model in anomaly detection.

In order to make it work with more than three channels the input layers of the model, as well as the given teacher
checkpoint, were adapted.

An ImageNet subset which is used in training can be
found [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz).
This dataset has to be adapted to six channels in order to be used in our training loop. We did this by duplicating the
three RGB channels to get a six channel image.

## The dataset

In our example dataset we used a custom build camera assembly which creates six channels cubes. Three channels are from
a 24 megapixel RGB-camera and the other three are SWIR channels with 1050nm, 1200nm and 1450nm wavelength respectively.

The dataset is precisely built for unsupervised multi spectral anomaly detection. It features (ENTER NUMBER HERE) images
which are divided into normal and anomalous images. We used sawdust in a wooden tray to create unique images on which
the model could lean. As anomalies, we used (ENTER ALL ANOMALIES HERE) as well as water to showcase the
use of our SWIR setup.

## How to train

After downloading the sample dataset and ImageNet dataset and extracting them into the data folder, the
``6_channel_data_generator.py`` must be run. This will automatically create a six channel version of ImageNet.

Now the train.py script can be run.

```
train.py -c example_train_config.yaml
```

The `example_train_config.yaml` has every parameter and path in it for the model and dataloader to work.
If you chose to alter the folder structure you may need to change some paths in there before the training is able to
run.

## How to predict data using cuvis.ai

A detailed description on how to infer cubes using our trained model (or any other model in that regard) can be found in
the `inference.ipynb` notebook.
