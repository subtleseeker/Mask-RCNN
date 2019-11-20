Mask R-CNN adapted for semantic segmentation of custom dataset for autonomous driving to augment Visual SLAM.

## Directory structure
ROOT_DIR = `Mask_RCNN/`
```
ROOT_DIR/auto
├── auto.py
├── dataset
│   ├── images
│   │   ├── predicted
│   │   ├── train
│   │   └── val
│   └── masks
├── inspect_balloon_data.ipynb
├── inspect_balloon_model.ipynb
└── train_colab.ipynb
```

## Steps to replicate
- Make a python3 virtual environment and install the required dependencies.
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
If there is an issue, then follow these steps:   
	- Use the provided tar `env` by extracting it in ROOT_DIR.   
	- Activate virtual environment from ROOT_DIR by executing `source env/bin/activate`. Find more info about virtual environment in [this](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) link.
- Download `mask_rcnn_coco.h5` in the ROOT_DIR from [this](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) link. If that link fails, open [this](https://github.com/matterport/Mask_RCNN/releases) link and download the file under `v2.0` tag.
- `cd ROOT_DIR/auto`
- Create the directory structure as above. Particularly, the `dataset` folder should be strictly leveled and named as above. The `predicted` directory is initially an empty.

## Training
Training using pretrained coco weights, execute the following command:
```
python3 auto.py train --dataset=./dataset/images --weights=coco
```

Other commands which could be used:
```
# Train a new model starting from pre-trained COCO weights
python3 auto.py train --dataset=/path/to/auto/dataset --weights=coco

# Resume training a model that you had trained earlier
python3 auto.py train --dataset=/path/to/auto/dataset --weights=last

# Train a new model starting from ImageNet weights
python3 auto.py train --dataset=/path/to/auto/dataset --weights=imagenet

# Apply color overlay to an image
python3 auto.py overlay --weights=/path/to/weights/file.h5 --image=<URL or path to file>

# Apply color overlay to video using the last weights you trained
python3 auto.py overlay --weights=last --video=<URL or path to file>
```

Logs will be stored in `ROOT_DIR/logs` for every epoch. Use the last `auto_xxxxxx.h5` file as the trained model weights to use for your dataset.

## Training over Colab
1. Login with the Gmail account.
2. Open `colab.research.google.com`.
3. Click on `Upload` tab.
4. Upload the `train_colab.ipynb` present in `ROOT_DIR/auto` .
5. Follow the instructions written in the notebook.

**Points to note**
- `dataset.tar.gz` is the compressed tar of `dataset` directory. Thus it will follow the same directory structure as mentioned above.
- Download `mask_rcnn_coco.h5` from [this](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) link. If this link fails, open [this](https://github.com/matterport/Mask_RCNN/releases) link and download the file under `v2.0` tag.

## Overlay
Overlays a mask on a single image or video, and stores in `$PWD`.
```
python3 auto.py overlay --weights=<.h5 weights path> --image=<path to image>
```

## Evaluation
Evaluates a set of images present in `<dataset path>/val` and stores the predicted results in `<dataset path>/predicted` .   
Also finds stats like Average IOU, per-class IOU, time per image, etc.
```
python3 auto.py evaluate --dataset=<dataset path> --weights=<.h5 weights path>
```

## Adding/Removing class
Follow the `#classchange` comment, using `Find` (Ctrl/Cmd + F) in the file `auto.py` and change the lines to fulfill custom requirements.