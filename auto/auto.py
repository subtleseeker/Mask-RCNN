"""
Mask R-CNN
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

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
"""

import os
import sys
import datetime

import enum
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from math import isnan

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

#classchange
class ClassName(enum.Enum):
    lane = 1
    pedestrian = 2
    vehicle = 3
    sign_board = 4
    street_light = 5

class AutoConfig(Config):
    """Configuration for training on the auto dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "auto"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + 5 custom classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    global dataset_dir


############################################################
#  Dataset
############################################################

class AutoDataset(utils.Dataset):

    def load_auto(self, dataset_dir, subset):
        """Load a subset of the Auto dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. #classchange
        self.add_class("auto", 1, "lane")
        self.add_class("auto", 2, "pedestrian")
        self.add_class("auto", 3, "vehicle")
        self.add_class("auto", 4, "sign_board")
        self.add_class("auto", 5, "street_light")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        images = os.listdir(dataset_dir)

        for i in images:
            if i == ".directory":
                continue
            image_path = os.path.join(dataset_dir, i)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source='auto',
                image_id=i,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        mask_dir = os.path.join(dataset_dir, "../masks")
        image_path = os.path.join(mask_dir, str(info["id"]))
        image = skimage.io.imread(image_path)

        #classchange :start:
        lane = np.all(image == (0, 255, 0), axis=-1)
        pedestrian = np.all(image == (255, 0, 255), axis=-1)
        vehicle = np.all(image == (0, 255, 255), axis=-1)
        sign_board = np.all(image == (255, 0, 0), axis=-1)
        street_light = np.all(image == (255, 255, 0), axis=-1)

        mask = np.stack((lane, pedestrian, vehicle, sign_board, street_light), axis=2).astype(np.bool)
        class_ids = np.arange(1, 6).astype(np.int32)    #classchange (includes background)
        #classchange :end:
        return mask, class_ids


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = AutoDataset()
    dataset_train.load_auto(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AutoDataset()
    dataset_val.load_auto(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_overlay(image, mask, class_ids):
    """Apply color overlay.
    image: RGB image [height, width, 3]
    mask: segmentation mask [height, width, #classes]
    Returns result image.
    """
    overlayed = image
    print("Found classes: ", [ClassName(class_id).name for class_id in class_ids])
    if mask.shape[-1] > 0:
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            m = np.stack((m, m, m), axis=2)
            # classchange
            if class_ids[i] == 1:
                overlayed = np.where(m, (115, 255, 115), overlayed).astype(np.uint8)
            elif class_ids[i] == 2:
                overlayed = np.where(m, (255, 115, 255), overlayed).astype(np.uint8)
            elif class_ids[i] == 3:
                overlayed = np.where(m, (115, 255, 255), overlayed).astype(np.uint8)
            elif class_ids[i] == 4:
                overlayed = np.where(m, (255, 115, 115), overlayed).astype(np.uint8)
            elif class_ids[i] == 5:
                overlayed = np.where(m, (255, 255, 115), overlayed).astype(np.uint8)
    else:
        overlayed = overlayed.astype(np.uint8)
    return overlayed


def detect_and_overlay(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color overlay
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Convert grayscale images to 3D
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=2)

        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Color overlay
        overlay = color_overlay(image, r['masks'], r['class_ids'])
        # Save output
        file_name = "overlay_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, overlay)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "overlay_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color overlay
                overlay = color_overlay(image, r['masks'], r['class_ids'])
                # RGB -> BGR to save image to video
                overlay = overlay[..., ::-1]
                # Add image to video writer
                vwriter.write(overlay)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Evaluate
############################################################

def get_mask(image, mask, class_ids):
    """Apply color overlay.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    overlay = image
    pd_mask = np.zeros([overlay.shape[0], overlay.shape[1], 5], dtype=np.uint8) #classchange

    if mask.shape[-1] > 0:
        for i in range(mask.shape[-1]):
            m = mask[:, :, i:i+1]
            pd_mask[:,:,class_ids[i]-1:class_ids[i]] = np.where(m, True, pd_mask[:,:,class_ids[i]-1:class_ids[i]]).astype(np.uint8)

    ############## For visualizing mask ##############
    # pd_mask = np.zeros([overlay.shape[0], overlay.shape[1], 3], dtype=np.uint8)
    # if mask.shape[-1] > 0:
    #     for i in range(mask.shape[-1]):
    #         m = mask[:, :, i]
    #         m = np.stack((m, m, m), axis=2)
    #         #classchange
    #         if class_ids[i] == 1:
    #             pd_mask = np.where(m, (0, 255, 0), pd_mask).astype(np.uint8)
    #         elif class_ids[i] == 2:
    #             pd_mask = np.where(m, (255, 0, 255), pd_mask).astype(np.uint8)
    #         elif class_ids[i] == 3:
    #             pd_mask = np.where(m, (0, 255, 255), pd_mask).astype(np.uint8)
    #         elif class_ids[i] == 4:
    #             pd_mask = np.where(m, (255, 0, 0), pd_mask).astype(np.uint8)
    #         elif class_ids[i] == 5:
    #             pd_mask = np.where(m, (255, 255, 0), pd_mask).astype(np.uint8)
    #################################################
    return pd_mask


def evaluate(model, dataset, limit=0, image_ids=None):
    """Evaluates a set of data for IOU scores.
    dataset: A Dataset object with validation data
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    import time
    t_prediction = 0
    t_start = time.time()

    results = []
    total_iou_score = 0
    total_class_iou = np.zeros(5)   #classchange
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        gt_mask, class_ids = dataset.load_mask(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        ################ Save predicted images ##############
        # Color overlay
        overlay = color_overlay(image, r['masks'], r['class_ids'])
        # Save output
        file_name = "overlay_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave("dataset/images/predicted/" + file_name, overlay)
        #####################################################

        pd_mask = get_mask(image, r['masks'], r['class_ids'])

        intersection = np.logical_and(gt_mask, pd_mask)
        union = np.logical_or(gt_mask, pd_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        total_iou_score += iou_score

        class_iou = np.zeros(5) #classchange
        for j in range(5):  #classchange
            inter = np.logical_and(gt_mask[:,:,j], pd_mask[:,:,j])
            un = np.logical_or(gt_mask[:,:,j], pd_mask[:,:,j])
            class_iou[j] = np.sum(inter) / np.sum(un)
            if not isnan(class_iou[j]):
                total_class_iou[j] += class_iou[j]

        class_names = [ClassName(class_id).name for class_id in class_ids]
        print(f"Class IOU scores")
        for j in range(5):  #classchange
            print(class_names[j].ljust(14) + ": " + str(class_iou[j]))

        print(f"IOU score for {image_id} = {iou_score}")
        print("".ljust(50,'-'))
        results.extend((image_id, iou_score))

    print("IOUs = ", results)
    print()
    print("".ljust(50,'-'))

    class_names = [ClassName(class_id).name for class_id in class_ids]
    print(f"Average Class IOU scores")
    for j in range(5):  #classchange
        print(class_names[j].ljust(14) + ": " + str((total_class_iou[j]/len(image_ids))))

    print(f"------ Average IOU score = {total_iou_score/len(image_ids)} ------\n".ljust(50,'-'))

    print("Prediction time: {}. \nAverage time: {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect classes for autonomous driving.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'overlay'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/auto/dataset/",
                        help='Directory of the required dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color overlay on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color overlay on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "overlay":
        assert args.image or args.video, \
            "Provide --image or --video to apply color overlay"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    dataset_dir = args.dataset

    # Configurations
    if args.command == "train":
        config = AutoConfig()
    else:
        class InferenceConfig(AutoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "overlay":
        detect_and_overlay(model, image_path=args.image,
                           video_path=args.video)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = AutoDataset()
        dataset_val.load_auto(args.dataset, "val")
        dataset_val.prepare()

        # print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate(model, dataset_val)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'overlay'".format(args.command))