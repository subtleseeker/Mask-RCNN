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

    # Apply color splash to an image
    python3 auto.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 auto.py splash --weights=last --video=<URL or path to file>
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
        # Add classes. We have only one class to add.
        self.add_class("auto", 1, "lane")
        self.add_class("auto", 2, "pedestrian")
        self.add_class("auto", 3, "vehicle")
        self.add_class("auto", 4, "sign_board")
        self.add_class("auto", 5, "street_light")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # annotations = list(annotations.values())  # don't need the dict keys
        #
        # # The VIA tool saves images in the JSON even if they don't have any
        # # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]
        #
        # # Add images
        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. These are stores in the
        #     # shape_attributes (see json format above)
        #     # The if condition is needed to support VIA versions 1.x and 2.x.
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']]

        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only managable since the dataset is tiny.

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
                # polygons=polygons
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a auto dataset image, delegate to parent class.
        # image_info = self.image_info[image_id]
        # if image_info["source"] != "auto":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # print(info){'id': '01300.png', 'source': 'auto', 'path': './dataset/images/val/01300.png', 'width': 1280, 'height': 720}

        mask_dir = os.path.join(dataset_dir, "../masks")
        image_path = os.path.join(mask_dir, str(info["id"]))
        # print("\n$$$$$$$$$$$", image_path)
        image = skimage.io.imread(image_path)
        # height, width = image.shape[:2]

        lane = np.all(image == (0, 255, 0), axis=-1)
        pedestrian = np.all(image == (255, 0, 255), axis=-1)
        vehicle = np.all(image == (0, 255, 255), axis=-1)
        sign_board = np.all(image == (255, 0, 0), axis=-1)
        street_light = np.all(image == (255, 255, 0), axis=-1)

        # mask = lane | pedestrian | street_light | sign_board | vehicle
        # # mask = mask.reshape(info["height"], info["width"], 1).astype(bool)
        # mask = np.stack((mask, mask, mask), axis=2).astype(bool)
        mask = np.stack((lane, pedestrian, vehicle, sign_board, street_light), axis=2).astype(np.bool)

        # class_ids = np.zeros([info["height"], info["width"]], dtype=np.uint8)
        # class_ids = np.where(street_light == True, 5, class_ids)
        # class_ids = np.where(sign_board == True, 4, class_ids)
        # class_ids = np.where(vehicle == True, 3, class_ids)
        # class_ids = np.where(pedestrian == True, 2, class_ids)
        # class_ids = np.where(lane == True, 1, class_ids)
        # class_ids = class_ids.astype(np.int32)

        class_ids = np.arange(1, 6).astype(np.int32)

        # print("\n%%%%%%%%%%%%%%%", mask.shape, class_ids.shape, image.shape)

        return mask, class_ids

        #
        # mask = np.zeros([info["height"], info["width"], 1],
        #                 dtype=np.uint8)
        # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1
        #
        # # Return mask, and array of class IDs of each instance. Since we have
        # # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    # def image_reference(self, image_id):
    #     """Return the path of the image."""
    #     info = self.image_info[image_id]
    #     if info["source"] == "auto":
    #         return info["path"]
    #     else:
    #         super(self.__class__, self).image_reference(image_id)


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


def color_splash(image, mask, class_ids):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    splash = image
    # Copy color pixels from the original color image where mask is set
    # print("^^^^^^^^^pd_mask.shape", mask.shape)

    print("Found classes: ", [ClassName(class_id).name for class_id in class_ids])
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        # mask = (np.sum(mask, -1, keepdims=True) >= 1)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            m = np.stack((m, m, m), axis=2)
            if class_ids[i] == 1:
                splash = np.where(m, (115, 255, 115), splash).astype(np.uint8)
            elif class_ids[i] == 2:
                splash = np.where(m, (255, 115, 255), splash).astype(np.uint8)
            elif class_ids[i] == 3:
                splash = np.where(m, (115, 255, 255), splash).astype(np.uint8)
            elif class_ids[i] == 4:
                splash = np.where(m, (255, 115, 115), splash).astype(np.uint8)
            elif class_ids[i] == 5:
                splash = np.where(m, (255, 255, 115), splash).astype(np.uint8)
    else:
        splash = splash.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=2)

        print("%%%%%%\n\n\n%%%%%%%", image.shape)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print("$$$$$\n\n\n$$$$$", r)

        # Color splash
        splash = color_splash(image, r['masks'], r['class_ids'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
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
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Evaluate
############################################################

def get_mask(image, mask, class_ids):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    splash = image
    # Copy color pixels from the original color image where mask is set
    # print("^^^^^^^^^(detected)mask.shape, class_ids.shape = ", mask.shape, class_ids.shape)

    pd_mask = np.zeros([splash.shape[0], splash.shape[1], 5], dtype=np.uint8)

    if mask.shape[-1] > 0:
        for i in range(mask.shape[-1]):
            m = mask[:, :, i:i+1]
            # m = np.stack((m, m, m), axis=2)
            # m = mask.reshape(m.shape[0], m.shape[1], 1)
            # print("yeyeyeye", m.shape, pd_mask[:,:,i:i+1].shape)
            # if class_ids[i] == 1:
            pd_mask[:,:,class_ids[i]-1:class_ids[i]] = np.where(m, True, pd_mask[:,:,class_ids[i]-1:class_ids[i]]).astype(np.uint8)
            # elif class_ids[i] == 2:
            #     pd_mask[:,:,i:i+1] = np.where(m, True, pd_mask[:,:,i:i+1]).astype(np.uint8)
            # elif class_ids[i] == 3:
            #     pd_mask[:,:,i:i+1] = np.where(m, True, pd_mask[:,:,i:i+1]).astype(np.uint8)
            # elif class_ids[i] == 4:
            #     pd_mask[:,:,i:i+1] = np.where(m, True, pd_mask[:,:,i:i+1]).astype(np.uint8)
            # elif class_ids[i] == 5:
            #     pd_mask[:,:,i:i+1] = np.where(m, True, pd_mask[:,:,i:i+1]).astype(np.uint8)

    ############## For visualizing mask ##############
    # pd_mask = np.zeros([splash.shape[0], splash.shape[1], 3], dtype=np.uint8)
    # if mask.shape[-1] > 0:
    #     for i in range(mask.shape[-1]):
    #         m = mask[:, :, i]
    #         m = np.stack((m, m, m), axis=2)
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
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    import time
    t_prediction = 0
    t_start = time.time()

    results = []
    total_iou_score = 0
    total_class_iou = np.zeros(5)
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        gt_mask, class_ids = dataset.load_mask(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        ################ Save predicted images ##############
        # Color splash
        splash = color_splash(image, r['masks'], r['class_ids'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave("dataset/images/predicted/" + file_name, splash)
        #####################################################

        pd_mask = get_mask(image, r['masks'], r['class_ids'])
        # print("\n\n\n555555dt", pd_mask.shape)
        # print("\n\n\n666666gt", gt_mask.shape)

        intersection = np.logical_and(gt_mask, pd_mask)
        union = np.logical_or(gt_mask, pd_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        total_iou_score += iou_score

        class_iou = np.zeros(5)
        for j in range(5):
            inter = np.logical_and(gt_mask[:,:,j], pd_mask[:,:,j])
            un = np.logical_or(gt_mask[:,:,j], pd_mask[:,:,j])
            class_iou[j] = np.sum(inter) / np.sum(un)
            if not isnan(class_iou[j]):
                total_class_iou[j] += class_iou[j]

        class_names = [ClassName(class_id).name for class_id in class_ids]
        print(f"Class IOU scores")
        for j in range(5):
            print(class_names[j].ljust(14) + ": " + str(class_iou[j]))

        print(f"IOU score for {image_id} = {iou_score}")
        print("".ljust(50,'-'))

        # # Convert results to COCO format
        # # Cast masks to uint8 because COCO tools errors out on bool
        # image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
        #                                    r["rois"], r["class_ids"],
        #                                    r["scores"],
        #                                    r["masks"].astype(np.uint8))
        results.extend((image_id, iou_score))

    print("IOUs = ", results)
    print()
    print("".ljust(50,'-'))

    class_names = [ClassName(class_id).name for class_id in class_ids]
    print(f"Average Class IOU scores")
    for j in range(5):
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
                        help="'train' or 'splash'")
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
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

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
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
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
              "Use 'train' or 'splash'".format(args.command))


# def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
#     """Arrange resutls to match COCO specs in http://cocodataset.org/#format
#     """
#     # If no results, return an empty list
#     if rois is None:
#         return []
#
#     results = []
#     for image_id in image_ids:
#         # Loop through detections
#         for i in range(rois.shape[0]):
#             class_id = class_ids[i]
#             score = scores[i]
#             bbox = np.around(rois[i], 1)
#             mask = masks[:, :, i]
#
#             result = {
#                 "image_id": image_id,
#                 "category_id": dataset.get_source_class_id(class_id, "auto"),
#                 "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                 "score": score,
#                 "segmentation": maskUtils.encode(np.asfortranarray(mask))
#             }
#             results.append(result)
#     return results

