# Function to receive predictions and output class name, set a threshold to eliminate
import numpy as np
import cv2
import matplotlib.pyplot as plt

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_predictions(preds, threshold, objects=None):
    predictions = [(COCO_INSTANCE_CATEGORY_NAMES[i], p, [(b[0], b[1]), (b[2], b[3])]) for i, p, b in
                   zip(preds[0]['labels'], preds[0]['scores'], preds[0]['boxes'])]
    predictions = [tup for tup in predictions if tup[1] >= threshold]

    if objects and predictions:
        predictions = [tup for tup in predictions if tup[0] in objects]
    return predictions


# Function to draw bounding boxes. Receive bounding box and class details, display properties
def draw_boxes(predictions, image):
    img = (np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0,
                   1) * 255).astype(np.uint8).copy()

    for prediction in predictions:
        label = prediction[0]
        score = prediction[1].item()
        box = prediction[2]
        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])
        # print(score, type(score), x1, y1, x2, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, label + ':' + str(round(score, 2), ), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img

# Function to free memory
# def save_RAM(image_=False):
#     global image, img, pred
#     torch.cuda.empty_cache()
#     del img
#     del pred
#     if image_:
#         image.close()
#         del image
