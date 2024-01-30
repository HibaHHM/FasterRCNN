import os

import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from utils import get_predictions, draw_boxes


def model(model_, x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat


def infer(image_url, model_, threshold=0.9, objects=None):
    image = Image.open(image_url)
    image = image.resize((int(0.5 * s) for s in image.size))
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)

    pred = model(model_, image.unsqueeze(0))
    predictions = get_predictions(pred, threshold, objects)
    img = draw_boxes(predictions, image)
    cv2.imwrite(image_url.split('.')[0] + '_pred.jpeg', img)


def load_model():
    model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model_.eval()

    for name, param in model_.named_parameters():
        param.requires_grad = False

    return model_


def main():
    model_ = load_model()
    images = ['images/' + file for file in os.listdir('images')]
    print(images)
    for image_url in images:
        print('Begin', image_url)
        infer(image_url, model_)
        print('Done')


if __name__ == '__main__':
    main()
