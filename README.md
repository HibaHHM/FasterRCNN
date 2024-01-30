**Project Title**

Object detection with Faster R-CNN and PyTorch

**Project Description**

This project classifies entities in an image using Faster R-CNN model from torchvision.

**Steps**

Step 1: Install necessary libraries
    `pip install -r requirements.txt`

Step 2: Import necessary libraries 
* Getting Images from Web 
* Image Processing and Visualization

Step 3: Define Util Functions
* Processing and filtering predictions based on threshold
* Drawing bounding boxes

Step 4: Load Pretrained Faster-RCNN Model from torchvision

    `model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)`

Step 5: Inference Method
* Load Image
* Apply transforms
* Pass the Image to model, obtain outputs
    * For each image, we have {'boxes': Tensor([[],[],[],[],..]),
                                'labels': Tensor([a,b,c,d,..]),
                                'scores': Tensor([s1,s2,s3,s4,..])}
* Process predictions
* Draw bounding boxes
* Save output image



