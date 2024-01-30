# FasterRCNN
Object detection with Faster R-CNN and PyTorch

Apply Object detection with Faster R-CNN to classify predetermined objects

Step 1: Install necessary libraries
Step 2: Import necessary libraries
    -> Getting Images from Web
    -> Image Processing and Visualization
Step 3: Define Util Functions
Step 4: Load Pretrained Faster-RCNN Model from torchvision
    -> model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
Step 5: Create necessary transforms
Step 6: Load Image for inference, transform it
Step 7: Pass the Image to model, obtain outputs
    :: For each image, we have {'boxes': Tensor([[],[],[],[],..]),
                                'labels': Tensor([a,b,c,d,..]),
                                'scores': Tensor([s1,s2,s3,s4,..])}
Step 8: Extract the Class Name, Bounding Box details and Confidence
Step 9: Convert image tensor to openCV array and draw bounding boxes
Step 10: Save the opencv images



