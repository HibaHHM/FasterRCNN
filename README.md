# FasterRCNN
Object detection with Faster R-CNN and PyTorch

Apply Object detection with Faster R-CNN to classify predetermined objects

! pip3 install torch==1.13.0 torchvision==0.14.0 torchaudio
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/DLguys.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/watts_photos2758112663727581126637_b5d4d192d4_b.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/istockphoto-187786732-612x612.jpeg
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/jeff_hinton.png

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



