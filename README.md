# Fire detection from image and video frames
Aim of this project is to use Pretrained neural network VGG16 with transfer learning to detect fire.

# Introduction
The rapid advancement of vision-based fire detection models is driving the replacement of traditional fire detection methods with Computer Vision technology. Transfer learning, popular in recent years, enables new tasks to leverage pre-trained models that recognize patterns in images. These models can be adapted to detect fire in pictures or videos and offer advantages over traditional hardware-based systems, including higher accuracy, fewer errors, robustness to environmental factors, lower cost, and compatibility with existing camera surveillance systems.

# Data collection
- Dataset was taken from kaggle. It contains 755 Fire and 244 non-fire outdoor images in seperate folders.
- Since non-fire images were less compared to fire images data augmentation is appiled on non-fire images, so both fire and non-fire iamges are 755.
- 555 images were kept for training and 200 images for testing in both classes.

# Training
- Selected `VGG16` for this task, kept the convolutional layers only and added custom fully connected layers.
- Output activation function is taken to be `Sigmoid`, it gives fraction between 0-1.
- Model is trained on batches of data for 10 epochs resulted in `validation_accuracy` around 0.94.

# Steps to re-create this model
- Download the dataet from kaggle and run the data augmentation `.ipynb` code step by step.
- Run the fire-model-creation `.ipynb` file and save the model to use it on images and videos
- You could also run `app.py` in your system to get results on video.

# Further Improvements
- Get more variety in datset, e.g. indoor stove flames, camp fires, fire in buildings to reduce over-fitting.
- Try different models like MobileNET, ResNET and compare their accuracy against this model.
