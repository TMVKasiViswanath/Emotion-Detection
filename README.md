# Emotion Detection Project
This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

# Dependencies
- python3, [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/)<br>
- To install the required packages `pip install -r requirements.txt`

# Basic usage
The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

- Firstly clone the Repository
```
git clone https://github.com/TMVKasiViswanath/Emotion-Detection.git
cd Emotion-Detection
```
- Download the FER-2013 dataset [Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- If you want to train this model, use:
 ```
 cd Training
 python train.py
 ```
- All the best weights will be stored inside the test directory as `ResNet50_Transfer_Learning.keras`, so to test the model use:
```
cd ../test
python test.py
```
- This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 67% with average recall of 63% in 40 epochs.<br>
<img width="837" alt="accuracy_graph" src="https://github.com/TMVKasiViswanath/Emotion-Detection/assets/137616505/d64e9d8d-72ca-4d26-8956-5d35ba084368">

## Model Architectures
### Custom Architecture
Initially, a custom architecture was designed, but it did not achieve satisfactory accuracy and recall.

### Data Augmentation
To address the imbalance in the dataset, data augmentation techniques were applied. Despite this, the custom architecture did not yield improved results.

### Transfer Learning
Transfer learning was then employed by fine-tuning the VGG16 model. While this approach improved accuracy, recall remained suboptimal.

### ResNet50 with Data Augmentation and Class Weights
Finally, the ResNet50 model was used with data augmentation and class weights to handle the imbalance in the data. This approach resulted in a final accuracy of 67% and an average recall of 63% after just 10 epochs.

