# Emotion Detection Project
This project aims to classify facial expressions into 7 categories: angry, neutral, disgust, surprise, sad, fear, and happy. The dataset used for this project is the FER2013 dataset, which can be found on Kaggle: FER2013 Dataset.

# Model Architectures
## Custom Architecture
Initially, a custom architecture was designed, but it did not achieve satisfactory accuracy and recall.

## Data Augmentation
To address the imbalance in the dataset, data augmentation techniques were applied. Despite this, the custom architecture did not yield improved results.

## Transfer Learning
Transfer learning was then employed by fine-tuning the VGG16 model. While this approach improved accuracy, recall remained suboptimal.
![Alt Text](https://app.gemoo.com/share/image-annotation/646385369401106432?codeId=PalnKzoonGdAV&origin=imageurlgenerator&card=646385365840142336)

## ResNet50 with Data Augmentation and Class Weights
Finally, the ResNet50 model was used with data augmentation and class weights to handle the imbalance in the data. This approach resulted in a final accuracy of 67% and an average recall of 63% after just 10 epochs.

## Conclusion
The combination of ResNet50, data augmentation, and class weights proved to be the most effective approach for this emotion detection project. Further optimization and experimentation could potentially improve the model's performance
