# Age Classification from Face Images using CNN

This project explores the use of Convolutional Neural Networks (CNNs) for age classification from facial images. Using the UTKFace dataset, which contains over 20,000 images labeled with age, gender, and ethnicity, the focus of this work is on predicting age categories from face images. The model groups ages into five broad categories to simplify the classification task while maintaining meaningful distinctions.

The dataset used in this project is the UTKFace dataset, publicly available on Kaggle
. Only the age labels are used in this work, while gender and ethnicity are left for potential future extensions. The preprocessing stage involves extracting age labels from image filenames, converting them into defined age groups, normalizing pixel values, and preparing the data for training.

The model is built using TensorFlow and Keras, with a series of convolutional and pooling layers followed by fully connected dense layers. A softmax output layer is used to classify images into the defined age groups. Training and validation are performed on the UTKFace dataset, and performance is evaluated using standard metrics such as accuracy and loss.
