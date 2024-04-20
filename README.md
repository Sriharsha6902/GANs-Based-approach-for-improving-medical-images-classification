# Medical Image Classification using GANs

## Overview

Medical imagery classification has historically relied on limited datasets, often leading to suboptimal results due to insufficient data. Traditional approaches to augmenting datasets have been helpful but may introduce bias and limitations. This project proposes a novel solution using Generative Adversarial Networks (GANs) to generate synthetic datasets, thereby improving classification model performance.

## Objective

The objective of this project is to enhance the performance of medical image classification models by leveraging GANs to create realistic synthetic datasets. Specifically, we aim to classify medical images into different categories (e.g., normal and pneumonia) and demonstrate that utilizing synthetic data leads to improved accuracies compared to traditional methods.

## Methodology

1. **Data Collection**: We collected a dataset containing medical images, focusing on pneumonia cases. You can find the dataset [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Data Preprocessing**: Preprocessing steps included resizing, normalization, and label encoding.

3. **GAN-based Data Generation**: We trained a Auxiliary Classifier Generative Adversarial Network (ACGAN) to generate synthetic medical images and their corresponding labels.

4. **Classification Model Training**: We utilized the generated synthetic data along with the original dataset to train a classification model. In this case, a VGG16 model was employed for the classification task.


## Usage

1. Clone the repository: `git clone https://github.com/Sriharsha6902/GANs-Based-approach-for-improving-medical-images-classification.git`
2. Navigate to the project directory
3. Install dependencies: `pip install -r requirements.txt`
4. Execute the notebook to preprocess data, train the ACGAN, and train the classification model.

## Conclusion

This project demonstrates the effectiveness of utilizing GANs for generating synthetic data in medical image classification tasks. By leveraging synthetic datasets, we can overcome limitations associated with traditional data augmentation techniques, resulting in improved classification accuracies.
