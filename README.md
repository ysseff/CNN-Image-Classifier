# **CNN-Based Cat and Dog Classifier**

Welcome to the **CNN-Based Cat and Dog Classifier**! 🎉 This project is a user-friendly application that uses Convolutional Neural Networks (CNNs) to classify images as either **“cat”** or **“dog”** with high confidence. Whether you’re new to deep learning or looking to experiment with CNNs, this tool makes it easy to load datasets, train models, and classify images with just a few clicks.

# **Features**

# **🧠 Model Training & Loading**

- Train your own CNN model from scratch using your custom training and validation datasets.
- Load pretrained models for quicker classification.

# **📊 Real-Time Training Metrics**

- Monitor **accuracy** and **loss** during training to track model performance.

# **🖼️ Image Classification**

- Test images can be loaded into the app, and the classifier predicts whether the image is a “cat” or a “dog” with a confidence score.

# **⚙️ Hyperparameter Configuration**

- Adjust the number of training epochs directly through the GUI.

# **Key Results**

- Achieved **91.55% accuracy** during model training.
- Classified test images with a confidence level of **99.67%**.

# **Technologies Used**

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Tkinter
- **GUI Framework**: Tkinter for a seamless and interactive user interface.

# **Getting Started**

# **Prerequisites**

Make sure you have the following installed on your system:

- Python 3
- TensorFlow
- Keras
- NumPy

You can install the dependencies using pip:

pip install tensorflow keras numpy

# **How to Run the Project**

1. Clone the repository:
    
    → git clone https://github.com/ysseff/cnn-cat-dog-classifier.git
    
    → cd cnn-cat-dog-classifier
    
2. Run the application:
    
    → python [app.py](http://app.py/)
    
3. Use the GUI to:
    - Load training and validation datasets.
    - Train a new model or load an existing one.
    - Test images for classification.

# **How It Works**

1. **Dataset Preparation**:
    - Provide labeled datasets for training and validation.
    - Ensure folders are structured as:
        
        ```
        ├── training_set
        │   ├── cats
        │   └── dogs
        ├── validation_set
            ├── cats
            └── dogs
        
        ```
        
2. **Training the Model**:
    - Select the **number of epochs** and click “Train Network” to start training.
3. **Classifying Images**:
    - Load a test image, and the app will predict whether it’s a “cat” or “dog” with a confidence score.

# **Screenshots**

![PHOTO-2024-05-14-13-39-18](https://github.com/user-attachments/assets/c2909619-7288-4f16-b07e-bddea39b728b)

*Figure: Example of the application's GUI during image classification.*

# **Future Improvements**

- Add support for additional classes beyond cats and dogs.
- Implement model export for reuse in other applications.
- Transition the application to a **web-based platform** for broader accessibility and ease of use.
