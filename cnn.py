import tkinter as tk
from tkinter import filedialog, ttk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageTk
import random
import os


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Image Classifier")

        # Initialize the model
        self.model = create_model()


        # GUI components setup
        self.setup_widgets()

    def setup_widgets(self):
        frame_train = tk.LabelFrame(self.root, text="Training Images", padx=10, pady=10)
        frame_train.pack(padx=10, pady=5, fill="both", expand="yes")

        btn_load_train = ttk.Button(frame_train, text="Load Training Images", command=self.load_training)
        btn_load_train.pack(side=tk.LEFT)

        frame_test = tk.LabelFrame(self.root, text="Test Image", padx=10, pady=10)
        frame_test.pack(padx=10, pady=5, fill="both", expand="yes")

        self.btn_load_test = ttk.Button(frame_test, text="Load Testing Image", command=self.load_and_display_test)
        self.btn_load_test.pack(side=tk.LEFT)

        self.image_label = tk.Label(frame_test)
        self.image_label.pack(side=tk.LEFT, padx=30)

        frame_config = tk.LabelFrame(self.root, text="Configuration", padx=10, pady=10)
        frame_config.pack(padx=10, pady=5, fill="both", expand="yes")

        self.initial_value = tk.IntVar(value=10)
        tk.Label(frame_config, text="Epochs:").pack(side=tk.LEFT)
        self.entry_epochs = ttk.Spinbox(frame_config, from_=10, to=30, width=5, textvariable=self.initial_value, wrap=False)
        self.entry_epochs.pack(side=tk.LEFT, padx=5)

        btn_train = ttk.Button(root, text="Train Network", command=self.train_model)
        btn_train.pack(fill='x', padx=10, pady=5)

        btn_test = ttk.Button(root, text="Classify Image", command=self.classify_img)
        btn_test.pack(fill='x', padx=10, pady=5)

        self.output_text = tk.Text(self.root, height=4)
        self.output_text.pack(fill='both', padx=10, pady=5, expand=True)

    def load_training(self):
        self.training_images, self.training_labels = load_training_images()  # Your function to load images
        if self.training_images.size > 0 and len(self.training_labels) > 0:
            self.output_text.insert(tk.END, "Training images loaded successfully.\n")
        else:
            self.output_text.insert(tk.END, "Failed to load training data.\n")

    def train_model(self):
        try:
            if self.training_images.size > 0 and len(self.training_labels) > 0:
                epochs = int(self.entry_epochs.get())
                train_model(self.training_images, self.training_labels, self.model, epochs)
                self.output_text.insert(tk.END, "Training complete!.\n")
            else:
                self.output_text.insert(tk.END, "Failed to train the model due to the lack of training data.\n")
        except AttributeError:
            self.output_text.insert(tk.END, "No training data loaded.\n")


    def load_and_display_test(self):
        img_array, img = load_test_image()
        img = img.resize((150, 150), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.test_img = img_array
        self.output_text.insert(tk.END, "Test image loading complete.\n")


    def classify_img(self):
        try:
            if self.test_img is None:
                self.output_text.insert(tk.END, "Failed to load test data.\n")

            result = classify_image(self.test_img, ['cats', 'dogs'], self.model)
            self.output_text.insert(tk.END, f"Image classified as: {result}\n")
        except AttributeError:
            self.output_text.insert(tk.END, "Failed to classify the image due to the lack of test data.\n")


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(images, labels, model, epochs):
    model.fit(images, labels, epochs=epochs)


def augment_image(img):
    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Random rotation
    rotation_degree = random.randint(-30, 30)  # Adjust the degree range as needed
    img = img.rotate(rotation_degree)

    # Random color jitter
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.5, 1.5))

    return img


def load_training_images():
    filetypes = ('.png', '.jpg', '.jpeg')  # Simplified and correct handling for file extensions
    images = []
    labels = []
    directory = filedialog.askdirectory(title="Select Directory")
    if not directory:
        print("No training images selected")
        return np.array([]), np.array([])  # Return empty lists if no directory is selected

    label_mapping = {'cats': 0, 'dogs': 1}

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(filetypes):  # Check file extension
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        with Image.open(image_path) as img:
                            img = augment_image(img)
                            img = img.resize((224, 224))
                            img = img.convert('RGB')
                            img_array = np.array(img) / 255.0
                            images.append(img_array)
                            labels.append(label_mapping[folder_name])  # Assuming folder names are class names
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    return np.array(images), np.array(labels)


def load_test_image():
    filetypes = [('PNG files', '*.png'), ('JPG files', '*.jpg'), ('JPEG files', '*.jpeg')]
    image_name = filedialog.askopenfilename(title="Open file", filetypes=filetypes)

    if not image_name:
        print("No test images selected")
        return np.array([]), None

    img = Image.open(image_name)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0

    return img_array, img


def classify_image(image_array, class_names, model):
    if image_array is None:
        return "No image loaded"

    image_array = np.expand_dims(image_array, axis=0)  # Prepare batch
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_class_name = class_names[predicted_index]  # Map index to class name

    return predicted_class_name


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
