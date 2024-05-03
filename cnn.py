import tkinter as tk
from tkinter import filedialog, ttk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Image Classifier")

        # Initialize the model
        self.model = create_model()
        self.datagen = create_image_generator()
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)


        # GUI components setup
        self.setup_widgets()

    def setup_widgets(self):
        frame_train = tk.LabelFrame(self.root, text="Training Images", padx=10, pady=10)
        frame_train.pack(padx=10, pady=5, fill="both", expand="yes")

        btn_load_train = ttk.Button(frame_train, text="Load Training Images", command=self.load_training)
        btn_load_train.pack(side=tk.LEFT)

        self.accuracy_label = tk.Label(frame_train)
        self.accuracy_label.pack(side=tk.RIGHT, padx=30)

        frame_test = tk.LabelFrame(self.root, text="Test Image", padx=10, pady=10)
        frame_test.pack(padx=10, pady=5, fill="both", expand="yes")

        self.btn_load_test = ttk.Button(frame_test, text="Load Testing Image", command=self.load_and_display_test)
        self.btn_load_test.pack(side=tk.LEFT)

        self.image_label = tk.Label(frame_test)
        self.image_label.pack(side=tk.LEFT, padx=30)

        self.result_label = tk.Label(frame_test)
        self.result_label.pack(side=tk.RIGHT, padx=30)

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

        self.output_text = tk.Text(self.root, height=10, background='white', foreground='black')
        self.output_text.pack(fill='both', padx=10, pady=5, expand=True)
        self.output_text.configure( font='helvetica 14')

    def load_training(self):
        try:
            self.train_gen = load_training_images(self.datagen)  # Your function to load images
            self.output_text.insert(tk.END, "Training images loaded successfully.\n")
        except FileNotFoundError:
            self.output_text.insert(tk.END, "Training images not found.\n")

    def train_model(self):
        try:
            epochs = int(self.entry_epochs.get())
            loss, accuracy = train_model(self.train_gen, self.model, epochs)
            self.accuracy_label.config(text=f"Accuracy:{accuracy*100:.2f}%, Loss: {loss*100:.2f}%", font=("Helvetica", 16))
            self.output_text.insert(tk.END, f"Training complete with loss: {loss*100:.2f} and accuracy: {accuracy*100:.2f}\n")
        except Exception as e:
            self.output_text.insert(tk.END, "No training data loaded.\n")
            print(e)


    def load_and_display_test(self):
        try:
            img_array, img = load_test_image(self.test_datagen)  # Ensure to pass the correct ImageDataGenerator instance
            if img is not None:
                # Prepare the image for display
                img = img.resize((180, 180), Image.LANCZOS)  # Resize using PIL if needed for display purposes
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                self.test_img = img_array  # This should be the processed array used for prediction
                self.output_text.insert(tk.END, "Test image loading complete.\n")
            else:
                self.output_text.insert(tk.END, "Failed to load test image.\n")
        except FileNotFoundError:
            self.output_text.insert(tk.END, "Test image not found.\n")


    def classify_img(self):
        try:
            if self.test_img is None:
                self.output_text.insert(tk.END, "Failed to load test data.\n")

            result, confidence = classify_image(self.test_img, ['cats', 'dogs'], self.model)
            self.result_label.config(text=f"class: {result}, confidence: {confidence:.2f}%", font=("Helvetica", 16))
            self.output_text.insert(tk.END, f"Image classified as: {result}\n")
        except AttributeError:
            self.output_text.insert(tk.END, "Failed to classify the image due to the lack of test data.\n")


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),  # Dropout after first pooling layer

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),  # Dropout after second pooling layer

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Higher dropout rate before the final dense layer
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_image_generator():
    datagen = ImageDataGenerator(
        rotation_range=40,  # Randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # Randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # Randomly shift images vertically (fraction of total height)
        shear_range=0.2,  # Randomly shear transformations
        zoom_range=0.2,  # Randomly zoom image
        horizontal_flip=True,  # Randomly flip images
        fill_mode='nearest',  # Strategy used for filling in newly created pixels
        rescale=1. / 255  # Rescale the pixel values (important for normalization)
    )

    return datagen

def train_model(images, model, epochs):
    model.fit(images, epochs=epochs)
    loss, accuracy = model.evaluate(images)
    return loss, accuracy


def load_training_images(datagen):
    directory = filedialog.askdirectory(title="Select training data directory")
    train_generator = datagen.flow_from_directory(
        directory,  # Path to the target directory
        target_size=(224, 224),  # Resizes all images to 224 x 224
        batch_size=32,  # Size of the batches of data (default: 32)
        class_mode='binary'  # Type of classification (binary for 2 classes)
    )

    return train_generator


def load_test_image(test_datagen):
    filetypes = [('PNG files', '*.png'), ('JPG files', '*.jpg'), ('JPEG files', '*.jpeg')]
    image_path = filedialog.askopenfilename(title="Open file", filetypes=filetypes)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = next(test_datagen.flow(img_array, batch_size=1))[0]  # Apply the same transformations as training

    # Convert array back to PIL image for display
    img = Image.fromarray((img_array * 255).astype('uint8'), 'RGB')

    return img_array, img


def classify_image(image_array, class_names, model):
    if image_array is None:
        return "No image loaded", None

    image_array = np.expand_dims(image_array, axis=0)  # Prepare batch
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_class_name = class_names[predicted_index]  # Map index to class name
    confidence = np.max(predictions) * 100  # Confidence percentage

    return predicted_class_name, confidence



if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("740x640")
    app = ImageClassifierApp(root)
    root.mainloop()
