import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import time


def load_images(is_train):
    filetypes = [('PNG files', '*.png'), ('JPG files', '*.jpg'), ('JPEG files', '*.jpeg')]
    if is_train:
        directory = filedialog.askdirectory(title="Select training images directory")
        if directory:
            print(f"Training images loaded from: {directory}")
            process_images(directory)
            print(f"Loading is complete.\nLoaded {images}\nLoaded {labels}\nDone")
        else:
            print("No training images selected")
    else:
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            print(f"Test image loaded: {filename}")
            display_image(filename)
        else:
            print("No test image selected")

def process_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            image = Image.open(file_path)
            image = image.resize((224, 224), Image.LANCZOS)
            images.append(image)
            labels.append(filename)

def display_image(path):
    image = Image.open(path)
    image = image.resize((150, 150), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

def train_network():
    print("Training started...")
    for i in range(100):
        progress_var.set(i + 1)
        root.update_idletasks()
        time.sleep(0.05)
    output_text.insert(tk.END, "Training complete!\n")

def classify_image():
    print("Classification started...")
    output_text.insert(tk.END, f"Image Classified as: <class_name>")


images = []
labels = []

root = tk.Tk()
root.title("CNN Image Classification")


frame_train = tk.LabelFrame(root, text="Training Images", padx=10, pady=10)
frame_train.pack(padx=10, pady=5, fill="both", expand="yes")

btn_load_train = ttk.Button(frame_train, text="Load Images", command=lambda: load_images(True))
btn_load_train.pack(side=tk.LEFT)

frame_test = tk.LabelFrame(root, text="Test Images", padx=10, pady=10)
frame_test.pack(padx=10, pady=5, fill="both", expand="yes")

btn_load_test = ttk.Button(frame_test, text="Load Images", command=lambda: load_images(False))
btn_load_test.pack(side=tk.LEFT)

image_label = tk.Label(frame_test)
image_label.pack(side=tk.LEFT, padx= 30)

frame_config = tk.LabelFrame(root, text="Config", padx=10, pady=10)
frame_config.pack(padx=10, pady=5, fill="both", expand="yes")

initial_value = tk.IntVar(value=10)
tk.Label(frame_config, text="Epochs:").pack(side=tk.LEFT)
entry_epochs = ttk.Spinbox(frame_config, from_=1, to=30, width=5, textvariable=initial_value, wrap=False)
entry_epochs.pack(side=tk.LEFT, padx=5)

btn_train = ttk.Button(root, text="Train Network", command=train_network)
btn_train.pack(fill='x', padx=10, pady=5)

btn_classify = ttk.Button(root, text="Classify Image", command=classify_image)
btn_classify.pack(fill='x', padx=10, pady=5)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(fill='x', padx=10, pady=5)

output_text = tk.Text(root, height=4)
output_text.pack(fill='both', padx=10, pady=5, expand=True)

root.mainloop()
