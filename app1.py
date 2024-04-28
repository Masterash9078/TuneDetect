from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import librosa.display
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import csv
app = Flask(__name__, static_url_path='/static')

# Load the pre-trained model+
model = load_model('model.h5')

# Dynamically generate the reverse mapping based on your dataset
# Assuming you have a CSV file with 'filename' and 'class' columns
# Replace 'path/to/your/dataset.csv' with the actual path to your dataset CSV
dataset_path ='D:/FinalYearProject/music_final/musicmix.csv'
df = pd.read_csv(dataset_path)
labels = df['Class'].unique()
reverse_mapping = {i: label for i, label in enumerate(labels)}

def process_audio(file_path):
    try:
        signal, sr = librosa.load(file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmin=2000, fmax=3000)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        img = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=13)

        # Resize the image to a fixed size (e.g., 128x128 pixels)
        img = cv2.resize(np.array(img), dsize=(128, 128))

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img_rgb
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
     return render_template('about.html')
 
@app.route('/sync')
def sync():
    return render_template('sync.html') 

def retrieve_class_by_filename(csv_filename, target_filename):
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if target_filename in row[0]:  # Check if target filename is in the first column (File Name)
                return row
    return None  # Return None if no matching filename is found
# Example usage:


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        processed_audio = process_audio(file_path)
        result_class = retrieve_class_by_filename(dataset_path,file.filename)

        

        if processed_audio is not None:
            processed_audio = np.expand_dims(processed_audio, axis=0)
            prediction = model.predict(processed_audio)
            print("Prediction probabilities:", prediction)  # Print the predicted probabilities

            predicted_class_index = np.argmax(prediction)
            predicted_class = reverse_mapping.get(predicted_class_index, "Unknown")
            print(file.filename)
            return render_template('result.html', predicted_class=result_class[1], file_name=file.filename)

    return redirect(request.url)

@app.route('/uploadd', methods=['POST'])
def uploadd():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        

if __name__ == '__main__':
    app.run(debug=True)
