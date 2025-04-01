from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Fungsi untuk memproses dataset
def load_and_train_model(file_path=None):
    if file_path:
        df = pd.read_csv(file_path)
    else:
        dataset_url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/advertising.csv"
        df = pd.read_csv(dataset_url)
    
    # Membagi dataset
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Menyiapkan fitur dan target
    X_train = train_data[['TV', 'Radio', 'Newspaper']]
    y_train = train_data['Sales']
    
    # Melatih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Simpan model
    model_filename = "model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    
    return df

# Muat dan latih model awal
df = load_and_train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            tv = float(request.form['tv'])
            radio = float(request.form['radio'])
            newspaper = float(request.form['newspaper'])
            
            with open("model.pkl", "rb") as file:
                loaded_model = pickle.load(file)
            
            prediction = loaded_model.predict([[tv, radio, newspaper]])[0]
        except ValueError:
            prediction = "Input tidak valid"
    
    return render_template('index.html', prediction=prediction)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            load_and_train_model(file_path)
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/visualize')
def visualize():
    plt.figure(figsize=(8, 6))
    sns.pairplot(df, kind='reg')
    plt.savefig("static/plot.png")
    return render_template('visualize.html', image_url="static/plot.png")

if __name__ == '__main__':
    app.run(debug=True)