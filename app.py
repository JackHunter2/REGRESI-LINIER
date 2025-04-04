import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Pastikan folder uploads tersedia
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values  # Semua kolom kecuali target (TV, Radio, Newspaper)
    y = df.iloc[:, -1].values   # Kolom target (Sales)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.savefig('static/prediction.png')
    plt.close()
    
    return model, X_test, y_test, y_pred


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            global X_test, y_test, predictions  # Simpan data untuk halaman result
            model, X_test, y_test, predictions = train_model(filepath)
            return redirect(url_for('result'))
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html', image='static/prediction.png', 
                           X_test=X_test, y_test=y_test, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
