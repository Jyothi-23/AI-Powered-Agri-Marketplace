from flask import Flask, render_template, request, session
import webbrowser
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import joblib
import config
import requests
import sqlite3
import base64
from keras.models import load_model
import json
import random
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import pickle
import nltk
lemmatizer = WordNetLemmatizer()


app = Flask(__name__)
app.config['SECRET_KEY'] = '895623741'



database="new2.db"

def createtable():
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("create table if not exists register(id integer primary key autoincrement, name text,email text,password text,status text)")

    cursor.execute("create table if not exists user_register(id integer primary key autoincrement, name text,email text,password text,status text)")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sellers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            location TEXT,
            shop TEXT,
            logo BLOB,
            product TEXT,
            quantity TEXT,
            price REAL,
            description TEXT
        )
    ''')
    conn.commit()
    conn.close()
createtable()

model1 = load_model('class1_model.h5')  

class_labels = ['healthy', 'unhealthy'] 

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/seller_reg')
def seller_reg():
    return render_template('seller_register.html')

@app.route('/leaf_prediction')
def leaf_prediction():
    return render_template('leaf_prediction.html')

@app.route('/disease_prediction')
def disease_prediction():
    return render_template('disease_prediction.html')


@app.route('/crop_predict_page')
def crop_predict_page():
    return render_template('crop_predict.html')

@app.route('/seller_register', methods=["GET","POST"])
def seller_register():
    if request.method=="POST":
        name=request.form['name']
        email=request.form['email']

        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute(" SELECT email FROM register WHERE email=?",(email,))
        registered=cursor.fetchall()
        if registered:
            return render_template('seller_register.html', alert_message="Email Already Registered")
        else:
            cursor.execute("insert into register(name,email,password,status) values(?,?,?,?)",(name,email,password,0))
            conn.commit()
            return render_template('seller_login.html', alert_message="Registered Succussfully")
    return render_template('seller_register.html')



@app.route('/seller_login', methods=["GET", "POST"])
def seller_login():
    global data
    global email
    if request.method == "POST":        
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM register WHERE email=? AND password=?", (email, password))
        data = cursor.fetchone()


        if data is None:
            return render_template('seller_register.html', alert_message="Email Not Registered or Check Password")
        else:
            session['email'] = email
            return render_template('seller_dashboard.html')

    return render_template('seller_login.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Important! same as training
        
        predictions = model1.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        prediction_label = class_labels[predicted_class]

        return render_template('leaf_result.html', image_path=filepath, prediction=prediction_label)


model2 = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')



def weather_fetch(city_name):
    """
    Fetch and return the temperature and humidity of a city.
    :param city_name: str
    :return: (temperature, humidity) or None
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    data = response.json()

    if data.get("cod") != 404 and "main" in data:
        main_data = data["main"]
        temperature = round(main_data["temp"] - 273.15, 2)
        humidity = main_data["humidity"]
        return temperature, humidity
    else:
        print(f"Error fetching weather for '{city_name}': {data.get('message', 'Unknown error')}")
        return None


@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    state = request.form.get("stt")
    city = request.form.get("city")
    
    weather_data = weather_fetch(city)
    if weather_data is not None:
        temperature, humidity = weather_data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model2.predict(input_scaled)
        predicted_crop = prediction[0]

        return render_template(
            'crop_result.html',
            prediction_text=f"The crop best suited for {city}, {state} is: {predicted_crop}",
            N=N, P=P, K=K, temperature=temperature,
            humidity=humidity, ph=ph, rainfall=rainfall,
            state=state, city=city, predicted_crop=predicted_crop
        )
    else:
        return render_template(
            'crop_result.html',
            prediction_text="Weather data not available for the selected city.",
            N=N, P=P, K=K, ph=ph, rainfall=rainfall,
            state=state, city=city
        )

    

@app.route('/seller_product', methods=['GET', 'POST'])
def seller_product():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        location = request.form['location']
        shop = request.form['shop']
        product = request.form['product']
        quantity = request.form['quantity']
        price = request.form['price']
        description = request.form['description']

        # Convert uploaded image to binary
        file = request.files['logo']
        logo_blob = file.read()

        # Insert into DB
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO sellers (name, email, phone, location, shop,logo, product,quantity,price, description)
            VALUES (?, ?, ?, ?, ?, ?, ?,?,?,?)
        ''', (name, email, phone, location, shop,logo_blob, product,quantity,price, description))
        conn.commit()
        conn.close()

        return render_template('seller_dashboard.html' , alert_message="Form submitted and image saved in database!")

    return render_template('seller_dashboard.html')


@app.route('/buy_product_page')
def buy_product():
    return render_template('user_register.html')


@app.route('/user_register', methods=["GET","POST"])
def user_register():
    if request.method=="POST":
        name=request.form['name']
        email=request.form['email']

        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute(" SELECT email FROM user_register WHERE email=?",(email,))
        registered=cursor.fetchall()
        if registered:
            return render_template('user_register.html', alert_message="Email Already Registered")
        else:
            cursor.execute("insert into user_register(name,email,password,status) values(?,?,?,?)",(name,email,password,0))
            conn.commit()
            return render_template('user_register.html', alert_message="Registered Succussfully")
    return render_template('register.html')



@app.route('/user_login', methods=["GET", "POST"])
def user_login():
    global data
    global email
    if request.method == "POST":        
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_register WHERE email=? AND password=?", (email, password))
        data = cursor.fetchone()


        if data is None:
            return render_template('user_register.html', alert_message="Email Not Registered or Check Password")
        else:
            session['email'] = email
            return render_template('buy_product.html')

    return render_template('user_register.html')




@app.route('/show_products', methods=['POST'])
def show_products():
    selected_location = request.form['location']
    
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("SELECT name, email, phone, location, shop,logo, product,quantity, price, description FROM sellers WHERE location=?", (selected_location,))
    rows = cur.fetchall()
    conn.close()

    products = []
    for row in rows:
        product = {
            'name': row[0],
            'email': row[1],
            'phone': row[2],
            'location': row[3],
            'shop' : row[4],
            'logo': base64.b64encode(row[5]).decode('utf-8'),
            'product' : row[6],
            'quantity': row[7],
            'price': row[8],
            'description': row[9]
        }
        products.append(product)

    return render_template('buy_product.html', products=products, location=selected_location)   


model = load_model("chatbot_model.h5")
filename="intents.json"
intents = json.loads(open(filename, encoding="utf8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

@app.route("/get", methods=["POST"])
def get_bot_response():
    msg = request.form["msg"]
    return chatbot_response(msg)


import re

expense_keywords = []
expense_amounts = {}

def chatbot_response(msg):
    messg = msg.lower()

    
    
    ints = predict_class(messg, model)
    res = getResponse(ints, intents)
    return res


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list


def getResponse(ints, intents_json):
    if not ints:  
        return "Sorry, I didn't understand that. Could you please clarify?"

    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


UPLOAD_FOLDER1 = 'static/image_uploads'
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1

if not os.path.exists(UPLOAD_FOLDER1):
    os.makedirs(UPLOAD_FOLDER1)

model_ef = load_model("model_eff.h5")

class_names = ['Rust and Scab Disease', 'Health Leaf', 'Rust Disease', 'Scab Disease']

def predict_image(image_path, model, class_names):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, predictions[0]


a=[]
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER1'], 'original.jpg')
        file.save(filepath)

        predicted_class, probabilities = predict_image(filepath, model_ef, class_names)
        a.append(predicted_class)
        print(a)

        return render_template('disease_prediction.html', image_path=filepath, predicted_class=predicted_class, probabilities=probabilities)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:1000")  
    app.run(port=1000)
