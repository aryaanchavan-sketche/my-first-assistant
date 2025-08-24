# Core system and utility libraries
import difflib
import os
import select
import sys
import json
from turtle import mode
import psutil
import pyttsx3
import webbrowser
import pyjokes
import datetime
import sched
import requests
import wikipedia
import pyautogui
import random
import cv2
import time
import threading
import smtplib
import pickle
import speech_recognition as sr
import platform
if platform.system() == "Windows":
      import msvcrt
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from scipy.spatial.distance import cosine
import tempfile
import numpy as np
import getpass
import sounddevice as sd
import platform
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from transformers import pipeline

VOICEPRINT_FILE = "voiceprint.npy"




def command():
    r = sr.Recognizer()
    print("Available microphones:")
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{idx}: {name}")
    with sr.Microphone() as source:
        speak("Listening...")
        r.adjust_for_ambient_noise(source, duration=3.0)  # More time for noise calibration
        r.energy_threshold = 100  # Lower for more sensitivity
        r.pause_threshold = 0.5
        r.phrase_time_limit = 30  # Allow longer phrases
        r.non_speaking_duration = 0.1
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=30)
        except Exception as e:
            speak("I couldn't hear anything. Please try again.")
            return "None"
    for attempt in range(2):  # Try twice for clarity
        try:
            speak("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said : {query} \n")
            speak(f"You said: {query}")
            return query.lower()
        except sr.UnknownValueError:
            speak("Sorry, I did not catch that. Please speak clearly and close to the mic.")
            print("Sorry, I did not catch that.")
            if attempt == 1:
                return "None"
        except sr.RequestError as e:
            # Only show error if no valid command was recognized at all
            if attempt == 1:
                # Do not speak or print error if user already said something valid
                return "None"
    return "None"
# The above block is invalid and should be removed.
# If you intended to use an API call, please implement it properly elsewhere.

def get_news():
    api_key = "your_newsapi_key"
    url = f"https://newsapi.org/v2/top-headlines?country=us&apikey={api_key}"
    news = requests.get(url).json()
    articles = news["articles"][:5]
    for article in articles:
        speak(article["title"])
scheduler = sched.scheduler(time.time, time.sleep)

def set_alarm(text, seconds):
    scheduler.enter(seconds, 1, speak, argument=("Alarm: {text}",))
    scheduler.run()



with open("intents.json") as file:
    data = json.load(file)

def initialize_engine():
    try:
        engine = pyttsx3.init("sapi5")
    except Exception:
        engine = pyttsx3.init()
    voice = engine.getProperty("voices")
    engine.setProperty("voice", voice[1].id)
    rate = engine.getProperty("rate")
    engine.setProperty("rate", rate-50)
    volume = engine.getProperty("volume")
    engine.setProperty("volume", volume+0.25)
    return engine
 
 
 
 
def speak(text):
    engine = initialize_engine()
    print(f"Lakshmi says: {text}")
    engine.say(text)
    engine.runAndWait()
    
def command():
    r = sr.Recognizer()
    
    print("Available Microphones:")
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{idx}: {name}")



    with sr.Microphone() as source:
        speak("Listening...")
        r.adjust_for_ambient_noise(source, duration=3.0)  # More time for noise calibration
        r.energy_threshold = 100  # Lower for more sensitivity
        r.pause_threshold = 0.5
        r.phrase_time_limit = 30  # Allow longer phrases
        r.non_speaking_duration = 0.1
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=30)
        except Exception as e:
            speak("I couldn't hear anything. Please try again.")
            return "None"
    for attempt in range(2):  # Try twice for clarity
        try:
            speak("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said : {query} \n")
            speak(f"You said: {query}")
            return query.lower()
        except sr.UnknownValueError:
            speak("Sorry, I did not catch that. Please speak clearly.")
            print("Sorry, I did not catch that.")
            if attempt == 1:
                return "None"
        except sr.RequestError as e:
            speak("Sorry, I am having trouble connecting to the speech service.")
            print(f"Speech recognition error: {e}")
            return "None"
    return query

def cal_day():
   day = datetime.datetime.today().weekday() +1
   day_dict={
       1:"MONDAY",
       2:"TUESDAY",
       3:"WEEDNESDAY",
       4:"THURSDAY",
       5:"FRIDAY",
       6:"SATURDAY",
       7:"SUNDAY"
   }
   if day in day_dict.keys():
      day_of_week =  day_dict[day]
      print(day_of_week)
   return day_of_week
       
       
def WishMe():
    hour=int(datetime.datetime.now().hour)
    t=time.strftime("%I:%M:%p")
    day= cal_day
    
    if(hour>=0) and (hour<=12) and ("AM" in t):
        speak(f"GOOD MORNING ARYAAN, it's {day} and the time is{t}")
    elif(hour>=12) and(hour<=16) and("PM" in t):
        speak(f"GOOD AFTERNOON ARYAAN, its {day} and the time is {t}")
    else:
        speak(f"GOOD EVENING ARYAAN, it's {day} and the time is{t}")




def social_media(command):
    if "facebook" in command:
        speak("Opening your facebook")
        webbrowser.open("https://www.facebook.com/")
    elif "whatsapp" in command:
        speak("opening yourr whatsapp")
        webbrowser.open("https://web.whatsapp.com/")
    elif "instagram" in command:
        speak("opening your instagram")
        webbrowser.open("https://www.instagram.com/")
    else:
        speak("no result found")
def schedule():
    day = cal_day().lower()
    speak(f"Boss today's schedule is{[day]}")
    week={
        "monday":("Boss from 9:00 am to 3:00 pm you have  regular classes, from 4:00 pm to 5:30 pm you have DSA course,from 5:30 to 6:30 free time for coding and playing,from 6:30 pm to 7:30 pm you have html and css course, from 7:30 to 8:00 you have dinner time,from 8:00 to 8:45 its your homework time, 9:00 to 9:30 you have free time and do something productive , 10:00 pm sleep ,4:45 AM wakeup"),
        "tuesday":("Boss from 9:00 am to 3:00 pm you have regular classes, from 4:00 pm to 5:30 pm you have DSA course,from 5:30 to 6:30 free time for coding and playing,from 6:30 pm to 7:30 pm you have html and css course, from 7:30 to 8:00 you have dinner time,from 8:00 to 8:45 its your homework time, 9:00 to 9:30 you have free time and do something productive , 10:00 pm sleep ,4:45 AM wakeup"),
        "wednesday":("Boss from 9:00 am to 3:00 pm you have regular classes, from 4:00 pm to 5:30 pm you have DSA course,from 5:30 to 6:30 free time for coding and playing,from 6:30 pm to 7:30 pm you have html and css course, from 7:30 to 8:00 you have dinner time,from 8:00 to 8:45 its your homework time, 9:00 to 9:30 you have free time and do something productive , 10:00 pm sleep ,4:45 AM wakeup"),
        "thursday":("Boss from 9:00 am to 3:00 pm you have  regular classes, from 4:00 pm to 5:30 pm you have DSA course,from 5:30 to 6:30 free time for coding and playing,from 6:30 pm to 7:30 pm you have html and css course, from 7:30 to 8:00 you have dinner time,from 8:00 to 8:45 its your homework time, 9:00 to 9:30 you have free time and do something productive , 10:00 pm sleep ,4:45 AM wakeup"),
        "friday":("Boss from 9:00 am to 3:00 pm you have  regular classes, from 4:00 pm to 5:30 pm you have DSA course,from 5:30 to 6:30 free time for coding and playing,from 6:30 pm to 7:30 pm you have html and css course, from 7:30 to 8:00 you have dinner time,from 8:00 to 8:45 its your homework time, 9:00 to 9:30 you have free time and do something productive , 10:00 pm sleep ,4:45 AM wakeup"),
        "saturday":("Boss from 9:00 am to 12:00 pm you have regular classes, from 4:00 pm to 5:30 pm you have DSA course,from 5:30 to 6:30 free time for coding and playing,from 6:30 pm to 7:30 pm you have html and css course, from 7:30 to 8:00 you have dinner time,from 8:00 to 8:45 its your homework time, 9:00 to 9:30 you have free time and do something productive , 10:00 pm sleep ,4:45 AM wakeup"),
        "sunday":("Boss sunday is a holiday, so you can rest and do some productive work and explore new things")
    }
    if day in week.keys():
        speak(week[day])   
        
def openApp(command):
    if "notepad" in command:
        speak("Opening Notepad")
        os.startfile("notepad.exe")
    elif "calculator" in command:
        speak("Opening Calculator")
        os.startfile("calc.exe")

def browsing(query):
    if "google" in query or "open google" in query:
        speak("Boss, what should I search on Google?")
        if mode == "voice":
            s = command().lower()
        else:
            s = input("Enter your Google search query: ").strip()
        if s:
            webbrowser.open(f"https://www.google.com/search?q={s}")
            speak(f"Searching Google for {s}")
        else:
            speak("No search query provided.")
    elif "edge" in query or "open edge" in query:
        speak("Opening Microsoft Edge")
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        if os.path.exists(edge_path):
            os.startfile(edge_path)
        else:
            speak("Microsoft Edge not found.")

def tell_joke():
    joke = pyjokes.get_joke()
    speak(joke)
    if "calculater" in command:
        speak("opening your calculater")
        os.startfile(r"c:\windows\system32\calc.exe")
    elif "notepad" in command:
        speak("opening your notepad")
        os.startfile(r"c:\windows\system32\notepad.exe")
    elif "paint" in command:
        speak("opening your paint")
        os.startfile(r"c:\windows\system32\mspaint.exe")
    elif "cmd" in command:
        speak("opening your command prompt")
        os.startfile(r"c:\windows\system32\cmd.exe")
    elif "word" in command:
        speak("opening your word")
        os.startfile(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE")
    elif "visual studio code" in command:
        speak("opening your visual studio code")
        os.startfile(r"C:\Program Files\Microsoft VS Code\Code.exe")

def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        speak(result)
    except Exception:
        speak("Sorry, I couldn't find anything on Wikipedia.")

def wikipedia_voice_search():
    speak("What should I search on Wikipedia?")
    if mode == "voice":
        topic = command().lower()
    else:
        topic = input("Enter topic for Wikipedia search: ").strip()
    if topic:
        search_wikipedia(topic)
    else:
        speak("No topic provided.")

def get_weather(city):
    api_key = "your_openweathermap_api_key" # Replace with your API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data["cod"] == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            speak(f"The temperature in {city} is {temp}°C with {desc}.")
        else:
            speak("City not found.")
    except Exception:
        speak("Unable to get weather information.")

def get_response_from_intents(query):

    best_match = None
    best_score = 0
    best_intent = None
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            score = difflib.SequenceMatcher(None, pattern.lower(), query.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = pattern
                best_intent = intent
    if best_score > 0.6:
        response = random.choice(best_intent["responses"])
        speak(response)
        return response
    speak("Sorry, I don't understand.")
    return None


 

conversation_history = []

def register_face_once():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if os.path.exists("registered_eyes.jpg"):
        speak(" Aryan Sir, your eyes are already registered.")
        return
    cap = cv2.VideoCapture(0)
    speak("Please look at the camera to register your eyes.")
    saved = False
    while not saved:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                (ex1, ey1, ew1, eh1) = eyes[0]
                (ex2, ey2, ew2, eh2) = eyes[1]
                eye_img = roi_color[min(ey1,ey2):max(ey1+eh1,ey2+ew2), min(ex1,ex2):max(ex1+ew1,ex2+ew2)]
                cv2.imwrite("registered_eyes.jpg", eye_img)
                speak("Eyes registered and saved. Lakshmi will activate only for your eyes scan.")
                saved = True
        cv2.imshow('Register Eyes - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def register_additional_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    speak("Look at the camera to add another face image.")
    saved = False
    idx = 2
    # Find next available filename
    while os.path.exists(f"registered_face_{idx}.jpg"):
        idx += 1
    while not saved:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(f"registered_face_{idx}.jpg", face_img)
            speak(f"Additional face image saved as registered_face_{idx}.jpg.")
            saved = True
        cv2.imshow('Add Face - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def password_fallback():
    
    if platform.system() == "Windows":
        import msvcrt
    speak("Face not recognized. Please say or type your password.")
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        print("Enter password or type 'voice' to speak password: ", end="", flush=True)
        password = ""
        if platform.system() == "Windows":
            
            start_time = time.time()
            while True:
                if msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char == '\r':
                        break
                    password += char
                if time.time() - start_time > 10:  # 10 second timeout
                    break
            password = password.strip()
        else:
            import select, sys, time
            start_time = time.time()
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 1)
                if ready:
                    password = sys.stdin.readline().strip()
                    break
                if time.time() - start_time > 10:
                    break
        if not password:
            attempts += 1
            speak("No input detected. Please try again.")
            continue
        if password.lower() == "voice":
            speak("Listening...")
            password = command()
            speak(f"You said: {password}")
        if password:
            # Replace 'Aryan' with your actual password or password check logic
            if password == "Aryan":
                speak("Access granted.")
                return True
            else:
                attempts += 1
                speak("Incorrect password. Please try again.")
        else:
            attempts += 1
            continue
    speak("Maximum attempts reached. Access denied.")
    return False

def activate_by_eyes():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if not os.path.exists("registered_eyes.jpg"):
        speak("No registered eyes found. Please register your eyes first.")
        return False
    registered_eyes = cv2.imread("registered_eyes.jpg", cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(0)
    speak("Looking for your eyes to activate Lakshmi...")
    activated = False
    start_time = time.time()
    while not activated and time.time() - start_time < 15:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue  # Skip if frame is not valid
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                (ex1, ey1, ew1, eh1) = eyes[0]
                (ex2, ey2, ew2, eh2) = eyes[1]
                eye_img = roi_gray[min(ey1,ey2):max(ey1+eh1,ey2+eh2), min(ex1,ex2):max(ex1+ew1,ex2+ew2)]
                eye_img_resized = cv2.resize(eye_img, registered_eyes.shape[::-1])
                diff = cv2.absdiff(registered_eyes, eye_img_resized)
                score = np.mean(diff)
                if score < 30:  # Threshold for matching
                    if not activated:
                        speak("Eyes recognized. Lakshmi activated.")
                    activated = True
                    break
        cv2.imshow('Eye Scan Activation - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not activated:
        speak("Eyes not recognized. Activation failed.")
    return activated


def take_photo():
    cap = cv2.VideoCapture(0)
    speak("Taking photo. Please look at the camera.")
    ret, frame = cap.read()
    if ret:
        filename = f"photo_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        speak(f"Photo saved as {filename}.")
    else:
        speak("Failed to take photo.")
    cap.release()
    cv2.destroyAllWindows()

# Record a sample of user's voice for enrollment
def enroll_voice():
   
    speak("Please say a short phrase to enroll your voice.")
    duration = 3  # seconds
    fs = 16000
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write("enroll.wav", audio, fs)
    features = extract_mfcc("enroll.wav")
    np.save(VOICEPRINT_FILE, features)
    speak("Voice enrollment complete.")

# Extract MFCC features from audio file
def extract_mfcc(filename):
    audio, fs = sf.read(filename)
    mfcc_feat = mfcc(audio, fs, numcep=13)
    return np.mean(mfcc_feat, axis=0)

# Verify if the speaker matches the enrolled voice
def verify_speaker(audio_file):
    if not os.path.exists(VOICEPRINT_FILE):
        enroll_voice()
    enrolled = np.load(VOICEPRINT_FILE)
    test = extract_mfcc(audio_file)
    similarity = 1 - cosine(enrolled, test)
    return similarity > 0.85  # Threshold for acceptance

# Modified command function with speaker verification
import tempfile

def command_verified():
    r = sr.Recognizer()
    
    
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            speak("Listening timed out. Please speak clearly and try again.")
            return ""
        except Exception as e:
            speak(f"Microphone error: {e}")
            return ""
        # Convert audio to numpy array for soundfile
        audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                sf.write(tmp.name, audio_data, 16000)
                if verify_speaker(tmp.name):
                    print("Speaker verified.")
                    try:
                        query = r.recognize_google(audio, language='en-in')
                        print(f"User said : {query}\n")
                    except Exception as e:
                        print(e)
                        speak("Sorry, I did not catch that.")
                        query = ""
                else:
                    speak("Voice not recognized. Access denied.")
                    query = ""
            except Exception as e:
                print(f"Audio processing error: {e}")
                speak("Audio error. Please try again.")
                query = ""
        os.remove(tmp.name)
    return query.lower()

def eyescan_password_fallback():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    speak("Eyes not recognized. Please scan your eyes for password.")
    cap = cv2.VideoCapture(0)
    registered_eyes = cv2.imread("registered_eyes.jpg", cv2.IMREAD_GRAYSCALE)
    start_time = time.time()
    while time.time() - start_time < 15:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray)
        if len(eyes) >= 2:
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]
            eye_img = gray[min(ey1,ey2):max(ey1+eh1,ey2+eh2), min(ex1,ex2):max(ex1+ew1,ex2+ew2)]
            eye_img_resized = cv2.resize(eye_img, registered_eyes.shape[::-1])
            diff = cv2.absdiff(registered_eyes, eye_img_resized)
            score = np.mean(diff)
            if score < 30:
                speak("Eyescan password accepted. Access granted.")
                cap.release()
                cv2.destroyAllWindows()
                return True
    cap.release()
    cv2.destroyAllWindows()
    speak("Eyescan password not recognized. Access denied.")
    return False

def open_application(app_name, search_term=None):
    # Open browser and search if app_name is a browser or 'google'
    browsers = ["chrome", "edge", "firefox", "opera", "brave", "google"]
    if app_name.lower() in browsers:
        if app_name.lower() == "google" or app_name.lower() == "chrome":
            url = f"https://www.google.com"
            if not search_term:
                speak("Boss, what should I search on Google?")
                search_term = input("Enter your Google search query: ").strip()
            if search_term:
                url = f"https://www.google.com/search?q={search_term}"
            webbrowser.open(url)
            speak(f"Opening Google and searching for {search_term if search_term else 'Google'}")
            return True
        elif app_name.lower() == "edge":
            edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            if os.path.exists(edge_path):
                os.startfile(edge_path)
                speak("Opening Microsoft Edge")
                speak("Boss, what should I search on Edge?")
                search_term = input("Enter your Edge search query: ").strip()
                if search_term:
                    webbrowser.open(f"https://www.bing.com/search?q={search_term}")
                    speak(f"Searching Edge for {search_term}")
                return True
            else:
                speak("Microsoft Edge not found.")
                return False
        else:
            webbrowser.open(f"https://www.{app_name.lower()}.com")
            speak(f"Opening {app_name}")
            speak(f"Boss, what should I search on {app_name}?")
            search_term = input(f"Enter your {app_name} search query: ").strip()
            if search_term:
                webbrowser.open(f"https://www.{app_name.lower()}.com/search?q={search_term}")
                speak(f"Searching {app_name} for {search_term}")
            return True
    # Try to open application by name
    try:
        os.startfile(app_name)
        speak(f"Opening {app_name}")
        return True
    except Exception:
        possible_paths = [
            f"C:\\Program Files\\{app_name}\\{app_name}.exe",
            f"C:\\Program Files (x86)\\{app_name}\\{app_name}.exe",
            f"C:\\Windows\\System32\\{app_name}.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.startfile(path)
                speak(f"Opening {app_name}")
                return True
        speak(f"Could not find {app_name} on your device.")
        return False


app = Flask(__name__)
CORS(app)

# NLP: spaCy for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None

def extract_entities(text):
    if nlp is None:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Sentiment analysis
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    sentiment_analyzer = None

def analyze_sentiment(text):
    if sentiment_analyzer is None:
        return []
    return sentiment_analyzer(text)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()
    if 'goodbye lakshmi' in user_message or 'bye' in user_message:
        response = "Have a good day, bye sir."
        threading.Thread(target=shutdown_server).start()
        return jsonify({'response': response, 'shutdown': True})

    # Social media query
    if any(social in user_message for social in ["facebook", "instagram", "whatsapp", "discord"]):
        social_media(user_message)
        return jsonify({'response': f"Opening {user_message.split()[1]} for you."})

    # Open app query
    if user_message.startswith("open "):
        app_name = user_message.replace("open ", "").strip()
        open_application(app_name)
        return jsonify({'response': f"Opening {app_name} for you."})

    # Known app commands
    known_apps = ["calculator", "notepad", "paint", "word", "excel", "edge", "chrome", "firefox"]
    for app in known_apps:
        if app in user_message:
            open_application(app)
            return jsonify({'response': f"Opening {app} for you."})

    # Fallback: NLP
    entities = extract_entities(user_message)
    sentiment = analyze_sentiment(user_message)
    response = f"Entities: {entities}, Sentiment: {sentiment}"
    return jsonify({'response': response})

if __name__ == '__main__':
    register_face_once()
    activated = activate_by_eyes()
    if not activated:
        speak("Eyes not recognized. Please enter your password in text or say 'voice' to use voice mode.")
        if not password_fallback():
            speak("Access denied. Exiting.")
            exit()
    speak("Lakshmi is ready.")
    input_mode = "voice"
    while True:
        query = ""
        mode_switch = None
        voice_query = command()
        if voice_query in ["text", "voice"]:
            mode_switch = voice_query
        elif voice_query:
            query = voice_query
        # Check for text command (non-blocking, always)
        print("Type your command (or press Enter to skip): ", end="", flush=True)
        text_query = ""
        
        if platform.system() == "Windows":
            import msvcrt
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                while char != '\r':
                    text_query += char
                    char = msvcrt.getwch()
                text_query = text_query.strip().lower()
        else:
            
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if ready:
                text_query = sys.stdin.readline().strip().lower()
        if text_query in ["text", "voice"]:
            mode_switch = text_query
        elif text_query:
            query = text_query
        # Handle mode switch immediately
        if mode_switch:
            input_mode = mode_switch
            speak(f"{input_mode.capitalize()} mode activated.")
            continue
        # If both are empty, continue
        if not query:
            continue
        # Command handling
        if "take photo" in query:
            take_photo()
            continue
        # Deactivation
        if "good bye lakshmi" in query or "exit" in query or "off" in query or "bye" in query:
            speak("Goodbye! Turning off.")
            sys.exit()

            # Close all opened apps
            if "lakshmi close" in query:
            
              continue

            if ("facebook" in query) or ("instagram" in query) or ("whatsapp" in query) or ("discord" in query):
                social_media(query)
            elif ("university timetable" in query) or ("schedule" in query):
                schedule()
            elif ("volume up" in query):
                pyautogui.press("volumeup")
                speak("volume increased")
            elif ("volume down" in query):
                pyautogui.press("volumedown")
                speak("volume decreased")
            elif ("volume mute" in query):
                pyautogui.press("volumemute")
                speak("volume muted")
            elif ("open google" in query) or ("open edge" in query):
                browsing(query)
            elif "joke" in query:
                tell_joke()
            elif "wikipedia" in query:
                topic = query.replace("wikipedia","").strip()
                if topic:
                    search_wikipedia(topic)
                else:
                    wikipedia_voice_search()
            elif "weather" in query:
                speak("Which city?")
                city = input("Enter city name: ").strip()
                if city:
                    get_weather(city)
            elif "call" in query:
                # ...existing call code...
                pass
            elif "lakshmi exit" in query or "lakshmi off" in query or "lakshmi bye" in query:
                speak("Goodbye! Turning off.")
                sys.exit()
        # Enhanced: open app/browser and search
        if query.startswith("open "):
            app_name = query.replace("open ", "").strip()
            open_application(app_name)
            continue
        elif query.startswith("search "):
            search_term = query.replace("search ", "").strip()
            open_application("google", search_term)
            continue
        elif "google" in query:
            # If user says 'google' and a search term
            parts = query.split("google")
            search_term = parts[1].strip() if len(parts) > 1 else ""
            open_application("google", search_term)
            continue
        # Known app commands
        known_apps = ["calculator", "notepad", "paint", "word", "excel", "edge", "chrome", "firefox"]
        for app in known_apps:
            if app in query:
                open_application(app)
                continue
        # ...existing command handling code...
        # Command handling
        if ("facebook" in query) or ("instagram" in query) or ("whatsapp" in query) or ("discord" in query):
            social_media(query)
        elif ("university timetable" in query) or ("schedule" in query):
            schedule()
        elif ("volume up" in query):
            pyautogui.press("volumeup")
            speak("volume increased")
        elif ("volume down" in query):
            pyautogui.press("volumedown")
            speak("volume decreased")
        elif ("volume mute" in query):
            pyautogui.press("volumemute")
            speak("volume muted")
        elif ("open google" in query) or ("open edge" in query):
            browsing(query)
        elif "joke" in query:
            tell_joke()
        elif "wikipedia" in query:
            topic = query.replace("wikipedia","").strip()
            if topic:
                search_wikipedia(topic)
            else:
                wikipedia_voice_search()
        elif "weather" in query:
            speak("Which city?")
            city = input("Enter city name: ").strip()
            if city:
                get_weather(city)
        elif "lakshmi exit" in query or "lakshmi off" in query or "lakshmi bye" in query  or "Goodbye lakshmi" :
            speak("Goodbye! Turning off.")
            sys.exit()
        else:
            get_response_from_intents(query)
