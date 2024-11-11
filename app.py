# from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
# import random
# import datetime
# from fpdf import FPDF
# import cv2
# import numpy as np
# import torch
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models
# emotion_model = load_model('5_30AMmodel.h5')  # Emotion detection model
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses and Big 5 scores
# flagged_responses = []
# big5_scores = {}

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test')
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Collect answers from form and calculate scores
#     big5_scores['openness'] = (int(request.form['q1']) + int(request.form['q2']))  # Sum for Openness
#     big5_scores['conscientiousness'] = (int(request.form['q3']) + int(request.form['q4']))/2  # Sum for Conscientiousness
#     big5_scores['extraversion'] = (int(request.form['q5']) + int(request.form['q6']))/2  # Sum for Extraversion
#     big5_scores['agreeableness'] = (int(request.form['q7']) + int(request.form['q8']))/2  # Sum for Agreeableness
#     big5_scores['neuroticism'] = (int(request.form['q9']) + int(request.form['q10']))/2  # Sum for Neuroticism
    
#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, "Neutral")

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate PDF report of flagged responses and Big 5 scores
# def generate_report():
#     report_filename = "suicidal_intent_report.pdf"
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="### Suicidal Intent & Personality Report ###", ln=True, align='C')
#     pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    
#     pdf.cell(200, 10, txt="\nBig 5 Personality Scores:", ln=True)
#     for trait, score in big5_scores.items():
#         pdf.cell(200, 10, txt=f"{trait.capitalize()}: {score}", ln=True)
    
#     pdf.cell(200, 10, txt="\nFlagged Messages for Suicidal Intent:", ln=True)
#     for idx, message in enumerate(flagged_responses, 1):
#         pdf.cell(200, 10, txt=f"{idx}. {message}", ln=True)
    
#     pdf.output(report_filename)
#     return report_filename

# # Download report
# @app.route('/download_report')
# def download_report():
#     report_file = generate_report()
#     return send_file(report_file, as_attachment=True)

# # Chatbot Route
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_message = request.json['message'].strip()

#     # Check if the user inputs 'exit' to end the conversation
#     if user_message.lower() == "exit":
#         return jsonify({'reply': "Thank you for chatting with me. Your report has been generated. <a href='/download_report'>Download Report</a>"})

#     # Detect suicidal intent
#     is_suicidal = detect_suicidal_intent(user_message)
    
#     if is_suicidal:
#         reply = random.choice([  # Responses for suicidal intent
#             "Are you okay? How long have you been feeling this way?",
#             "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
#         ])
#     else:
#         inputs = chatbot_tokenizer([user_message], return_tensors="pt")
#         reply_ids = chatbot_model.generate(**inputs)
#         reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
#     return jsonify({'reply': reply})

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)





# from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
# import random
# import datetime
# from fpdf import FPDF
# import cv2
# import numpy as np
# import torch
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import os

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models
# emotion_model = load_model('5_30AMmodel.h5')  # Emotion detection model
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Track flagged responses, Big 5 scores, and emotion counters
# flagged_responses = []
# big5_scores = {}
# total_frames = 0
# sad_emotion_count = 0

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test')
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Collect answers from form and calculate scores
#     big5_scores['openness'] = (int(request.form['q1']) + int(request.form['q2']))  # Sum for Openness
#     big5_scores['conscientiousness'] = (int(request.form['q3']) + int(request.form['q4']))/2  # Sum for Conscientiousness
#     big5_scores['extraversion'] = (int(request.form['q5']) + int(request.form['q6']))/2  # Sum for Extraversion
#     big5_scores['agreeableness'] = (int(request.form['q7']) + int(request.form['q8']))/2  # Sum for Agreeableness
#     big5_scores['neuroticism'] = (int(request.form['q9']) + int(request.form['q10']))/2  # Sum for Neuroticism
    
#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model, total_frames, sad_emotion_count
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, "Neutral")

#             if emotion_label == "Sad":
#                 sad_emotion_count += 1
#             total_frames += 1

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection
# def detect_suicidal_intent(user_message):
#     global flagged_responses
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]
#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate PDF report of flagged responses, Big 5 scores, and emotion analysis
# def generate_report():
#     global sad_emotion_count, total_frames
#     report_filename = "suicidal_intent_report.pdf"
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="### Suicidal Intent & Personality Report ###", ln=True, align='C')
#     pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    
#     pdf.cell(200, 10, txt="\nBig 5 Personality Scores:", ln=True)
#     for trait, score in big5_scores.items():
#         pdf.cell(200, 10, txt=f"{trait.capitalize()}: {score}", ln=True)
    
#     pdf.cell(200, 10, txt="\nFlagged Messages for Suicidal Intent:", ln=True)
#     for idx, message in enumerate(flagged_responses, 1):
#         pdf.cell(200, 10, txt=f"{idx}. {message}", ln=True)
    
#     # Add average sad emotion percentage
#     if total_frames > 0:
#         sad_emotion_avg = (sad_emotion_count / total_frames) * 100
#         pdf.cell(200, 10, txt=f"\nSad Emotion Frequency: {sad_emotion_avg:.2f}%", ln=True)
#     else:
#         pdf.cell(200, 10, txt="\nSad Emotion Frequency: N/A (No frames detected)", ln=True)
    
#     pdf.output(report_filename)
#     return report_filename

# # Download report
# @app.route('/download_report')
# def download_report():
#     report_file = generate_report()
#     return send_file(report_file, as_attachment=True)

# # Chatbot Route
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_message = request.json['message'].strip()

#     # Check if the user inputs 'exit' to end the conversation
#     if user_message.lower() == "exit":
#         return jsonify({'reply': "Thank you for chatting with me. Your report has been generated. <a href='/download_report'>Download Report</a>"})

#     # Detect suicidal intent
#     is_suicidal = detect_suicidal_intent(user_message)
    
#     if is_suicidal:
#         reply = random.choice([  # Responses for suicidal intent
#             "Are you okay? How long have you been feeling this way?",
#             "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
#         ])
#     else:
#         inputs = chatbot_tokenizer([user_message], return_tensors="pt")
#         reply_ids = chatbot_model.generate(**inputs)
#         reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
#     return jsonify({'reply': reply})

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)




# from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
# import random
# import datetime
# from fpdf import FPDF
# import cv2
# import numpy as np
# import torch
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
# from tensorflow.keras.models import load_model
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# import os

# app = Flask(__name__)

# # Temporary in-memory user database for login
# users = {"test@example.com": "password123"}

# # Load models
# emotion_model = load_model('5_30AMmodel.h5')  # Emotion detection model
# chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
# suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
# suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# # Initialize VADER sentiment analyzer
# nltk.download('vader_lexicon')
# sia = SentimentIntensityAnalyzer()

# # Track flagged responses, Big 5 scores, and emotion counters
# flagged_responses = []
# big5_scores = {}
# sentiment_scores = []
# total_frames = 0
# sad_emotion_count = 0

# # Route for the login page
# @app.route('/')
# def login():
#     return render_template('login.html')

# # Login authentication
# @app.route('/login', methods=['POST'])
# def handle_login():
#     email = request.form['email']
#     password = request.form['password']
    
#     if email in users and users[email] == password:
#         return redirect(url_for('big5_test'))
#     else:
#         return "Invalid credentials, try again."

# # Big 5 Personality Test page
# @app.route('/big5_test')
# def big5_test():
#     return render_template('big5.html')

# # Handle Big 5 Test submission and redirect to the main app
# @app.route('/submit_test', methods=['POST'])
# def handle_test():
#     # Collect answers from form and calculate scores
#     big5_scores['openness'] = (int(request.form['q1']) + int(request.form['q2']))/2  # Sum for Openness
#     big5_scores['conscientiousness'] = (int(request.form['q3']) + int(request.form['q4']))/2  # Sum for Conscientiousness
#     big5_scores['extraversion'] = (int(request.form['q5']) + int(request.form['q6']))/2  # Sum for Extraversion
#     big5_scores['agreeableness'] = (int(request.form['q7']) + int(request.form['q8']))/2  # Sum for Agreeableness
#     big5_scores['neuroticism'] = (int(request.form['q9']) + int(request.form['q10']))/2  # Sum for Neuroticism
    
#     return redirect(url_for('main_page'))

# # Main page with video and chatbot
# @app.route('/main_page')
# def main_page():
#     return render_template('index.html')

# # Emotion Detection: Generate frames from webcam
# def generate_camera_feed():
#     global emotion_model, total_frames, sad_emotion_count
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 7)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = cv2.resize(roi_gray, (48, 48))
#             cropped_img = cropped_img.astype('float32') / 255
#             cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

#             # Predict emotion
#             prediction = emotion_model.predict(cropped_img)
#             max_index = int(np.argmax(prediction))
#             emotion_label = emotion_dict.get(max_index, "Neutral")

#             if emotion_label == "Sad":
#                 sad_emotion_count += 1
#             total_frames += 1

#             cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Suicidal Intent Detection and VADER Sentiment Analysis
# def detect_suicidal_intent(user_message):
#     global flagged_responses, sentiment_scores

#     # Detect suicidal intent with the model
#     inputs = suicide_tokenizer(user_message, return_tensors="pt")
#     logits = suicide_model(**inputs).logits
#     prediction = torch.softmax(logits, dim=1).tolist()[0]

#     # VADER sentiment analysis
#     sentiment = sia.polarity_scores(user_message)
#     sentiment_scores.append(sentiment['compound'])

#     if prediction[1] > 0.5:  # If the message is flagged as suicidal
#         flagged_responses.append(user_message)
#         return True
#     return False

# # Generate PDF report of flagged responses, Big 5 scores, and emotion analysis
# def generate_report():
#     global sad_emotion_count, total_frames, sentiment_scores
#     report_filename = "suicidal_intent_report.pdf"
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="### Suicidal Intent & Personality Report ###", ln=True, align='C')
#     pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    
#     pdf.cell(200, 10, txt="\nBig 5 Personality Scores:", ln=True)
#     for trait, score in big5_scores.items():
#         pdf.cell(200, 10, txt=f"{trait.capitalize()}: {score}", ln=True)
    
#     pdf.cell(200, 10, txt="\nFlagged Messages for Suicidal Intent:", ln=True)
#     for idx, message in enumerate(flagged_responses, 1):
#         pdf.cell(200, 10, txt=f"{idx}. {message}", ln=True)
    
#     # Add average sad emotion percentage
#     if total_frames > 0:
#         sad_emotion_avg = (sad_emotion_count / total_frames) * 100
#         pdf.cell(200, 10, txt=f"\nSad Emotion Frequency: {sad_emotion_avg:.2f}%", ln=True)
#     else:
#         pdf.cell(200, 10, txt="\nSad Emotion Frequency: N/A (No frames detected)", ln=True)

#     # Average sentiment score from VADER
#     if sentiment_scores:
#         avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
#         pdf.cell(200, 10, txt=f"\nAverage Sentiment Score: {avg_sentiment:.2f}", ln=True)
#     else:
#         pdf.cell(200, 10, txt="\nAverage Sentiment Score: N/A (No messages analyzed)", ln=True)
    
#     pdf.output(report_filename)
#     return report_filename

# # Download report
# @app.route('/download_report')
# def download_report():
#     report_file = generate_report()
#     return send_file(report_file, as_attachment=True)

# # Chatbot Route
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_message = request.json['message'].strip()

#     # Check if the user inputs 'exit' to end the conversation
#     if user_message.lower() == "exit":
#         return jsonify({'reply': "Thank you for chatting with me. Your report has been generated. <a href='/download_report'>Download Report</a>"})

#     # Detect suicidal intent
#     is_suicidal = detect_suicidal_intent(user_message)
    
#     if is_suicidal:
#         reply = random.choice([  # Responses for suicidal intent
#             "Are you okay? How long have you been feeling this way?",
#             "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
#         ])
#     else:
#         inputs = chatbot_tokenizer([user_message], return_tensors="pt")
#         reply_ids = chatbot_model.generate(**inputs)
#         reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
#     return jsonify({'reply': reply})

# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)


from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
import random
import datetime
from fpdf import FPDF
import cv2
import numpy as np
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
import requests
import os

app = Flask(__name__)

# Temporary in-memory user database for login
users = {"test@example.com": "password123"}

# Load models
emotion_model = load_model('5_30AMmodel.h5')  # Emotion detection model
chatbot_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
chatbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
suicide_tokenizer = AutoTokenizer.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")
suicide_model = AutoModelForSequenceClassification.from_pretrained("C:/Users/heema/Desktop/Capstone_Project_Suicidal_intent/suicidal-text-detection/Models/electra")

# Track flagged responses, Big 5 scores, and emotion counters
flagged_responses = []
big5_scores = {}
total_frames = 0
sad_emotion_count = 0

# Telegram bot setup
TELEGRAM_BOT_TOKEN = "7767750634:AAGA4S5rKI7eu9hljGzLHJg39NpSKNI4LbE"
TELEGRAM_CHAT_ID = "6209014747"  # Replace with the recipient's chat ID

# Route for the login page
@app.route('/')
def login():
    return render_template('login.html')

# Login authentication
@app.route('/login', methods=['POST'])
def handle_login():
    email = request.form['email']
    password = request.form['password']
    
    if email in users and users[email] == password:
        return redirect(url_for('big5_test'))
    else:
        return "Invalid credentials, try again."

# Big 5 Personality Test page
@app.route('/big5_test')
def big5_test():
    return render_template('big5.html')

# Handle Big 5 Test submission and redirect to the main app
@app.route('/submit_test', methods=['POST'])
def handle_test():
    # Collect answers from form and calculate scores
    big5_scores['openness'] = (int(request.form['q1']) + int(request.form['q2']))  # Sum for Openness
    big5_scores['conscientiousness'] = (int(request.form['q3']) + int(request.form['q4']))/2  # Sum for Conscientiousness
    big5_scores['extraversion'] = (int(request.form['q5']) + int(request.form['q6']))/2  # Sum for Extraversion
    big5_scores['agreeableness'] = (int(request.form['q7']) + int(request.form['q8']))/2  # Sum for Agreeableness
    big5_scores['neuroticism'] = (int(request.form['q9']) + int(request.form['q10']))/2  # Sum for Neuroticism
    
    return redirect(url_for('main_page'))

# Main page with video and chatbot
@app.route('/main_page')
def main_page():
    return render_template('index.html')

# Emotion Detection: Generate frames from webcam
def generate_camera_feed():
    global emotion_model, total_frames, sad_emotion_count
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Surprise'}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype('float32') / 255
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

            # Predict emotion
            prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            emotion_label = emotion_dict.get(max_index, "Neutral")

            if emotion_label == "Sad":
                sad_emotion_count += 1
            total_frames += 1

            cv2.putText(frame, emotion_label, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Suicidal Intent Detection
def detect_suicidal_intent(user_message):
    global flagged_responses
    inputs = suicide_tokenizer(user_message, return_tensors="pt")
    logits = suicide_model(**inputs).logits
    prediction = torch.softmax(logits, dim=1).tolist()[0]
    if prediction[1] > 0.5:  # If the message is flagged as suicidal
        flagged_responses.append(user_message)
        return True
    return False

# Generate PDF report of flagged responses, Big 5 scores, and emotion analysis
def generate_report():
    global sad_emotion_count, total_frames
    report_filename = "suicidal_intent_report.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="### Suicidal Intent & Personality Report ###", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    
    pdf.cell(200, 10, txt="\nBig 5 Personality Scores:", ln=True)
    for trait, score in big5_scores.items():
        pdf.cell(200, 10, txt=f"{trait.capitalize()}: {score}", ln=True)
    
    pdf.cell(200, 10, txt="\nFlagged Messages for Suicidal Intent:", ln=True)
    for idx, message in enumerate(flagged_responses, 1):
        pdf.cell(200, 10, txt=f"{idx}. {message}", ln=True)
    
    # Add average sad emotion percentage
    if total_frames > 0:
        sad_emotion_avg = (sad_emotion_count / total_frames) * 100
        pdf.cell(200, 10, txt=f"\nSad Emotion Frequency: {sad_emotion_avg:.2f}%", ln=True)
    else:
        pdf.cell(200, 10, txt="\nSad Emotion Frequency: N/A (No frames detected)", ln=True)
    
    pdf.output(report_filename)
    return report_filename

# Download report
@app.route('/download_report')
def download_report():
    report_file = generate_report()
    return send_file(report_file, as_attachment=True)

# Chatbot Route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json['message'].strip()

    # Check if the user inputs 'exit' to end the conversation
    if user_message.lower() == "exit":
        # Send report to Telegram
        send_report_to_telegram()
        return jsonify({'reply': "Thank you for chatting with me. Your report has been sent. Check your Telegram."})

    # Detect suicidal intent
    is_suicidal = detect_suicidal_intent(user_message)
    
    if is_suicidal:
        reply = random.choice([  # Responses for suicidal intent
            "Are you okay? How long have you been feeling this way?",
            "That sounds so painful, and I appreciate you sharing that with me. How can I help?"
        ])
    else:
        inputs = chatbot_tokenizer([user_message], return_tensors="pt")
        reply_ids = chatbot_model.generate(**inputs)
        reply = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    
    return jsonify({'reply': reply})

# Send the generated report to the Telegram bot
def send_report_to_telegram():
    report_file = generate_report()

    # Sending the report to the Telegram bot
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    files = {'document': open(report_file, 'rb')}
    data = {'chat_id': TELEGRAM_CHAT_ID}
    response = requests.post(url, data=data, files=files)
    
    if response.status_code == 200:
        print("Report sent to Telegram.")
    else:
        print("Failed to send report to Telegram.")

if __name__ == "__main__":
    app.run(debug=True)
