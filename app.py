from flask import Flask,render_template,jsonify,request
import os
import shutil
import speech_recognition as sr
import VOICE_AI



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('mainpage.html')

@app.route('/tic_tac_toe.html')
def tixtactoe():
    return render_template('tic_tac_toe.html')

@app.route('/flipcard.html')
def flipcards():
    return render_template('flipcard.html')

@app.route('/games.html')
def quiz():
    return render_template('games.html')

UPLOAD_DIR = "static/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/bear.html')
def bear():
    return render_template("bear.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']
    audio_path = os.path.join(UPLOAD_DIR, 'uploaded.wav')
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except Exception as e:
            text = "Could not recognize speech."

    # Simulated response (use TTS here if needed)
    response_audio = os.path.join(UPLOAD_DIR, 'response.wav')
    voiceprocess=VOICE_AI.content_model(text)  # Call the content model to get a response

    


    if voiceprocess:
        return jsonify({
        "text": text,
        "audio_url": f"/static/audio/response.wav"})

@app.route('/tell-story')
def tell_story():

    return jsonify({
        "audio_url": f"/static/audio/story.wav"
    })

if __name__== '__main__':
    app.run(debug=True)