from flask import Flask, render_template, Response
from imutils.video import VideoStream
import cv2
import imutils
import numpy as np
import pyaudio
import librosa
from tensorflow.keras.models import load_model, model_from_json
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
from PIL import Image, ImageTk

from imutils import face_utils
from threading import Thread
import argparse
import time
import dlib
import cv2



app = Flask(__name__)

def ExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model


# Load pre-trained facial emotion recognition model
facial_emotion_model = ExpressionModel('model_optimal_facial_emotion.json','model_weights_optimal_facial_emotion.h5')

# Load pre-trained speech emotion recognition model
speech_emotion_model = ExpressionModel("model_voice_emotion.json","model_voice_emotion.h5")


# Create VideoCapture object (0 is the default camera)
cap = cv2.VideoCapture(0)

# facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}
EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def recognize_facial_emotion(frame, panel):
    # Preprocess frame for facial emotion recognition
    # ...
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,3)
    for x,y,w,h in faces:
        subface = gray[y:y+h,x:x+w]
        resized_img = cv2.resize(subface,(48,48))
        normalize_img = resized_img/255.0
        reshaped_img = np.reshape(normalize_img,(1,48,48,1))
        result = facial_emotion_model.predict(reshaped_img)
        label = np.argmax(result,axis=1)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255), 2)
        cv2.rectangle(frame,(x,y-50),(x+w,y),(50,50,255), -1)
        cv2.putText(frame,labels[label],(x,y-10),cv2.FONT_ITALIC,0.8,(255,))
        # update_panel(frame, panel)
        cv2.imshow("Emotion Detection",frame)
        return labels[label]

def update_panel(frame, panel):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    
    



# # Function to perform speech emotion recognition on an audio clip
def recognize_speech_emotion(audio_clip,sr):
    features = get_features(audio_clip,sr)
    features = np.expand_dims(features, axis=-1)  # Add batch dimension
    emotion_probabilities = speech_emotion_model.predict(features)
    emotion_index = np.argmax(emotion_probabilities)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral","Surprise", "Sad","Calm"]
    if 0 <= emotion_index < len(emotions):
        predicted_emotion = emotions[emotion_index]
    else:
        # Handle the case where emotion_index is out of range
        predicted_emotion = "Unknown"
    # predicted_emotion = emotions[emotion_index]
    return predicted_emotion




#data augmentation 
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data,sr=sampling_rate,n_steps=pitch_factor)


def extract_features(data,sample_rate):
    
    audio_data_float = librosa.util.normalize(data.astype(np.float32))
    # ZCR
    
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_data_float).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(audio_data_float))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data_float, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    
    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=audio_data_float).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio_data_float, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(data,sample_rate):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    # data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    data_float = librosa.util.normalize(data.astype(float))
    # without augmentation
    res1 = extract_features(data_float,sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data_float)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
       
    # data with stretching and pitching
    new_data = stretch(data_float)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())


# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.68

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)


def mouth_state_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
    rects = detector(gray, 0)

	# loop over the face detections
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extract the mouth coordinates, then use the
# coordinates to compute the mouth aspect ratio
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
        mouthHull = cv2.convexHull(mouth)
		
        # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:cv2.putText(frame, "Mouth is Open!", (30,45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
    # out.write(frame)
	# show the frame
    cv2.imshow("Emotion Detection", frame)





@app.route('/')
def index():
    return render_template('template/index.html')

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform facial emotion recognition
        facial_emotion = recognize_facial_emotion(frame, panel)
        print("Facial Emotion:", facial_emotion)

        # Perform mouth state detection
        mouth_state_detection(frame)

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 8025

        # Record audio for speech emotion recognition
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Perform speech emotion recognition
        speech_emotion = recognize_speech_emotion(audio_data, RATE)
        print("Speech Emotion:", speech_emotion)

        # Update speech emotion label
        speech_label_text = f"Speech Emotion: {speech_emotion}"

        # Display the resulting frame
        cv2.putText(frame, f"Speech: {speech_emotion}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Emotion Detection", frame)

        # Convert frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Create VideoCapture object (0 is the default camera)
    cap = cv2.VideoCapture(0)

    # Load pre-trained facial emotion recognition model
    facial_emotion_model = ExpressionModel('model_optimal_facial_emotion.json', 'model_weights_optimal_facial_emotion.h5')

    # Load pre-trained speech emotion recognition model
    speech_emotion_model = ExpressionModel("model_voice_emotion.json", "model_voice_emotion.h5")

    # Create PyAudio object
    p = pyaudio.PyAudio()

    # Create PyAudio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=8025,
                    input=True,
                    frames_per_buffer=1024)

    app.run(debug=True)
