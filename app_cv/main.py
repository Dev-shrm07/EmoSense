from flask import Flask, render_template, request, flash, Response
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras import models

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

cap = cv2.VideoCapture(0)
new_model =  models.load_model("emotion_detector.h5")

def gen_frames():
    path = "haarcascade_frontalface_default.xml"
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    rectangle_bgr = (255, 255, 255)
    img= np.zeros((500, 500))
    text= "Some text in a box!"
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1) [0]
    text_offset_x = 10
    text_offset_y=img.shape [0] - 25
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords [0], box_coords [1], rectangle_bgr, cv2.FILLED)
    cv2.putText (img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:

            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale (gray,1.004,5)
            for x,y,w,h in faces:
                roi_gray =  gray[y:y+h, x:x+w]
                roi_color= frame [y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                facess = faceCascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey: ey+eh, ex:ex + ew] ## cropping the face
                    final_image =cv2.resize(face_roi, (48,48))
                    final_image= np.expand_dims (final_image, axis =0) ## need fourth dimension
                    final_image=final_image/255.0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    p = new_model.predict(final_image)
                    font_scale = 1.5
                    font = cv2.FONT_HERSHEY_PLAIN
                    count = np.argmax(p[0])
                    status = classes[count]
                    x1, y1,w1, h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1, x1), (x1+w1, y1 + h1), (0,0,0), -1)
                    cv2.putText (frame, status, (x1+ int(w1/10),y1 + int (h1/2)), cv2. FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText (frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
                    
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filenamex):
    img = cv2.imread(f"uploads/{filenamex}")
    model = models.load_model("emotion_detector.h5")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    last_face_coords = None
    for x,y,w,h in faces:
        roi_gray =gray[y:y+h, x:x+w]
        roi_color= img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faces =faceCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.004,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in faces:
                last_face_coords = (x,y,ex,ey)
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]
    
    final_image =cv2.resize(face_roi, (48, 48)) 
    final_image= np.expand_dims (final_image, axis =0)
    final_image=final_image/255.0
    pred = model.predict(final_image)
    count = np.argmax(pred[0])
    if last_face_coords is not None:
        label = classes[count]
        x, y, w, h = last_face_coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_x = x
        label_y = y - 10
        cv2.rectangle(img, (label_x - 5, label_y - label_height - 5),
                    (label_x + label_width + 5, label_y + 5), (255, 0, 0), -1)
        cv2.putText(img, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(f"static/{filenamex}", img)
    

    



    


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/edit', methods=['GET', 'POST'])
def success():
    if 'file' not in request.files:
        flash('No file part')
        return "error"
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    #empty file without a filename.
    if file.filename == '':
        #flash('No selected file')
        return "error no selected file" 
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save (os.path.join(app.config['UPLOAD_FOLDER'], filename))
        process_image(filename)
        flash(f"Your Image have been processed and is available <a href ='/static/{filename}'  target = '_blank'>Here</a> ")
        return render_template("upload.html")
    return render_template("upload.html")



@app.route("/upload")
def uploadx():
    return render_template("upload.html")

@app.route("/live")
def livex():
    return render_template("live.html")

@app.route('/live/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
app.run(debug = True)