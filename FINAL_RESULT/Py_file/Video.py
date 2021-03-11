def Video_emotion_record():
    from keras.models import load_model
    import pandas as pd
    import cv2
    import numpy as np   
    from keras.preprocessing.image import img_to_array
    import pyaudio
    import wave
    
    face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)
    EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]
    
    
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


    camera = cv2.VideoCapture(0)
    num=0
    emotion_list={}
    frames=[]
    
    while True:
        # Capture image from camera
        ret, frame = camera.read()

        # Convert color to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection in frame
        faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30,30))

        # Create empty image
        canvas = np.zeros((250, 300, 3), dtype="uint8")

        # Perform emotion recognition only when face is detected
        if len(faces) > 0:
            # For the largest image
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Emotion predict
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # Assign labeling
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)


            lis=[]
            # Label printing
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)    
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


                lis.append(prob)
                
            for i in range(0, int(RATE / CHUNK)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            #emotion 확률 저장 / fram - record_images 파일에 저장
            cv2.imwrite("record_images/frame%d.jpg" % num, frame)
            emotion_list[num]=lis
            num+=1

        # Open two windows
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
        cv2.imshow('Emotion Recognition', frame)
        cv2.imshow("Probabilities", canvas)

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            break

    # Clear program and close windows
    camera.release()
    cv2.destroyAllWindows()
    
    df=pd.DataFrame(emotion_list,index=EMOTIONS)
   
    return df.mean(axis=1)