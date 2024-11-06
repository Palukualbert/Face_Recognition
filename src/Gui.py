import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np

window = tk.Tk()
window.title("Face Recognition system by PLK")
 
l1 = tk.Label(window, text="Name", font=("Times new roman",20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)
 
l2 = tk.Label(window, text="Age", font=("Times new roman",20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)
 
l3 = tk.Label(window, text="Address", font=("Times new roman",20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

#TRAINING DATASET


def train_classifier():
    data_dir = "C:/Users/alber/Documents/face_recognition/src/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
     
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
         
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result',' training dataset completed...')

b1 = tk.Button(window, text="Training", font=("Times new roman",20),bg="red",fg="white",command=train_classifier)
b1.grid(column=0, row=4)

#DETECTION FACES
def detect_faces():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        
        coords=[]
        
        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
            
            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))
            
            if confidence>80:
                if id==1:
                    cv2.putText(img, "alba", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id==2:
                    cv2.putText(img, "christelle", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id==3:
                    cv2.putText(img, "jeanne", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
            coords= [x,y,w,h]
        return coords
    
    def recognize (img,clf,faceCascade):
    # loading classifier
        coords= draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face Detection by PLK", img)
        
        if cv2.waitKey(1)==13:
            break
        # Quitte si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()
    messagebox.showinfo('Result','face detected')
b2 = tk.Button(window, text="Detect the faces", font=("Times new roman",20), bg="green", fg="white", command=detect_faces)
b2.grid(column=1, row=4)

#GENERATING DATASET

def generate_dataset():
    if(t1.get()=="" or t2.get()=="" or t3.get()==""):
        messagebox.showinfo('Result','Please complete all details needed...')
    else:
        # Provide the correct path to the Haar cascade file
        cascade_path = "haarcascade_frontalface_default.xml"
        face_classifier = cv2.CascadeClassifier(cascade_path)

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # scaling factor = 1.3
            # minimum neighbor = 5

            if faces == ():
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
                return cropped_face
            return None

        cap = cv2.VideoCapture(0)  # Use the default camera
        id = 1
        img_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)

            if cv2.waitKey(1) == 13 or img_id == 200:  # 13 is the ASCII character for Enter
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result','Generating dataset completed!!!')
        

b3 = tk.Button(window, text="Generate dataset", font=("Times new roman",20), bg="blue", fg="white", command=generate_dataset)
b3.grid(column=2, row=4)
 
window.geometry("700x200")
window.mainloop()
 
#other just copy code from previous part same like as I have done in this video
