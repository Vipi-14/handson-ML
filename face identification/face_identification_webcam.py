import face_recognition
import cv2,os
import numpy as np

path = os.getcwd()

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture('WhatsApp Video 2024-05-02 at 3.26.26 PM.mp4') 

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(video_capture.get(3)), int(video_capture.get(4))))

# Create arrays of known face encodings and their names
known_face_encodings = [
    
]

dbimgs = os.listdir(os.path.join(path,'db'))
dbimgs.sort()

for dbimg in dbimgs:
    # Load a sample picture and learn how to recognize it.
    image = face_recognition.load_image_file(f"db/{dbimg}")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)


known_face_names = [
    "biden",
    "trump"
]

known_face_names.sort()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
cnt = 0
while True:
    cnt+=1
    # Grab a single frame of video
    ret, frame = video_capture.read()

   
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.rectangle(frame, (left, top - text_size[1]), (left + text_size[0], top),  (0, 0, 255), -1)
        cv2.putText(frame, name, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    

    # Display the resulting image
    # cv2.imshow('Video', frame)
    path = os.getcwd()
    out_dir = os.path.join(path,'Output')
    os.makedirs(out_dir,exist_ok=True)
    cv2.imwrite(os.path.join(out_dir,f'{cnt}.jpg'),frame)
    print(f'{cnt}.jpg')


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()