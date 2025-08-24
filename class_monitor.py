# import cv2
# from deepface import DeepFace

# # Load face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start capturing video
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Convert frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Convert grayscale frame to RGB format
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi = rgb_frame[y:y + h, x:x + w]

        
#         # Perform emotion analysis on the face ROI
#         result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

#         # Determine the dominant emotion
#         emotion = result[0]['dominant_emotion']

#         # Draw rectangle around face and label with predicted emotion
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Real-time Emotion Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()







# import cv2
# from deepface import DeepFace
# import time

# # Load face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize Video Capture
# cap = cv2.VideoCapture(0)

# # Initialize variables for attention and sleep tracking
# attention_threshold = 5  # Number of frames to consider for attention loss
# sleepy_threshold = 5  # Number of frames to consider for drowsiness
# attention_counter = 0
# sleep_counter = 0
# alert_color = (0, 0, 255)  # Default rectangle color (Red)
# alert_text = ""

# # Function to determine if a student is attentive
# def is_attentive(emotion):
#     return emotion not in ["sad", "angry", "fear", "disgust", "neutral"]

# # Start capturing video
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale and RGB format
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi = rgb_frame[y:y + h, x:x + w]

#         try:
#             # Perform emotion analysis on the face ROI
#             result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
#             emotion = result[0]['dominant_emotion']

#             # Determine attentiveness
#             if is_attentive(emotion):
#                 attention_counter = max(0, attention_counter - 1)  # Decrease counter if attentive
#                 alert_color = (0, 255, 0)  # Green for attentive
#                 alert_text = "Attentive"
#             else:
#                 attention_counter += 1
#                 if attention_counter >= attention_threshold:
#                     alert_color = (0, 0, 255)  # Red for not attentive
#                     alert_text = "Not Attentive"

#             # Check for drowsiness
#             if emotion == "neutral":
#                 sleep_counter += 1
#                 if sleep_counter >= sleepy_threshold:
#                     alert_text = "Drowsy"
#                     alert_color = (0, 255, 255)  # Yellow for drowsy
#             else:
#                 sleep_counter = max(0, sleep_counter - 1)  # Reset counter if not drowsy

#             # Draw rectangle around the face and label with attention status
#             cv2.rectangle(frame, (x, y), (x + w, y + h), alert_color, 2)
#             cv2.putText(frame, f"{emotion} - {alert_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 2)

#         except Exception as e:
#             print(f"Error analyzing emotion: {e}")

#     # Display the resulting frame
#     cv2.imshow('Classroom Behavior & Attention Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()









# import cv2
# from deepface import DeepFace
# import time

# # Load face and eye cascade classifiers
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Initialize Video Capture
# cap = cv2.VideoCapture(0)

# # Initialize variables for attention and sleep tracking
# attention_threshold = 5  # Number of frames to consider for attention loss
# sleepy_eye_closed_threshold = 10  # Time in seconds for eyes closed to consider as sleepy
# sleep_start_time = None
# attention_counter = 0
# alert_color = (0, 0, 255)  # Default rectangle color (Red)
# alert_text = ""

# # Function to determine if a student is attentive
# def is_attentive(emotion):
#     return emotion not in ["sad", "angry", "fear", "disgust"]

# # Function to check if eyes are detected (open)
# def eyes_open(face_roi):
#     eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10)
#     return len(eyes) > 0

# # Start capturing video
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale and RGB format
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi_gray = gray_frame[y:y + h, x:x + w]
#         face_roi_color = frame[y:y + h, x:x + w]

#         try:
#             # Perform emotion analysis on the face ROI
#             result = DeepFace.analyze(face_roi_color, actions=['emotion'], enforce_detection=False)
#             emotion = result[0]['dominant_emotion']

#             # Check if the person is attentive
#             if is_attentive(emotion):
#                 attention_counter = max(0, attention_counter - 1)  # Decrease counter if attentive
#                 alert_color = (0, 255, 0)  # Green for attentive
#                 alert_text = "Attentive"
#             else:
#                 attention_counter += 1
#                 if attention_counter >= attention_threshold:
#                     alert_color = (0, 0, 255)  # Red for not attentive
#                     alert_text = "Not Attentive"

#             # Check for eye status to detect drowsiness
#             if eyes_open(face_roi_gray):
#                 sleep_start_time = None  # Reset drowsiness if eyes are open
#             else:
#                 if sleep_start_time is None:
#                     sleep_start_time = time.time()
#                 elif time.time() - sleep_start_time >= sleepy_eye_closed_threshold:
#                     alert_text = "Sleepy"
#                     alert_color = (0, 255, 255)  # Yellow for drowsy

#             # Draw rectangle around the face and label with attention status
#             cv2.rectangle(frame, (x, y), (x + w, y + h), alert_color, 2)
#             cv2.putText(frame, f"{emotion} - {alert_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 2)

#         except Exception as e:
#             print(f"Error analyzing emotion: {e}")

#     # Display the resulting frame
#     cv2.imshow('Classroom Behavior & Attention Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# from deepface import DeepFace
# import numpy as np
# from collections import deque

# # Load face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start capturing video
# cap = cv2.VideoCapture(0)

# # Parameters for smoothing
# emotion_history_size = 10  # Number of frames to keep in history for averaging
# emotion_history = deque(maxlen=emotion_history_size)  # Deque for storing emotion history

# def average_emotions(emotion_list):
#     """Calculate the average of emotions over the last few frames."""
#     if not emotion_list:
#         return {}

#     # Aggregate the emotion probabilities
#     summed_emotions = {}
#     for emotions in emotion_list:
#         for emotion, probability in emotions.items():
#             summed_emotions[emotion] = summed_emotions.get(emotion, 0) + probability

#     # Calculate the average for each emotion
#     average_emotions = {emotion: prob / len(emotion_list) for emotion, prob in summed_emotions.items()}
#     return average_emotions

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale and RGB format
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi = rgb_frame[y:y + h, x:x + w]

#         try:
#             # Perform emotion analysis on the face ROI
#             result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
#             emotions = result[0]['emotion']  # Get all emotion probabilities

#             # Update emotion history for smoothing
#             emotion_history.append(emotions)

#             # Average emotions over the recent frames
#             averaged_emotions = average_emotions(emotion_history)

#             # Display the averaged emotions
#             emotion_text = ', '.join([f"{emotion}: {int(prob * 100)}%" for emotion, prob in averaged_emotions.items()])
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         except Exception as e:
#             print(f"Error analyzing emotion: {e}")

#     # Display the resulting frame
#     cv2.imshow('Real-time Emotion Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()






# import cv2
# from deepface import DeepFace
# import numpy as np
# from collections import deque
# import os

# # Load face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start capturing video
# cap = cv2.VideoCapture(0)

# # Parameters for emotion smoothing
# emotion_history_size = 10  # Number of frames to keep in history for averaging
# emotion_history = deque(maxlen=emotion_history_size)  # Deque for storing emotion history

# # Dictionary to hold known face encodings and names
# known_faces = {}
# # Load known faces (Assuming images are stored in 'known_faces' directory)
# for filename in os.listdir('known_faces'):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # Load the image
#         img_path = os.path.join('known_faces', filename)
#         img = cv2.imread(img_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # Detect the face and encode it
#         result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
#         known_faces[filename[:-4]] = img_rgb  # Store the image with the filename as the key

# def average_emotions(emotion_list):
#     """Calculate the average of emotions over the last few frames."""
#     if not emotion_list:
#         return {}

#     # Aggregate the emotion probabilities
#     summed_emotions = {}
#     for emotions in emotion_list:
#         for emotion, probability in emotions.items():
#             summed_emotions[emotion] = summed_emotions.get(emotion, 0) + probability

#     # Calculate the average for each emotion
#     average_emotions = {emotion: prob / len(emotion_list) for emotion, prob in summed_emotions.items()}
#     return average_emotions

# def recognize_student(face_roi):
#     """Recognize the student from the face ROI."""
#     try:
#         result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=True)
#         emotion = result[0]['dominant_emotion']
        
#         # Use the face_roi for recognition (you might need a different library for actual recognition)
#         # Here we'll simulate recognition
#         recognized_name = "Unknown"
#         for name, known_face in known_faces.items():
#             # Compare the face_roi with known_face (This is a placeholder for actual recognition)
#             if np.array_equal(face_roi, known_face):
#                 recognized_name = name
#                 break
        
#         return recognized_name, emotion
#     except Exception as e:
#         print(f"Error recognizing student: {e}")
#         return "Unknown", "Neutral"

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale and RGB format
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi = rgb_frame[y:y + h, x:x + w]

#         # Recognize the student and analyze emotion
#         recognized_name, dominant_emotion = recognize_student(face_roi)

#         # Update emotion history for smoothing
#         emotion_history.append({dominant_emotion: 1.0})  # Simplified: assume full probability for the detected emotion

#         # Average emotions over the recent frames
#         averaged_emotions = average_emotions(emotion_history)

#         # Display the recognized name and dominant emotion
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{recognized_name} - {dominant_emotion}", 
#                     (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Real-time Emotion and Student Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()







# import cv2
# from deepface import DeepFace
# import numpy as np
# from collections import deque
# import dlib
# from scipy.spatial import distance
# from imutils import face_utils
# import face_recognition

# # Load face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load dlib's face detector and facial landmarks predictor for drowsiness detection
# drowsiness_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# drowsiness_detector = dlib.get_frontal_face_detector()

# # Define constants for EAR and frame checks
# EAR_THRESHOLD = 0.25
# FRAME_CHECK = 20
# lStart, lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# rStart, rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# # Parameters for emotion smoothing
# emotion_history_size = 10  # Number of frames to keep in history for averaging
# emotion_history = deque(maxlen=emotion_history_size)  # Deque for storing emotion history

# # Load known face encodings for student identification
# known_face_encodings = []  # Load these with actual images of students
# known_face_names = []      # Load corresponding student names

# # Start capturing video
# cap = cv2.VideoCapture(0)

# def eye_aspect_ratio(eye):
#     """Calculate the eye aspect ratio (EAR) to detect drowsiness."""
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def average_emotions(emotion_list):
#     """Calculate the average of emotions over the last few frames."""
#     if not emotion_list:
#         return {}

#     # Aggregate the emotion probabilities
#     summed_emotions = {}
#     for emotions in emotion_list:
#         for emotion, probability in emotions.items():
#             summed_emotions[emotion] = summed_emotions.get(emotion, 0) + probability

#     # Calculate the average for each emotion
#     average_emotions = {emotion: prob / len(emotion_list) for emotion, prob in summed_emotions.items()}
#     return average_emotions

# flag = 0  # Flag for drowsiness detection
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale and RGB format
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     # Drowsiness Detection using dlib
#     subjects = drowsiness_detector(gray_frame, 0)

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest) for emotion analysis
#         face_roi = rgb_frame[y:y + h, x:x + w]

#         try:
#             # Perform emotion analysis on the face ROI
#             result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
#             emotions = result[0]['emotion']  # Get all emotion probabilities

#             # Update emotion history for smoothing
#             emotion_history.append(emotions)

#             # Average emotions over the recent frames
#             averaged_emotions = average_emotions(emotion_history)

#             # Get the most dominant emotion
#             dominant_emotion = max(averaged_emotions, key=averaged_emotions.get)
#             emotion_text = f"Emotion: {dominant_emotion}"

#             # Display the averaged dominant emotion
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         except Exception as e:
#             print(f"Error analyzing emotion: {e}")

#     # Drowsiness Detection
#     for subject in subjects:
#         shape = drowsiness_predictor(gray_frame, subject)
#         shape = face_utils.shape_to_np(shape)  # converting to NumPy Array

#         # Get eye coordinates
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         # Draw eye contours
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         # Check if eyes are closed
#         if ear < EAR_THRESHOLD:
#             flag += 1
#             if flag >= FRAME_CHECK:
#                 cv2.putText(frame, "****************ALERT!****************", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             flag = 0

#     # Display the resulting frame
#     cv2.imshow('Real-time Emotion and Drowsiness Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
from deepface import DeepFace
from scipy.spatial import distance
from imutils import face_utils
import dlib
import numpy as np

# Load face cascade classifier for initial face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's face detector and facial landmarks predictor for drowsiness detection
drowsiness_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
drowsiness_detector = dlib.get_frontal_face_detector()

# Define constants for EAR and frame checks
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20
lStart, lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize drowsiness flag
flag = 0

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR) to detect drowsiness."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and RGB format
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame using OpenCV Haar Cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Drowsiness Detection using dlib
    subjects = drowsiness_detector(gray_frame, 0)

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest) for emotion analysis
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            # Perform emotion analysis on the face ROI using DeepFace
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with the predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error analyzing emotion: {e}")

    # Drowsiness Detection with EAR
    for subject in subjects:
        shape = drowsiness_predictor(gray_frame, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array

        # Get eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if eyes are closed for drowsiness
        if ear < EAR_THRESHOLD:
            flag += 1
            if flag >= FRAME_CHECK:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0

    # Display the resulting frame
    cv2.imshow('Real-time Emotion and Drowsiness Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

