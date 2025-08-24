import cv2 
from deepface import DeepFace
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from PIL import Image, ImageTk
import mediapipe as mp

# Load face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Emotion weights for engagement scoring
emotion_weights = {
    'happy': 2,
    'surprise': 2,
    'neutral': 1,
    'fear': -1,
    'sad': -2,
    'angry': -2,
    'disgust': -2,
    'sleepy': -3
}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Data storage for exporting
data_records = []

# Create a window for the dashboard
dashboard = tk.Tk()
dashboard.title("ClassMonitor")
dashboard.geometry("800x600")  # Set the window size

# Define the main frame
main_frame = tk.Frame(dashboard)
main_frame.pack()

# First Page - Title and Start Button
def first_page():
    global main_frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    title_label = tk.Label(main_frame, text="ClassMonitor", font=("Helvetica", 24))
    title_label.pack(pady=20)

    start_button = tk.Button(main_frame, text="Start Monitoring", font=("Helvetica", 16), command=start_camera)
    start_button.pack(pady=20)

# Second Page - Camera Feed and Results Display
def start_camera():
    global main_frame, cap, eye_closed_start, eye_closed_duration, eye_closed_threshold, status_label
    eye_closed_start = None
    eye_closed_duration = 0
    eye_closed_threshold = 6  # seconds to trigger "Sleepy" state

    for widget in main_frame.winfo_children():
        widget.destroy()

    # Create a frame for the camera feed
    camera_frame = tk.Frame(main_frame)
    camera_frame.pack()

    # Create labels to display results
    global emotion_label, score_label, eye_closed_label, camera_frame_label
    camera_frame_label = tk.Label(camera_frame)
    camera_frame_label.pack()

    emotion_label = tk.Label(main_frame, text="Emotion: ", font=("Helvetica", 16))
    emotion_label.pack(pady=10)

    score_label = tk.Label(main_frame, text="Engagement Score: ", font=("Helvetica", 16))
    score_label.pack(pady=10)

    eye_closed_label = tk.Label(main_frame, text="Eyes Closed Duration: 0s", font=("Helvetica", 16))
    eye_closed_label.pack(pady=10)

    # Button Frame for options
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=10)

    # Buttons for displaying data, exporting data, and stopping the camera
    display_button = tk.Button(button_frame, text="Display Data", font=("Helvetica", 16), command=display_data)
    display_button.pack(side=tk.LEFT, padx=5)

    export_button = tk.Button(button_frame, text="Export Data", font=("Helvetica", 16), command=export_data)
    export_button.pack(side=tk.LEFT, padx=5)

    stop_button = tk.Button(button_frame, text="Stop Monitoring", font=("Helvetica", 16), command=stop_camera)
    stop_button.pack(side=tk.LEFT, padx=5)

    # Status Frame to show camera and export status
    status_frame = tk.Frame(main_frame)
    status_frame.pack(pady=10)
    
    global status_label
    status_label = tk.Label(status_frame, text="Camera Status: Not Started | Data Export Status: Not Exported", font=("Helvetica", 12))
    status_label.pack()

    cap = cv2.VideoCapture(0)
    update_frame()
    status_label.config(text="Camera Status: Started | Data Export Status: Not Exported")

# Function to stop the camera and close the application
def stop_camera():
    global cap, data_records
    if cap.isOpened():
        cap.release()  # Release the video capture
    data_records.clear()  # Clear the data records
    messagebox.showinfo("Camera Stopped", "The camera has been stopped and data cleared.")
    dashboard.destroy()  # Close the application

# Function to export data to CSV
def export_data():
    if data_records:
        df = pd.DataFrame(data_records)
        df.to_csv('engagement_data.csv', index=False)
        messagebox.showinfo("Export Successful", "Data has been exported successfully!")
        status_label.config(text="Camera Status: Started | Data Export Status: Exported")
    else:
        messagebox.showwarning("No Data", "No data available to export.")
        status_label.config(text="Camera Status: Started | Data Export Status: Exported")

# Function to display the recorded data
def display_data():
    if not data_records:
        messagebox.showwarning("No Data", "No data available to display.")
        return

    # Create a new window for displaying data
    display_window = tk.Toplevel(dashboard)
    display_window.title("Recorded Data")

    # Create a text widget to display the data
    text_widget = tk.Text(display_window, wrap='word')
    text_widget.pack(expand=True, fill='both')

    # Insert data into the text widget
    for record in data_records:
        text_widget.insert(tk.END, f"Emotion: {record['Emotion']}, Engagement Score: {record['Engagement Score']:.2f}, Eyes Closed Duration: {record['Eyes Closed Duration']:.1f}s\n")

    # Add a scrollbar
    scrollbar = tk.Scrollbar(display_window, command=text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.config(yscrollcommand=scrollbar.set)

# Function to update the dashboard with new data
def update_dashboard(emotion, score, eye_closed_duration):
    emotion_label.config(text=f"Emotion: {emotion}")
    score_label.config(text=f"Engagement Score: {score:.2f}")
    eye_closed_label.config(text=f"Eyes Closed Duration: {eye_closed_duration:.1f}s")

    # Record the data for display and export
    data_records.append({
        "Emotion": emotion,
        "Engagement Score": score,
        "Eyes Closed Duration": eye_closed_duration
    })

# Start capturing video and display updates
def update_frame():
    global cap, eye_closed_start, eye_closed_duration
    ret, frame = cap.read()
    if not ret:
        return

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize variables
    emotion = "Unknown"  # Default emotion if not detected
    engagement_score = 0  # Default score if no engagement

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(rgb_frame[y:y + h, x:x + w], scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        if len(eyes) == 0:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            else:
                eye_closed_duration += time.time() - eye_closed_start
                eye_closed_start = time.time()

            if eye_closed_duration >= eye_closed_threshold:
                emotion = "Sleepy"
        else:
            eye_closed_start = None

        engagement_score = sum(emotion_weights.get(em, 0) for em in [emotion])

        # Draw emotion and score on the frame
        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(rgb_frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(rgb_frame, f"Score: {engagement_score:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Update the dashboard with emotion and engagement score
    update_dashboard(emotion, engagement_score, eye_closed_duration)

    # Convert frame back to BGR for OpenCV
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Convert to Image and then to PhotoImage for Tkinter
    img = Image.fromarray(bgr_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Display the image in Tkinter
    camera_frame_label.imgtk = imgtk
    camera_frame_label.configure(image=imgtk)

    # Call update_frame after 10 ms
    dashboard.after(10, update_frame)

# Display the first page initially
first_page()
dashboard.mainloop()
