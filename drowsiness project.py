import cv2
import numpy as np
from playsound import playsound  # for alert sound (install using pip)

# Constants for drowsiness detection
EAR_THRESHOLD = 0.25  # Adjust as needed
MAR_THRESHOLD = 0.5   # Adjust as needed
CONSEC_FRAMES = 30

# Variables to simulate vehicle speed
vehicle_speed = 60  # Starting speed (60 km/h)
max_speed = 100     # Maximum speed (100 km/h)
min_speed = 0       # Minimum speed (0 km/h)
speed_increment = 5  # Speed increment (5 km/h)
stop_speed = 0       # Speed to stop the vehicle

# Variable to simulate hazard lights
hazard_lights_on = False

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Check if the eye landmarks list has enough elements
    if len(eye) != 6:
        return 0.0

    # Calculate the horizontal distance between the left and right eye landmarks
    horizontal_distance = eye[3][0] - eye[0][0]

    # Calculate the vertical distance between the top and bottom eye landmarks
    vertical_distance = (eye[1][1] + eye[2][1]) / 2 - (eye[4][1] + eye[5][1]) / 2

    # Calculate EAR
    ear = horizontal_distance / (2.0 * vertical_distance)
    return ear

# Function to detect yawning based on mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    # Check if the mouth landmarks list has enough elements
    if len(mouth) != 20:
        return 0.0

    # Calculate MAR
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    MAR = (A + B) / (2.0 * C)
    return MAR

# Function to slow down the vehicle
def slow_down():
    global vehicle_speed
    if vehicle_speed > stop_speed:
        vehicle_speed -= speed_increment
    else:
        vehicle_speed = stop_speed

# Function to speed up the vehicle
def speed_up():
    global vehicle_speed
    if vehicle_speed < max_speed:
        vehicle_speed += speed_increment

# Function to stop the vehicle
def stop_vehicle():
    global vehicle_speed
    vehicle_speed = stop_speed

# Function to toggle hazard lights
def toggle_hazard_lights():
    global hazard_lights_on
    hazard_lights_on = not hazard_lights_on

# Initialize variables
frame_counter = 0
alarm_on = False

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for each detected face
        roi_gray = gray[y:y+h, x:x+w]

        # Calculate EAR for the left and right eyes (example landmarks)
        left_eye = [(x + w // 7, y + h // 4), (x + 2 * w // 7, y + h // 4), (x + w // 2, y + h // 2),
                    (x + w // 7, y + 3 * h // 4), (x + 2 * w // 7, y + 3 * h // 4)]
        right_eye = [(x + 3 * w // 4, y + h // 4), (x + 4 * w // 7, y + h // 4), (x + 3 * w // 4, y + h // 2),
                     (x + 3 * w // 4, y + 3 * h // 4), (x + 4 * w // 7, y + 3 * h // 4)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR for the mouth (example landmarks)
        mouth = [(x + w // 7, y + 3 * h // 4), (x + 2 * w // 7, y + 3 * h // 4), (x + w // 2, y + 7 * h // 8),
                 (x + 3 * w // 7, y + 7 * h // 8), (x + 4 * w // 7, y + 3 * h // 4), (x + 3 * w // 7, y + h // 2),
                 (x + w // 2, y + h // 2)]

        mar = mouth_aspect_ratio(mouth)

        if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
            frame_counter += 1

            if frame_counter >= CONSEC_FRAMES:
                if not alarm_on:
                    playsound("alarm.mp3")  # Play an alert sound
                    alarm_on = True
                    slow_down()  # Simulate slowing down the vehicle
                    toggle_hazard_lights()  # Turn on hazard lights

                # Display the text when drowsiness or yawning is detected
                cv2.putText(frame, "Drowsiness/Yawning Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_counter = 0
            alarm_on = False  # Reset the alarm state when eyes and mouth are not indicative of drowsiness/yawning

    # Display the frame
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Check if 'q' key is pressed
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
