import cv2
from fer import FER

def capture_video_with_emotion():
    cap = cv2.VideoCapture(0)
    detector = FER()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions
        emotions = detector.detect_emotions(frame)

        if emotions:
            dominant_emotion = emotions[0]['dominant_emotion']
            engagement_score = emotions[0]['emotions'][dominant_emotion]  # Get engagement score based on the dominant emotion

            # Display engagement score on frame
            cv2.putText(frame, f'Engagement Score: {engagement_score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Class Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_video_with_emotion()
