import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
#comment previous and uncomment next line to use sample video
#cap = cv2.VideoCapture('vid-run.mp4')
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(1920,1080))
        image_hight, image_width, _ = image.shape
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not results.pose_landmarks:
          continue
        # Extraction of location of shoulders and hips
        pt1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        pt2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        pt3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        pt4 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculation of torso centroid
        centroid = [((pt1.x+pt2.x+pt3.x+pt4.x)/4)*image_width,((pt1.y+pt2.y+pt3.y+pt4.y)/4)*image_hight]
        print( f'Body centroid:' f'{centroid}')

        # Visualizing the tracked point
        cv2.circle(image,center = (int(centroid[0]),int(centroid[1])),radius = 2,color=(245,117,66),thickness=10 )

        # Render detections of standard mediapipe landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()