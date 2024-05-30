import torch
import cv2

# Load your custom model
model = torch.hub.load('.', 'custom', path='best.pt', source='local')

# Open the video file
video_path = 'video4.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    for detection in results.xyxy[0]: 
        box = detection[:4]  
        label = detection[5]
        confidence = detection[4] 
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        text = f'{label}: {confidence:.2f}'
        cv2.putText(frame, text, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
