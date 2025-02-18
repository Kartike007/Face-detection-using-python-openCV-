import cv2
import os

class FaceCapture:
    def __init__(self, capture_directory):
        self.capture_directory = capture_directory
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Create directo
        # ry to store images
        if not os.path.exists(self.capture_directory):
            os.makedirs(self.capture_directory)

    def capture_images(self, user_id, user_name):
        cap = cv2.VideoCapture(0)
        image_count = 0

        while image_count < 100:
            ret, frame = cap.read()

            # Detect faces in the frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangle around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display frame with rectangle in real-time
            cv2.imshow("Capture", frame)

            # Save captured image to file
            if len(faces) == 1:
                img_path = os.path.join(self.capture_directory, f'user_{user_id}_{user_name}_{image_count}.jpg')
                cv2.imwrite(img_path, gray_frame[y:y+h, x:x+w])
                print(f"Image {image_count + 1} captured and saved for user {user_id} ({user_name})")
                image_count += 1

            # Break loop if 'q' is pressed or 100 images are captured
            if cv2.waitKey(1) == ord('q') or image_count == 100:
                break

        # Release video capture object and close windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_directory = 'captured_images'
    face_capture = FaceCapture(capture_directory)

    # Check if there is an existing user ID file
    user_id_file = 'current_user.txt'
    if os.path.exists(user_id_file):
        with open(user_id_file, 'r') as file:
            user_id = int(file.read().strip())
            user_id += 1
    else:
        user_id = 1

    # Prompt for user name
    user_name = input("Enter user name: ")

    # Save the current user ID for next run
    with open(user_id_file, 'w') as file:
        file.write(str(user_id))

    # Capture images for the current user
    face_capture.capture_images(user_id, user_name)
