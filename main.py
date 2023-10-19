import face_recognition as frec
import os, sys
import cv2
import numpy as np
import math
import threading


def face_confidence(face_distance, face_match_threshold=0.6):
    range = 1.0 - face_match_threshold
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (
            linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
        ) * 100
        return str(round(value, 2)) + "%"


class FaceRecognition:
    faceLocations = []  # faceLocation is for storing the rgb frames for matching
    faceEncodings = []  # faceEncoding is for encode the image for the video
    faceNames = []  # faceNames is store the name and confidence for the video
    knownFaceEncodings = (
        []
    )  # knownFaceEncodings is for the store the encoded of the external image
    knownFaceNames = (
        []
    )  # knownFaceNames is for the the store name in order to added a image in knownFaceEncodings
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        # this method get the images in form of encoded system , image name
        shivam = frec.load_image_file("Shivam.jpg")
        shivam_encoding = frec.face_encodings(shivam)[0]
        keyush = frec.load_image_file("Keyush.jpg")
        keyush_encoding = frec.face_encodings(keyush)[0]

        self.knownFaceEncodings = [
            shivam_encoding,
            keyush_encoding,
        ]
        self.knownFaceNames = ["shivam", "keyush"]
        print(self.knownFaceNames)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)  # This will take Web-cam source

        if (
            not video_capture.isOpened()
        ):  # This will check if your camera is found or not
            sys.exit("Video source not found...")

        def recognition_thread():
            process_current_frame = True  # Move this variable inside the method
            while True:
                (
                    ret,
                    frame,
                ) = (
                    video_capture.read()
                )  # Take the frame if available, then ret = True; otherwise, False
                frame = cv2.flip(frame, 1)  # Flip your webcam video

                cv2.putText(
                    frame,
                    "By Shivam Patel",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )  # Put the By Shivam Patel name in the corner in the live cam (BGR)

                if process_current_frame:
                    small_frame = cv2.resize(
                        frame, (0, 0), fx=0.25, fy=0.25
                    )  # Set the video frame and resize to 1/4
                    rgb_small_frame = small_frame[
                        :, :, ::-1
                    ]  # Return the BGR small_frame in the form of rgb_small_frame
                    self.faceLocations = frec.face_locations(rgb_small_frame)
                    self.faceEncodings = frec.face_encodings(
                        rgb_small_frame, self.faceLocations
                    )
                    self.faceNames = []
                    for (
                        face_encoding
                    ) in (
                        self.faceEncodings
                    ):  # Check if the video person is known or unknown
                        matches = frec.compare_faces(
                            self.knownFaceEncodings, face_encoding
                        )
                        name = "Unknown"]
                        confidence = "???"
                        face_distances = frec.face_distance(
                            self.knownFaceEncodings, face_encoding
                        )

                        best_match_index = np.argmin(face_distances)
                        if matches[
                            best_match_index
                        ]:  # If the person is known, then return the face name and confidence (percentage of accuracy)
                            name = self.knownFaceNames[best_match_index]
                            confidence = face_confidence(
                                face_distances[best_match_index]
                            )

                        self.faceNames.append(f"{name} ({confidence})")
                process_current_frame = not process_current_frame

                for (top, right, bottom, left), name in zip(
                    self.faceLocations, self.faceNames
                ):  # Make an annotation box and name box for the live webcam
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(
                        frame,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        name,
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow(
                    "Face Recognition", frame
                )  # Return the fully functional face recognition window
                v = cv2.waitKey(1)
                if v == ord("c"):  # Exit key = 'c'
                    break
            video_capture.release()  # Release video after capture and clear the buffer
            cv2.destroyAllWindows()

        recognition_thread = threading.Thread(target=recognition_thread)
        recognition_thread.start()


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()
