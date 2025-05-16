import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepface import DeepFace

def apply_fourier_transform(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    return magnitude_spectrum, fshift

def get_colormap_for_emotion(emotion):
    """ Assigns different color maps for each emotion """
    colormap_dict = {
        "happy": "inferno",
        "sad": "cool",
        "angry": "hot",
        "surprise": "spring",
        "neutral": "viridis"
    }
    return colormap_dict.get(emotion, "gray")

def show_3d_fourier(fshift, emotion):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rows, cols = fshift.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = np.abs(fshift)

    # Normalize the values to enhance contrast
    Z = np.log(Z + 1)

    # Select colormap based on detected emotion
    cmap = get_colormap_for_emotion(emotion)

    # Plot 3D Fourier with emotion-based colors
    ax.plot_surface(X, Y, Z, cmap=cmap)

    ax.set_title(f"3D Fourier Transform - Emotion: {emotion}")
    ax.set_xlabel("Frequency X")
    ax.set_ylabel("Frequency Y")
    ax.set_zlabel("Magnitude (Scaled)")
    
    plt.show()

def analyze_webcam():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        fourier_frame, fshift = apply_fourier_transform(frame)
        cv2.imshow('Fourier Transform', fourier_frame)
        cv2.imwrite('temp_frame.jpg', frame)

        try:
            result = DeepFace.analyze('temp_frame.jpg', actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            dominant_emotion = "neutral"

        cv2.imshow('Webcam Emotion Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('3'):
            show_3d_fourier(fshift, dominant_emotion)
    
    cap.release()
    cv2.destroyAllWindows()

analyze_webcam()
