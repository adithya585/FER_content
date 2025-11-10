import streamlit as st
import cv2
import numpy as np
import csv
import os

try:
    from keras.models import model_from_json
    HAS_TF = True
except Exception:
    model_from_json = None
    HAS_TF = False

# Function to predict emotions from the given image
def predict_emotions(image):
    # Load model architecture from JSON file
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_json_path = os.path.join(models_dir, 'fermodel2.json')
    model_weights_path = os.path.join(models_dir, 'fermodel28199.weights.h5')

    if not HAS_TF:
        st.error('TensorFlow/Keras not available. Enable Demo mode or install dependencies.')
        return [], [], image

    if not (os.path.exists(model_json_path) and os.path.exists(model_weights_path)):
        st.error('Model files not found. Please place "fermodel2.json" and "fermodel28199.weights.h5" under the "models" folder next to this script.')
        return [], [], image

    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # Load model weights
    model.load_weights(model_weights_path)

    # Load face cascade classifier
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    #top_emotions = []
    detected_emotions = []
    probabilities = []
    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + h, x:x + w]  # Extract face ROI
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.expand_dims(roi_gray, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)  # Add channel dimension
        img_pixels = img_pixels.astype('float32')
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        top_indices = np.argsort(predictions)[0][-2:][::-1]  # Indices of top two predicted emotions

        # If the second highest predicted emotion is neutral, happiness, or surprise, skip it
        if emotions[top_indices[1]] in ['neutral', 'happiness', 'surprise']:
            top_indices = top_indices[:1]  # Take only the highest predicted emotion

        detected_emotions.append([emotions[idx] for idx in top_indices])
        probabilities.append([predictions[0][idx] for idx in top_indices])

    return detected_emotions, probabilities, image

# Demo prediction when model is not available
def predict_emotions_demo(image):
    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    # pick a stable default for demo
    demo_emotion = ['happiness']
    demo_prob = [0.95]
    return [demo_emotion], [demo_prob], image

# Function to write data to CSV file and display in table format
def print_content(emotion, color):
    emotion_lower = emotion.lower() if isinstance(emotion, str) else str(emotion)
    base_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    file_paths = {
        'anger': {
            'movies': os.path.join(base_dir, 'anger_movies.csv'),
            'books': os.path.join(base_dir, 'anger_books.csv'),
            'music': os.path.join(base_dir, 'anger_music.csv')
        },
        'disgust': {
            'movies': os.path.join(base_dir, 'disgust_movies.csv'),
            'books': os.path.join(base_dir, 'disgust_books.csv'),
            'music': os.path.join(base_dir, 'disgust_music.csv')
        },
        'fear': {
            'movies': os.path.join(base_dir, 'fear_movies.csv'),
            'books': os.path.join(base_dir, 'fear_books.csv'),
            'music': os.path.join(base_dir, 'fear_music.csv')
        },
        'happiness': {
            'movies': os.path.join(base_dir, 'happiness_movies.csv'),
            'books': os.path.join(base_dir, 'happiness_books.csv'),
            'music': os.path.join(base_dir, 'happiness_music.csv')
        },
        'neutral': {
            'movies': os.path.join(base_dir, 'neutral_movies.csv'),
            'books': os.path.join(base_dir, 'neutral_books.csv'),
            'music': os.path.join(base_dir, 'neutral_music.csv')
        },
        'sadness': {
            'movies': os.path.join(base_dir, 'sadness_movies.csv'),
            'books': os.path.join(base_dir, 'sadness_books.csv'),
            'music': os.path.join(base_dir, 'sadness_music.csv')
        },
        'surprise': {
            'movies': os.path.join(base_dir, 'surprise_movies.csv'),
            'books': os.path.join(base_dir, 'surprise_books.csv'),
            'music': os.path.join(base_dir, 'surprise_music.csv')
        }
    }

    for category, path in file_paths[emotion_lower].items():
        with open(path, 'r') as file:
            reader = csv.reader(file)
            contents = list(reader)

        if contents:
            st.markdown(f"**_{category.capitalize()} for the {emotion_lower} emotion_**")
            st.table(contents)  # Print the first row in bold
            #st.table(contents[1:])
# Main function to create the Streamlit app
def main():
    st.title('Facial Emotion Recognition')
    demo_mode = st.checkbox('Demo mode (no model)', value=not HAS_TF)

    if st.button('Take Photo'):
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        if ret:
            if demo_mode:
                emotions, probabilities, image = predict_emotions_demo(frame)
            else:
                emotions, probabilities, image = predict_emotions(frame)

            # If multiple emotions are detected, print the content for both
            if len(emotions) > 1:
                st.image(image, channels="BGR", caption='Detected Emotions with Bounding Boxes')

                st.markdown("### Detected Emotions and Probabilities:")
                for emotion, prob in zip(emotions, probabilities):
                    st.write(f"- **{emotion.capitalize()}**: {prob:.2f}")

                st.markdown("### Content for Detected Emotions:")
                for emotion in emotions:
                    print_content(emotion, color='yellow')
            else:
                if len(emotions) > 0:
                    emotion = emotions[0]
                    st.image(image, channels="BGR", caption=f'Detected Emotion: {emotion}')

                    st.markdown("### Detected Emotion and Probability:")
                    prob = probabilities[0]
                    st.write("### Predicted Emotion:")
                    if len(emotion) > 1:
                        for emotion1, probability1 in zip(emotion, prob):
                            st.write(f"- **{emotion1.capitalize()}**: {probability1:.2f}")
                            st.markdown("### Content for Detected Emotion:")
                            print_content(emotion1, color='yellow')
                    elif len(emotion) == 1:
                        st.write(f"- **{emotion[0].capitalize()}**: {prob[0]:.2f}")
                        st.markdown("### Content for Detected Emotion:")
                        print_content(emotion[0], color='yellow')
                else:
                    st.error('Failed to capture image.')

        cap.release()

if __name__ == '__main__':
    main()
