import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import os
from ExerciseAiTrainer import Exercise, calculate_angle, calculate_distance, calculate_y_distance

class ModelTester:
    def __init__(self):
        # Load the model and preprocessing components
        self.model = load_model('final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5')
        self.scaler = joblib.load('thesis_bidirectionallstm_scaler.pkl')
        self.label_encoder = joblib.load('thesis_bidirectionallstm_label_encoder.pkl')
        self.exercise_classes = self.label_encoder.classes_
        
        # Initialize MediaPipe Pose
        self.pose = mp.solutions.pose.Pose()
        
        # Define relevant landmarks indices (same as in ExerciseAiTrainer.py)
        self.relevant_landmarks_indices = [
            11, 12, 13, 14, 15, 16,  # Arms
            23, 24, 25, 26, 27, 28,  # Legs
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  # Upper body
        ]

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(self.relevant_landmarks_indices) * 3:
            # Angles
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE

            # New angles
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))  # LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))

            # Distances
            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),  # LEFT_SHOULDER, RIGHT_SHOULDER
                calculate_distance(landmarks[18:21], landmarks[21:24]),  # LEFT_HIP, RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30]),  # RIGHT_HIP, RIGHT_KNEE
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                calculate_distance(landmarks[6:9], landmarks[24:27]),  # LEFT_ELBOW, LEFT_KNEE
                calculate_distance(landmarks[9:12], landmarks[27:30]),  # RIGHT_ELBOW, RIGHT_KNEE
                calculate_distance(landmarks[12:15], landmarks[0:3]),  # LEFT_WRIST, LEFT_SHOULDER
                calculate_distance(landmarks[15:18], landmarks[3:6]),  # RIGHT_WRIST, RIGHT_SHOULDER
                calculate_distance(landmarks[12:15], landmarks[18:21]),  # LEFT_WRIST, LEFT_HIP
                calculate_distance(landmarks[15:18], landmarks[21:24])   # RIGHT_WRIST, RIGHT_HIP
            ]

            # Y-coordinate distances
            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),  # LEFT_ELBOW, LEFT_SHOULDER
                calculate_y_distance(landmarks[9:12], landmarks[3:6])   # RIGHT_ELBOW, RIGHT_SHOULDER
            ]

            # Normalization factor based on shoulder-hip or hip-knee distance
            normalization_factor = -1
            distances_to_check = [
                calculate_distance(landmarks[0:3], landmarks[18:21]),  # LEFT_SHOULDER, LEFT_HIP
                calculate_distance(landmarks[3:6], landmarks[21:24]),  # RIGHT_SHOULDER, RIGHT_HIP
                calculate_distance(landmarks[18:21], landmarks[24:27]),  # LEFT_HIP, LEFT_KNEE
                calculate_distance(landmarks[21:24], landmarks[27:30])   # RIGHT_HIP, RIGHT_KNEE
            ]

            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break
            
            if normalization_factor == -1:
                normalization_factor = 0.5  # Fallback normalization factor
            
            # Normalize distances
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            # Combine features
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)

        else:
            print(f"Insufficient landmarks: expected {len(self.relevant_landmarks_indices)}, got {len(landmarks)//3}")
            features = [-1.0] * 22  # Placeholder for missing landmarks
        return features

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in self.relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

    def evaluate_video(self, video_path, true_label):
        """Evaluate a single video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None

        window_size = 30
        landmarks_window = []
        predictions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.preprocess_frame(frame)
            if len(landmarks) == len(self.relevant_landmarks_indices) * 3:
                features = self.extract_features(landmarks)
                if len(features) == 22:
                    landmarks_window.append(features)

            if len(landmarks_window) == window_size:
                landmarks_window_np = np.array(landmarks_window).flatten().reshape(1, -1)
                scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22)

                prediction = self.model.predict(scaled_landmarks_window)
                predicted_class = np.argmax(prediction, axis=1)[0]
                predictions.append(predicted_class)
                landmarks_window = []

        cap.release()
        
        if not predictions:
            return None
            
        # Get the most common prediction
        final_prediction = max(set(predictions), key=predictions.count)
        return final_prediction

    def evaluate_test_set(self, test_data_dir):
        """Evaluate all videos in the test directory"""
        true_labels = []
        predicted_labels = []
        
        # Walk through the test directory
        for root, dirs, files in os.walk(test_data_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    # Extract true label from directory name
                    true_label = os.path.basename(root)
                    if true_label in self.exercise_classes:
                        video_path = os.path.join(root, file)
                        prediction = self.evaluate_video(video_path, true_label)
                        
                        if prediction is not None:
                            true_labels.append(self.label_encoder.transform([true_label])[0])
                            predicted_labels.append(prediction)
        
        if not true_labels:
            print("No valid test videos found!")
            return None
            
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        # Print confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

if __name__ == "__main__":
    # Initialize the tester
    tester = ModelTester()
    
    # Path to your test data directory
    test_data_dir = "test_data"  # Replace with your actual test data directory
    
    # Run evaluation
    results = tester.evaluate_test_set(test_data_dir)
    
    if results:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed or no valid test data found.") 