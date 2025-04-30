import streamlit as st
# from Audio import text_to_speech, get_audio
import cv2
import tempfile
import ExerciseAiTrainer as exercise
from chatbot import chat_ui
import time
from PIL import Image
import os

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def show_exercise_guide(exercise_name):
    guides = {
        'Biceps Curl': """
        ### How to perform Biceps Curls correctly:
        1. Stand with feet shoulder-width apart
        2. Keep your back straight and core engaged
        3. Hold weights with palms facing forward
        4. Curl weights up while keeping elbows close to body
        5. Lower weights back down with control
        """,
        'Push Up': """
        ### How to perform Push-ups correctly:
        1. Start in a plank position with hands shoulder-width apart
        2. Keep your body straight from head to heels
        3. Lower your body until chest nearly touches the ground
        4. Push back up to starting position
        5. Keep elbows close to your body
        """,
        'Squat': """
        ### How to perform Squats correctly:
        1. Stand with feet shoulder-width apart
        2. Keep your back straight and chest up
        3. Lower your body as if sitting back into a chair
        4. Keep knees aligned with toes
        5. Push through heels to return to starting position
        """,
        'Shoulder Press': """
        ### How to perform Shoulder Press correctly:
        1. Stand with feet shoulder-width apart
        2. Hold weights at shoulder level
        3. Press weights overhead with control
        4. Keep core engaged and back straight
        5. Lower weights back to shoulders
        """
    }
    st.markdown(guides.get(exercise_name, ""))

def main():
    # Set configuration for the theme
    st.set_page_config(
        page_title='Fitness AI Coach',
        page_icon='💪',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Load custom CSS
    load_css()
    
    # Define App Title and Structure
    st.title('AI Exercise Pose Detection')
    st.markdown("""
    Welcome to your personal AI fitness coach! This application helps you perform exercises correctly
    by providing real-time feedback and counting your repetitions.
    """)
    
    # Sidebar Configuration
    st.sidebar.title('Navigation')
    
    # Mode Selection with clear label and styling
    st.sidebar.markdown('### Choose Mode')
    mode = st.sidebar.selectbox(
        'Select your preferred training mode',
        ('Video Analysis', 'Live Webcam', 'Auto Classification', 'AI Chat Assistant'),
        help='Choose how you want to interact with the AI trainer'
    )
    
    # Exercise Selection (show only for relevant modes)
    if mode in ['Video Analysis', 'Live Webcam']:
        st.sidebar.markdown('### Exercise Type')
        exercise_type = st.sidebar.selectbox(
            'Select the exercise you want to perform',
            ('Biceps Curl', 'Push Up', 'Squat', 'Shoulder Press'),
            help='Choose the exercise you want to analyze'
        )
    
    # Video Upload Section (only for Video Analysis)
    if mode == 'Video Analysis':
        st.sidebar.markdown("""
        <div class='uploadVideoSection'>
            <h3>Upload Exercise Video</h3>
            <p>Upload your video for AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        video_file = st.sidebar.file_uploader(
            "Choose video file",
            type=["mp4", "mov", 'avi', 'asf', 'm4v'],
            help='Supported formats: MP4, MOV, AVI, ASF, M4V'
        )
    
    # Main Content Area
    if mode == 'AI Chat Assistant':
        st.markdown('### AI Fitness Assistant')
        st.info('Ask me anything about fitness, exercises, or form!')
        chat_ui()
    
    elif mode == 'Video Analysis':
        st.markdown('### Video Analysis Mode')
        st.info('Upload your exercise video for analysis and feedback.')
        
        if video_file:
            tfflie = tempfile.NamedTemporaryFile(delete=False)
            tfflie.write(video_file.read())
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('### Your Video')
                st.video(tfflie.name)
            
            with col2:
                st.markdown('### AI Analysis')
                st.info('Processing your video...')
                
                exer = exercise.Exercise()
                if exercise_type == 'Biceps Curl':
                    counter, stage_right, stage_left = 0, None, None
                    exer.bicept_curl(cv2.VideoCapture(tfflie.name), is_video=True, 
                                   counter=counter, stage_right=stage_right, stage_left=stage_left)
                elif exercise_type == 'Push Up':
                    counter, stage = 0, None
                    exer.push_up(cv2.VideoCapture(tfflie.name), is_video=True, 
                               counter=counter, stage=stage)
                elif exercise_type == 'Squat':
                    counter, stage = 0, None
                    exer.squat(cv2.VideoCapture(tfflie.name), is_video=True, 
                             counter=counter, stage=stage)
                elif exercise_type == 'Shoulder Press':
                    counter, stage = 0, None
                    exer.shoulder_press(cv2.VideoCapture(tfflie.name), is_video=True, 
                                      counter=counter, stage=stage)
    
    elif mode == 'Auto Classification':
        st.markdown('### Automatic Exercise Detection')
        st.info('The AI will automatically detect and analyze your exercises in real-time.')
        
        if st.button('Start Auto Detection', help='Click to begin automatic exercise detection'):
            with st.spinner('Initializing camera...'):
                time.sleep(2)
                exer = exercise.Exercise()
                exer.auto_classify_and_count()
    
    elif mode == 'Live Webcam':
        st.markdown('### Real-time Exercise Analysis')
        st.info('Get instant feedback on your exercise form using your webcam.')
        
        if st.button('Start Live Analysis', help='Click to begin real-time exercise analysis'):
            with st.spinner('Initializing camera...'):
                time.sleep(2)
                cap = cv2.VideoCapture(0)
                exer = exercise.Exercise()
                
                if exercise_type == 'Biceps Curl':
                    counter, stage_right, stage_left = 0, None, None
                    exer.bicept_curl(cap, counter=counter, stage_right=stage_right, stage_left=stage_left)
                elif exercise_type == 'Push Up':
                    counter, stage = 0, None
                    exer.push_up(cap, counter=counter, stage=stage)
                elif exercise_type == 'Squat':
                    counter, stage = 0, None
                    exer.squat(cap, counter=counter, stage=stage)
                elif exercise_type == 'Shoulder Press':
                    counter, stage = 0, None
                    exer.shoulder_press(cap, counter=counter, stage=stage)

if __name__ == '__main__':
    main()
