import streamlit as st
import numpy as np
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import user_analytics
import plotly.graph_objects as go


# Load pretrained tokenizer and model
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Ensure model is in evaluation mode
model.eval()



emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 
            'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 
            'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Fit the MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=emotions)
mlb.fit([emotions])  # Fit the binarizer with the right number of emotions

def preprocess(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

# Streamlit app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Emotion Detection", "User Analytics"])

# Page 1: Emotion Detection
if page == "Emotion Detection":
    st.header("Let us see how you are doing!")


    user_input = st.text_area("Enter a sentence to detect emotion:")

    if user_input:
        # Preprocess the input
        inputs = preprocess(user_input)

        # Get predictions from the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits (output scores before softmax)
        logits = outputs.logits
        probs = torch.sigmoid(logits)  # Since it's multi-label, use sigmoid
        predictions = np.array(probs > 0.5).astype(int)  # Thresholding at 0.5 for multi-label classification

        # Map predictions back to the emotion labels
        predicted_emotions = mlb.inverse_transform(predictions)
        emotion_list = list(predicted_emotions[0]) if predicted_emotions else []

        # Display the predicted emotions
        predicted_emotions = mlb.inverse_transform(predictions)
        emotion_list = list(predicted_emotions[0]) if predicted_emotions else []

        # Check if emotion_list is not empty before accessing it
        if emotion_list:
            predicted_emotion = emotion_list[0]
            st.write(f"I can sense you are feeling {predicted_emotion}")
        else:
            st.write("I couldn't detect any specific emotion.")
    else:
        st.write("Please enter some text to analyze.")

elif page == "User Analytics":
    page2 = user_analytics
    page2.go_to_page_user_analytics()
    