import streamlit as st
import pandas as pd
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to download required NLTK data
def download_nltk_data():
    nltk.download("punkt")
    nltk.download("stopwords")

# Call function only once
download_nltk_data()

# Load a pre-trained Hugging Face model
chatbot = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# Function to preprocess user input
def preprocess_input(user_input):
    """Removes stopwords and tokenizes the input text."""
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(user_input)
    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
    return " ".join(filtered_sentence)

# Healthcare response logic
def healthcare_chatbot(user_input):
    """Processes user queries and provides healthcare-related responses."""
    processed_input = preprocess_input(user_input)

    # Predefined keyword-based responses
    responses = {
        "sneeze": "🤧 Frequent sneezing and coughing may indicate allergies. Consider consulting an allergist.",
        "cough": "😷 Persistent cough could be a sign of allergies or respiratory infection. Seek medical advice if necessary.",
        "fever": "🌡️ Fever is commonly associated with infections. It's advisable to consult a doctor for evaluation.",
        "symptom": "🩺 It seems like you're experiencing some symptoms. Please consult a doctor for proper diagnosis.",
        "appointment": "📅 Would you like to schedule an appointment with a doctor?",
        "medication": "💊 Always take medications as prescribed. If you have concerns, discuss them with your doctor."
    }

    # Check for keyword matches
    for keyword, response in responses.items():
        if keyword in processed_input:
            return response

    # If no keyword matches, use AI-based response
    context = (
        "Common healthcare-related scenarios include symptoms of allergies, "
        "infections, and the importance of taking medication guidance from a doctor."
    )
    response = chatbot(context=context, question=processed_input)
    return response["answer"]

# Streamlit app interface
def main():
    # Set page layout
    st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥", layout="wide")

    # Add a background image and styling
    st.markdown(
        """
        <style>
            body {
                background-color: #f4f4f4;
            }
            .chat-container {
                border-radius: 15px;
                padding: 10px;
                margin: 10px 0;
                width: 80%;
            }
            .user-chat {
                background-color: #dcf8c6;
                text-align: left;
                margin-left: auto;
            }
            .bot-chat {
                background-color: #f0f0f0;
                text-align: left;
                margin-right: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with extra information
    with st.sidebar:
        st.image("https://png.pngtree.com/png-vector/20241109/ourmid/pngtree-3d-male-doctor-icon-png-image_14328515.png", width=120)
        st.title("💡 Need Help?")
        st.write("This chatbot provides general healthcare guidance but does **not** replace a doctor’s advice.")
        st.write("👨‍⚕️ Always consult a professional for medical concerns.")
        st.markdown("---")
        st.write("📌 **Common Questions to Ask:**")
        st.write("- What should I do if I have a fever?")
        st.write("- How can I relieve a cough?")
        st.write("- Can I take medication for allergies?")
    
    # Main chatbot UI
    st.title("🩺 Healthcare Assistant Chatbot")
    st.write("🤖 Ask any health-related question, and I'll assist you!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area("💬 Type your question here:", height=100)

    if st.button("Submit", key="submit"):
        if user_input.strip():
            response = healthcare_chatbot(user_input)
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Bot", response))

    # Display chat history with a bubble-like interface
    st.write("### 💬 Chat History")
    for role, text in st.session_state.chat_history:
        class_name = "user-chat" if role == "User" else "bot-chat"
        st.markdown(f"<div class='chat-container {class_name}'><b>{role}:</b> {text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
