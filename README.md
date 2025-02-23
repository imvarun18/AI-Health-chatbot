# AI-Powered Healthcare Assistant 🤖🏥  

An AI-driven chatbot designed to assist users with healthcare-related queries using **NLP** and **pre-trained BERT models**. It provides **symptom analysis, medication guidance, and general health information** through an interactive **Streamlit-based UI**.  

---

## 🌟 Features  
✅ **Symptom Analysis** – Provides insights on common health conditions based on symptoms.  
✅ **Medication Guidance** – Offers dosage and side effect information for medications.  
✅ **Mental Health Support** – Suggests self-care and stress management tips.  
✅ **AI-Powered Q&A** – Uses a **BERT-based model** for healthcare-related question-answering.  
✅ **Interactive UI** – Built using **Streamlit** for easy user interaction.  

---

## 🛠️ Technologies Used  
- **Python 3.7+** – Programming language  
- **Streamlit** – Web-based user interface  
- **Hugging Face Transformers** – AI model for question-answering  
- **NLTK** – Natural language preprocessing (tokenization, stopword removal)  
- **Pandas** – (Optional) Data handling for structured health data  
- **Docker** – (Optional) Deployment containerization  
- **Git & GitHub** – Version control  

---

## 🚀 Installation & Setup  

### **1️⃣ Clone the Repository**  
    git clone https://github.com/imvarun18/AI-Health-chatbot.git
    cd healthcare-chatbot
### **2️⃣ Install Dependencies**
    pip install -r requirements.txt
### **3️⃣ Run the Application**
    streamlit run app.py

---

## User Interface using streamlit
![Screenshot 2025-02-23 2359251](https://github.com/user-attachments/assets/f29afb3e-060d-4ee3-a13d-6213c453b6ef)

---

## 📜 How It Works
1️.  **User Input** – The user types a health-related question in the chatbot.    
2️.  **Preprocessing** – The input is tokenized and cleaned using NLTK.   
3.  **AI Processing**– The system checks predefined rules or forwards the query to BERT-based model for AI-driven responses.   
4. **Response Generation**– The chatbot provides the best-matching answer. 

## 🏗️ Future Enhancements
🔹 Voice-enabled chatbot integration   
🔹 Multilingual support for broader accessibility  
🔹 Integration with EHR (Electronic Health Records) systems   
🔹 Improved AI model with fine-tuned medical datasets  
