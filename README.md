
# ⚕️ Healthcare Info Bot (RAG Q&A Chatbot)

This project implements a **Retrieval-Augmented Generation (RAG)** Q&A chatbot designed to provide information on common healthcare topics. It uses a local knowledge base combined with **Google Gemini 1.5 Flash API** for intelligent, fact-checked answers.

You can interact with the bot via a **terminal interface** or an **interactive web app using Streamlit**.

---

## ✨ Features

- 🔍 **RAG (Retrieval-Augmented Generation):** Combines semantic search with generative AI for accurate answers grounded in your documents.
- 🧠 **Google Gemini 1.5 Flash API:** Generates human-like responses based on context.
- 📚 **Local Knowledge Base:** Uses curated healthcare documents stored in `data/`.
- ⚡ **FAISS Vector Store:** Enables fast and accurate document retrieval.
- 💬 **Chat History & Memory:** Maintains conversational context in the web UI.
- 🌐 **Streamlit Web App:** A sleek and responsive chatbot interface.
- 🖥️ **Terminal Interface:** Quick testing and usage through the command line.
- 📦 **Self-contained Deployment:** Easily deployable on Streamlit Cloud with no external database required.

---

## 🧠 Knowledge Base Topics

The bot is capable of answering questions on the following healthcare topics:

### Common Illnesses
- Anxiety (Basics)
- Asthma (Basics)
- Breast Cancer
- Common Cold
- Dengue Fever
- Diabetes (Type 1 & 2 Basics)
- Headaches (Tension, Migraine, Cluster)
- Hypertension (High Blood Pressure)
- Malaria (Basics)
- Pneumonia (Children)
- Polycystic Ovary Syndrome (PCOS)
- Post-Traumatic Stress Disorder (PTSD)

### General Health & Wellness
- Basic Nutrition Guidelines
- Sleep Hygiene (Tips for Better Sleep)
- Mental Health Basics
- Basic First Aid (Cuts, Scrapes, Burns, Sprains)
- Vaccination Basics

> **Disclaimer:** This chatbot provides general health information only. For medical diagnosis or treatment, consult a certified healthcare professional.

---

## 🚀 Tech Stack

- **Python 3.9+**
- **LangChain:** LLM application framework
- **Streamlit:** Web app interface
- **Google Gemini API:** LLM backend
- **HuggingFace `sentence-transformers`:** For document embeddings
- **FAISS:** Fast similarity search
- **python-dotenv:** API key management

---

## 📂 Project Structure

```

healthcare\_info\_bot/
├── data/                    # Raw text/PDF health documents
│   ├── anxiety\_basics.txt
│   ├── breast\_cancer.txt
│   └── ...
├── vectorstore/
│   └── db\_faiss/            # FAISS vector database (pre-built)
├── .env                     # Your API key (excluded from Git)
├── requirements.txt         # Python dependencies
├── create\_vector\_db.py      # Script to build/update the vector DB
├── chatbot\_app.py           # Terminal-based chatbot
└── streamlit\_app.py         # Streamlit web application

````

---

## ⚙️ Local Setup

### ✅ Prerequisites

- Python 3.9 or later
- Git

---

### 🛠 Installation Steps

1. **Clone the Repository:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
````

2. **Create and Activate Virtual Environment:**

```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set Up Google Gemini API Key:**

* Visit [Google AI Studio](https://aistudio.google.com/), log in, and generate an API key.
* Create a `.env` file in your root directory:

```env
GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
```

5. **(Optional) Rebuild the Vector Store:**

If you update or add new documents in `data/`, rebuild the FAISS index:

```bash
python create_vector_db.py
```

---

## 🧪 Running the Chatbot

### 🔸 Option A: Terminal Interface

```bash
python chatbot_app.py
```

Type your questions and get answers in the terminal. Type `exit` to quit.

---

### 🔸 Option B: Web App with Streamlit

```bash
streamlit run streamlit_app.py
```

The app will launch in your default browser (usually at `http://localhost:8501`).

---

## ☁️ Deployment on Streamlit Cloud

### 📋 Requirements

* GitHub account
* Streamlit Cloud account (log in with GitHub)
* Code pushed to a **public GitHub repository**

---

### 🛫 Deployment Steps

1. **Push Your Code to GitHub:**

```bash
git add .
git commit -m "Deploy Healthcare Info Bot"
git push origin main
```

2. **Deploy on [Streamlit Cloud](https://share.streamlit.io/):**

* Click “New app”
* Choose your repo and branch
* Set the **Main file path** as:

```
streamlit_app.py
```

3. **Add API Key in Streamlit Secrets:**

In **Advanced Settings > Secrets**, add your Gemini API key like this:

```env
GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
```

✅ Click **Deploy**, and you’ll get a shareable public link to your healthcare chatbot!

---

## 📌 Notes

* You can extend the bot by adding more `.txt` or `.pdf` files to the `data/` folder and regenerating the vector store.
* All dependencies are pinned in `requirements.txt` for reproducibility.
* Avoid committing the `.env` file; it's excluded by `.gitignore`.


