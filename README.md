Markdown

# ⚕️ Healthcare Info Bot (RAG Q&A Chatbot)

This project implements a Retrieval-Augmented Generation (RAG) Q&A chatbot designed to provide information on common healthcare topics. It leverages document retrieval from a local knowledge base and the power of generative AI (Google Gemini API) to provide intelligent, context-aware responses.

The bot can be run locally via a terminal interface or deployed as an interactive web application using Streamlit.

## ✨ Features

* **Retrieval-Augmented Generation (RAG):** Combines information retrieval with a Large Language Model for accurate, fact-checked responses.
* **Generative AI:** Powered by the **Google Gemini 1.5 Flash API** for intelligent and human-like answer generation.
* **Local Knowledge Base:** Uses a FAISS vector store built from curated healthcare documents (provided in the `data/` directory).
* **FAISS Vector Store:** Efficiently stores and retrieves document embeddings for quick context lookup.
* **Chat History & Memory:** The Streamlit web application remembers previous conversation turns for a more natural interaction.
* **Streamlit Web UI:** An intuitive, interactive web interface for the chatbot.
* **Terminal Interface:** A command-line version (`chatbot_app.py`) for quick local testing and debugging.
* **Self-Contained Deployment:** Ready for easy deployment to Streamlit Cloud, with the vector store committed directly to the repository (due to its small size).

## 🧠 Knowledge Base Topics

This bot is equipped to answer questions on the following healthcare topics (based on the documents in the `data/` folder):

* **Common Illnesses:**
    * Anxiety (Basics)
    * Asthma (Basics)
    * Breast Cancer
    * Common Cold
    * Dengue Fever
    * Diabetes (Type 1 & 2 Basics)
    * Headaches (Common Types: Tension, Migraine, Cluster)
    * Hypertension (High Blood Pressure)
    * Malaria (Basics)
    * Pneumonia (Children)
    * Polycystic Ovary Syndrome (PCOS)
    * Post-Traumatic Stress Disorder (PTSD)
* **General Health & Wellness:**
    * Basic Nutrition Guidelines
    * Sleep Hygiene (Tips for Better Sleep)
    * Mental Health Basics
    * Basic First Aid (Minor Cuts, Scrapes, Burns, Sprains)
    * Vaccination Basics

**Disclaimer:** This bot provides general health information and should not replace professional medical advice. Always consult a healthcare professional for diagnosis and treatment.

## 🚀 Technologies Used

* **Python 3.9+**
* **LangChain:** Framework for building LLM applications.
* **Streamlit:** For creating the interactive web UI.
* **Google Gemini API:** The Generative AI model backend.
* **Hugging Face `sentence-transformers`:** For creating document embeddings (via `HuggingFaceEmbeddings`).
* **FAISS:** For efficient similarity search in the vector store.
* `python-dotenv`: For secure management of API keys.

## 📦 Project Structure

healthcare_info_bot/
├── data/
│   ├── anxiety_basics.txt
│   ├── breast_cancer.txt
│   └── ... (all your healthcare text/pdf files)
├── vectorstore/
│   └── db_faiss/       <-- Pre-built FAISS vector store (committed to repo)
├── .env                <-- Your API key (NOT committed to Git)
├── requirements.txt    <-- Project dependencies
├── create_vector_db.py <-- Script to create/update the vector store locally
├── chatbot_app.py      <-- Terminal-based Q&A bot
└── streamlit_app.py    <-- Streamlit web application


## ⚙️ Local Setup and Run

Follow these steps to get the bot running on your local machine:

### Prerequisites

* Python 3.9 or higher installed.
* `git` installed.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details)

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Google Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/) and log in with your Google account.
    * Generate a new API key.
    * Create a file named `.env` in the root of your `healthcare_info_bot` directory (the same folder as `requirements.txt`).
    * Add your API key to the `.env` file like this:
        ```
        GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
        ```
        **Important:** Do NOT commit your `.env` file to GitHub! It's already included in the `.gitignore` provided.

5.  **Prepare the Vector Store (If not already committed):**
    * Since this repository *includes* the `vectorstore/db_faiss` folder (as it's small), you generally don't need to run `create_vector_db.py` on the first setup.
    * However, if you add or modify documents in the `data/` folder, you *must* run this script to update your knowledge base:
        ```bash
        python create_vector_db.py
        ```
    * After running, you would then need to commit the updated `vectorstore/db_faiss` to your Git repository.

### Running the Chatbot

You have two options to run the bot locally:

#### a) Terminal-based Chatbot (`chatbot_app.py`)

For quick testing in your command line:
```bash
python chatbot_app.py
```
Type your questions at the prompt. Type exit to quit.

b) Streamlit Web Application (streamlit_app.py)
For the full interactive web UI:

```Bash

streamlit run streamlit_app.py
```
This will open the application in your default web browser (usually at http://localhost:8501).

## ☁️ Deployment to Streamlit Cloud

Your Healthcare Info Bot is set up for easy deployment as a web application on Streamlit Cloud.

### Prerequisites for Deployment

* A GitHub account.
* A Streamlit Cloud account (you can sign up with your GitHub account).
* Your project code, including the `data/` folder and the `vectorstore/db_faiss` folder, committed and pushed to a **public** GitHub repository.

### Deployment Steps

1.  **Push Your Code to GitHub:**
    Ensure all your project files (including the `data/` folder and the `vectorstore/db_faiss` folder) are committed and pushed to your GitHub repository's `main` branch.

    ```bash
    git add .
    git commit -m "Ready for Streamlit Cloud deployment"
    git push origin main
    ```

2.  **Deploy on Streamlit Cloud:**
    * Go to [Streamlit Cloud](https://share.streamlit.io/).
    * Log in to your workspace.
    * Click on "New app" in your workspace.
    * Select "From existing repo".
    * Connect your GitHub account and select your repository for the Healthcare Info Bot.
    * Configure the deployment settings:
        * **Repository:** `YOUR_USERNAME/YOUR_REPOSITORY_NAME` (e.g., `my-github-user/healthcare-info-bot`)
        * **Branch:** `main` (or the branch where your code is)
        * **Main file path:** `streamlit_app.py`
    * **Add Secrets (Important for API Key):**
        * Expand the "Advanced settings" section.
        * In the "Secrets" text area, add your `GOOGLE_API_KEY`.
        * **Crucially, enter it exactly like this (without quotes around the key itself):**
            ```
            GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
            ```
            (Replace `YOUR_ACTUAL_GEMINI_API_KEY_HERE` with the actual API key you generated from Google AI Studio.)
    * Click "Deploy!"

Streamlit Cloud will now build and deploy your application. The first deployment might take a few minutes as it installs all dependencies and sets up the environment. Once deployed, you'll get a unique URL to access your live chatbot!
