Okay, fantastic\! I've got your repository link.

Here is the complete `README.md` markdown text, fully customized for your repository: `https://github.com/AbirBanerjee1223/celebal-8-med_bot`.

Copy and paste this entire content into your `README.md` file in the root of your `celebal-8-med_bot` repository.

-----

```markdown
# ⚕️ Healthcare Info Bot (RAG Q&A Chatbot)

This project implements a Retrieval-Augmented Generation (RAG) Q&A chatbot designed to provide information on common healthcare topics. It leverages document retrieval from a local knowledge base and the power of generative AI (Google Gemini API) to provide intelligent, context-aware responses.

The bot can be run locally via a terminal interface or deployed as an interactive web application using Streamlit.

## ✨ Features

* **Retrieval-Augmented Generation (RAG):** Combines information retrieval with a Large Language Model for accurate, fact-checked responses.
* **Generative AI:** Powered by the **Google Gemini 1.5 Flash API** for intelligent and human-like answer generation.
* **Local Knowledge Base:** Uses a FAISS vector store built from curated healthcare documents (provided in the `data/` directory).
* **FAISS Vector Store:** Efficiently stores and retrieves document embeddings for quick context lookup. The vector store (`vectorstore/db_faiss`) is committed directly to this repository due to its small size (~500KB) for simplified deployment.
* **Chat History & Memory:** The Streamlit web application remembers previous conversation turns for a more natural interaction.
* **Streamlit Web UI:** An intuitive, interactive web interface for the chatbot.
* **Terminal Interface:** A command-line version (`chatbot_app.py`) for quick local testing and debugging.

## 🧠 Knowledge Base Topics

This bot is equipped to answer questions on the following healthcare topics, drawing information directly from its internal knowledge base:

* **Common Illnesses & Conditions:**
    * Anxiety (Basics)
    * Asthma (Basic Info)
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
    * Basic First Aid (Minor Cuts, Scrapes, Burns, Sprains)
    * Basic Nutrition Guidelines
    * Mental Health Basics
    * Sleep Hygiene (Tips for Better Sleep)
    * Vaccination Basics

**Disclaimer:** This bot provides general health information for educational purposes and should not replace professional medical advice. Always consult a qualified healthcare professional for diagnosis, treatment, and medical advice.

## 🚀 Technologies Used

* **Python 3.9+**
* **LangChain:** Framework for building LLM applications.
* **Streamlit:** For creating the interactive web UI.
* **Google Gemini API:** The Generative AI model backend.
* **Hugging Face `sentence-transformers`:** For creating document embeddings.
* **FAISS:** For efficient similarity search in the vector store.
* `python-dotenv`: For secure management of API keys.

## 📦 Project Structure

```

celebal-8-med\_bot/
├── data/
│   ├── anxiety\_basics.txt
│   ├── breast\_cancer.txt
│   └── ... (all your healthcare text/pdf files)
├── vectorstore/
│   └── db\_faiss/       \<-- Pre-built FAISS vector store (committed to repo)
│       ├── index.faiss
│       └── index.faiss.metadata.json
├── .env                \<-- Your API key (local only, NOT committed to Git)
├── requirements.txt    \<-- Project dependencies
├── create\_vector\_db.py \<-- Script to create/update the vector store locally
├── chatbot\_app.py      \<-- Terminal-based Q\&A bot
└── streamlit\_app.py    \<-- Streamlit web application

````

## ⚙️ Local Setup and Run

Follow these steps to get the bot running on your local machine:

### Prerequisites

* Python 3.9 or higher installed.
* `git` installed.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/AbirBanerjee1223/celebal-8-med_bot.git](https://github.com/AbirBanerjee1223/celebal-8-med_bot.git)
    cd celebal-8-med_bot
    ```

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
    * Create a file named `.env` in the root of your `celebal-8-med_bot` directory (the same folder as `requirements.txt`).
    * Add your API key to the `.env` file like this:
        ```
        GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
        ```
        **Important:** Do NOT commit your `.env` file to GitHub! It's already included in the `.gitignore` provided.

5.  **Update the Vector Store (Optional):**
    * This repository already includes a pre-built `vectorstore/db_faiss` folder. You generally don't need to run `create_vector_db.py` on the first setup.
    * However, if you add or modify documents in the `data/` folder, you *must* run this script to update your knowledge base:
        ```bash
        python create_vector_db.py
        ```
    * After running, you would then need to commit the updated `vectorstore/db_faiss` to your Git repository for the changes to reflect in deployed versions.

### Running the Chatbot

You have two options to run the bot locally:

#### a) Terminal-based Chatbot (`chatbot_app.py`)

For quick testing in your command line:
```bash
python chatbot_app.py
````

Type your questions at the prompt. Type `exit` to quit.

#### b) Streamlit Web Application (`streamlit_app.py`)

For the full interactive web UI:

```bash
streamlit run streamlit_app.py
```

This will open the application in your default web browser (usually at `http://localhost:8501`).

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
      * Connect your your GitHub account and select your repository: `AbirBanerjee1223/celebal-8-med_bot`.
      * Configure the deployment settings:
          * **Repository:** `AbirBanerjee1223/celebal-8-med_bot`
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
      * Click "Deploy\!"

Streamlit Cloud will now build and deploy your application. The first deployment might take a few minutes as it installs all dependencies and sets up the environment. Once deployed, you'll get a unique URL to access your live chatbot\!

