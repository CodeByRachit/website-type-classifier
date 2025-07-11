updated readme.
# Website Type Classifier API 🌐

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust FastAPI application that classifies the type of a given website (e.g., E-commerce, News, Blog, Corporate) using a hybrid approach. It combines powerful Machine Learning models with intelligent heuristic rules for accurate and reliable classification.

## ✨ Features

* **Hybrid Classification Engine**: Utilizes a trained scikit-learn model (Logistic Regression with TF-IDF features) alongside a comprehensive set of heuristic rules (keywords, domain suffixes, structural checks) for enhanced accuracy.
* **FastAPI Backend**: Provides a high-performance, asynchronous API for website classification.
* **Simple Web UI**: Includes a basic, embedded React-based frontend for easy testing and demonstration directly from your browser.
* **Extensible**: Easily extendable with new website types, refined heuristics, or updated ML models by training with more diverse data.
* **Robust Web Scraping**: Handles common issues like missing URL schemes and uses appropriate headers for fetching website content.

## 🚀 Getting Started

Follow these steps to set up and run the Website Type Classifier API on your local machine.

### Prerequisites

* Python 3.10 or higher (as seen in your screenshot, `python 3.10.11`)
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/website-type-classifier.git](https://github.com/your-username/website-type-classifier.git)
    cd website-type-classifier
    ```
    *(Replace `https://github.com/your-username/website-type-classifier.git` with your actual repository URL)*

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**

    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    First, you'll need a `requirements.txt` file (see the "Steps to take" section in the previous response). Once you have it:

    ```bash
    pip install -r requirements.txt
    ```
## 📦 Dependencies Explained

This project relies on the following Python libraries for its functionality, as listed in `requirements.txt`:

* **`fastapi`**: A modern, fast (high-performance) web framework for building APIs with Python 3.8+ based on standard Python type hints. It's the core framework powering your API.
* **`uvicorn`**: An ASGI (Asynchronous Server Gateway Interface) server, used to run FastAPI applications and handle incoming requests.
* **`pydantic`**: Data validation and settings management using Python type hints. Used for defining your `Website` request body, ensuring valid input.
* **`requests`**: An elegant and simple HTTP library for Python, used for making HTTP requests to fetch website content (i.e., sending GET requests to the URLs you want to classify).
* **`beautifulsoup4` (often imported as `bs4`)**: A library for pulling data out of HTML and XML files. It's used to parse the HTML content fetched from websites, allowing you to extract text, titles, meta descriptions, and find specific HTML elements.
* **`tldextract`**: Accurately separates a URL into its subdomain, domain, and top-level domain (TLD). This is crucial for your domain-based heuristics in `main.py`.
* **`scikit-learn` (often imported as `sklearn`)**: A comprehensive machine learning library for Python, providing various classification, regression, and clustering algorithms. It's used in `train_model.py` for TF-IDF vectorization, Logistic Regression model training, and evaluation metrics.
* **`joblib`**: A set of tools to provide lightweight pipelining in Python, primarily used for efficiently saving and loading Python objects to/from disk. In your project, it's essential for persisting your trained TF-IDF vectorizer and the ML classification model (`.pkl` files).
* **`numpy`**: The fundamental package for numerical computing with Python. It's essential for handling arrays and performing mathematical operations, especially when dealing with the numerical outputs and probabilities from your machine learning model.
* **`pandas`**: A powerful and flexible open-source data analysis and manipulation library. It's used in `train_model.py` for handling and processing your training datasets efficiently.

## 🧠 Train the Machine Learning Model

The API relies on a pre-trained ML model (`website_classifier_model.pkl`) and its corresponding vectorizer (`tfidf_vectorizer.pkl`). These files are **not** included in the repository and must be generated by running the training script.

1.  **Run the training script:**
    ```bash
    python train_model.py
    ```
    This script will download some sample data, train the `LogisticRegression` model, and save the `tfidf_vectorizer.pkl` and `website_classifier_model.pkl` files in your project root directory.

    **Note:** The `train_model.py` you provided has a very small dataset for demonstration. For a truly robust and accurate classifier, you'll need significantly more diverse and larger datasets with many examples per category.

## ▶️ Run the FastAPI Application

Once the ML model files are generated, you can start the FastAPI server.

1.  **Ensure your virtual environment is active.** (If not, activate it as shown in "Installation" step 3).

2.  **Start the Uvicorn server:**
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    (The `--reload` flag is useful for development, as it restarts the server automatically on code changes.)

    You should see output indicating that the server is running, typically on `http://127.0.0.1:8000`.

## 🔬 Usage

### Web Interface (UI)

Open your web browser and navigate to:
`http://127.0.0.1:8000/`

You will see a simple input field where you can enter a website URL and click "Classify Website" to see the results.

### API Endpoint (Programmatic Usage)

You can also interact with the API directly using `curl` or any HTTP client (like Postman, Insomnia, or a Python `requests` script).

**Endpoint:** `POST /classify`
**Content-Type:** `application/json`

**Example Request:**

```bash
curl -X POST "[http://127.0.0.1:8000/classify](http://127.0.0.1:8000/classify)" \
     -H "Content-Type: application/json" \
     -d '{"url": "[https://www.nytimes.com/](https://www.nytimes.com/)"}'









My Project Showcase

Here are some key screenshots from the project:

![google mail sample ](https://github.com/user-attachments/assets/e28b2077-a681-4d5a-9961-806e54e9e92f)


![news sample ](https://github.com/user-attachments/assets/73e65f2e-e28f-49fe-abd4-499c6cbf4cbc)


![youtube sample ](https://github.com/user-attachments/assets/6b244ac2-b2e1-4f1f-b8fa-ffb8befd1d27)

