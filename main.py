# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import tldextract
from typing import Optional
import re
import joblib # For loading ML models
import numpy as np # For numerical operations with ML probabilities

app = FastAPI(title="Website Type Classifier API")

class Website(BaseModel):
    url: str

# Comprehensive website types with refined keywords
WEBSITE_TYPES = {
    "e-commerce": ["shop", "cart", "buy", "store", "product", "price", "checkout", "add to cart", "payment", "deal", "coupon", "discount"],
    "blog": ["blog", "post", "article", "read more", "comment", "author", "category", "tag", "subscribe", "newsletter"],
    "news": ["news", "breaking", "headline", "report", "latest", "update", "journal", "editorial", "politics", "economy", "world"],
    "portfolio": ["portfolio", "project", "work", "gallery", "showcase", "creative", "design", "artist", "resume"],
    "forum": ["forum", "thread", "post", "discussion", "reply", "community", "board", "group", "topic", "members"],
    "corporate": ["company", "about us", "services", "contact us", "careers", "business", "enterprise", "solution", "investor"],
    "personal": ["about me", "personal", "bio", "resume", "cv", "profile", "hobby", "my story"],
    "educational": ["course", "learn", "university", "school", "education", "study", "academy", "lecture", "student", "syllabus"],
    "government": ["gov", "government", "official", "public", "department", "agency", "state", "policy", "citizen", "council"],
    "non-profit": ["donate", "charity", "non-profit", "volunteer", "mission", "cause", "foundation", "support", "fundraiser"],
    "social media": ["follow", "share", "like", "post", "profile", "connect", "network", "friend", "feed", "community"],
    "entertainment": ["video", "movie", "music", "stream", "watch", "play", "entertainment", "show", "artist", "album"],
    "wiki": ["wiki", "encyclopedia", "knowledge", "edit", "reference", "information", "article", "fandom"],
    "job board": ["job", "career", "hiring", "vacancy", "apply", "recruitment", "employment", "resume", "positions", "openings"],
    "directory": ["directory", "listing", "search", "find", "businesses", "categories", "reviews", "local", "contact"],
    "health": ["health", "medical", "doctor", "hospital", "wellness", "clinic", "patient", "therapy", "disease", "symptom"],
    "travel": ["travel", "tour", "booking", "destination", "hotel", "flight", "vacation", "itinerary", "explore", "journey"],
    "real estate": ["property", "real estate", "listing", "home", "rent", "buy", "mortgage", "broker", "house", "apartment"],
    "video streaming": ["stream", "video", "watch", "channel", "subscribe", "live", "episode", "series", "tv"],
    "gaming": ["game", "gaming", "play", "score", "leaderboard", "multiplayer", "console", "esports", "gamer", "level"],
    "event": ["event", "ticket", "festival", "conference", "seminar", "webinar", "schedule", "register", "date"],
    "food": ["recipe", "food", "cooking", "restaurant", "menu", "cuisine", "dine", "chef", "ingredient"],
    "sports": ["sport", "team", "score", "league", "match", "athlete", "tournament", "game", "championship"],
}

# --- Load the pre-trained ML model and vectorizer ---
vectorizer = None
model = None
trained_classes = []

try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('website_classifier_model.pkl')
    trained_classes = model.classes_ # Get the classes the model was trained on
    print("--- Machine learning model loaded successfully. ---")
except FileNotFoundError:
    print("\n--- WARNING: ML model files (tfidf_vectorizer.pkl, website_classifier_model.pkl) not found. ---")
    print("--- Please run 'python train_model.py' first to train and save the model. ---")
    print("--- The API will fall back to heuristic classification only, which may be less accurate. ---\n")
except Exception as e:
    print(f"\n--- ERROR loading ML model: {e}. Falling back to heuristic classification only. ---\n")

def classify_website(url: str) -> dict:
    # Ensure URL has scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # Extract domain information
        extracted = tldextract.extract(url)
        suffix = extracted.suffix

        # Fetch website content
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Parse HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text and metadata
        text = soup.get_text(separator=" ", strip=True).lower()
        title = soup.find("title").get_text().lower() if soup.find("title") else ""
        meta_description = ""
        for tag in soup.find_all("meta"):
            if tag.get("name") == "description":
                meta_description = tag.get("content", "").lower()
                break
        
        # Clean up excessive whitespace
        combined_text_raw = f"{title} {meta_description} {text}"
        combined_text = re.sub(r'\s+', ' ', combined_text_raw).strip()

        # Initialize heuristic scores
        heuristic_scores = {type_name: 0 for type_name in WEBSITE_TYPES}

        # Domain-based heuristics
        if suffix in ["edu", "ac"]:
            heuristic_scores["educational"] += 3
        if suffix == "gov":
            heuristic_scores["government"] += 3
        if suffix == "org" and "donate" in combined_text:
            heuristic_scores["non-profit"] += 2

        # Keyword-based scoring (using a portion of text for efficiency)
        text_for_heuristics = combined_text[:5000] # Use first 5000 chars for heuristic speed
        for type_name, keywords in WEBSITE_TYPES.items():
            for keyword in keywords:
                if keyword in text_for_heuristics:
                    heuristic_scores[type_name] += 1

        # Structural checks using regex for class names
        if soup.find_all(["form", "input", "select"]) and any(k in text_for_heuristics for k in ["cart", "checkout"]):
            heuristic_scores["e-commerce"] += 3
        if soup.find_all(["article", "section"]) and "blog" in text_for_heuristics:
            heuristic_scores["blog"] += 3
        if soup.find_all(["div"], class_=re.compile(r"gallery|portfolio")):
            heuristic_scores["portfolio"] += 3
        if soup.find_all(["form"], class_=re.compile(r"search")) and "directory" in text_for_heuristics:
            heuristic_scores["directory"] += 3
        if soup.find_all(["div"], class_=re.compile(r"job|career|hiring|vacanc")):
            heuristic_scores["job board"] += 3
        if soup.find_all(["a"], href=re.compile(r"wiki|edit")):
            heuristic_scores["wiki"] += 3
        if soup.find_all(["button", "a"], string=re.compile(r"donate", re.I)):
            heuristic_scores["non-profit"] += 3
        if soup.find_all(["video", "iframe"]) and any(k in text_for_heuristics for k in ["stream", "watch"]):
            heuristic_scores["video streaming"] += 3
        if soup.find_all(["div"], class_=re.compile(r"game|score|gaming")):
            heuristic_scores["gaming"] += 3
        if soup.find_all(["a", "button"], string=re.compile(r"ticket|event|conference|festival", re.I)):
            heuristic_scores["event"] += 3

        # --- ML-based Prediction ---
        ml_type_prediction = "unknown"
        ml_confidence_raw = 0.0
        
        ML_CONFIDENCE_THRESHOLD = 0.80 # Threshold for prioritizing ML prediction

        if model and vectorizer:
            try:
                # Transform the combined_text (full text) for ML prediction
                text_features = vectorizer.transform([combined_text])
                
                # Get probability predictions for all classes
                probabilities = model.predict_proba(text_features)[0]
                
                # Find the class with the highest probability
                max_prob_idx = np.argmax(probabilities)
                ml_type_prediction = trained_classes[max_prob_idx]
                ml_confidence_raw = probabilities[max_prob_idx]
                    
            except Exception as e:
                print(f"DEBUG: Error during ML prediction for {url}: {e}")
                # ML prediction failed, will proceed with heuristics
        
        # --- Final Determination ---
        final_type = "unknown"
        final_confidence = 0.0

        # Prioritize ML prediction if it's confident enough
        if ml_confidence_raw >= ML_CONFIDENCE_THRESHOLD:
            final_type = ml_type_prediction
            final_confidence = float(ml_confidence_raw)
        else:
            # Fallback to heuristic scores if ML is not confident or not available
            max_heuristic_score = max(heuristic_scores.values())

            if max_heuristic_score > 0:
                best_heuristic_type = max(heuristic_scores, key=heuristic_scores.get)
                total_heuristic_score = sum(heuristic_scores.values())
                
                # Normalize heuristic score to give an approximate confidence (0-1)
                # This is a simple normalization; a more complex one might use domain knowledge
                # For example, a perfect heuristic score might get 0.7 confidence by default
                heuristic_confidence_normalized = (max_heuristic_score / total_heuristic_score) if total_heuristic_score > 0 else 0.0
                
                # If ML was attempted but not confident, average with heuristic or take the best
                if model and ml_confidence_raw > 0: # If ML provided some (low confidence) probability
                    final_confidence = np.mean([heuristic_confidence_normalized, ml_confidence_raw])
                    # If ML has a specific type, but heuristic is also strong, prefer ML if it's not too far off
                    if ml_type_prediction != "unknown" and ml_confidence_raw > heuristic_confidence_normalized:
                        final_type = ml_type_prediction
                    else:
                        final_type = best_heuristic_type
                else: # Only heuristic is available
                    final_type = best_heuristic_type
                    final_confidence = heuristic_confidence_normalized
            
            # If still no strong signal, default to unknown
            if final_type == "unknown" and final_confidence == 0.0 and ml_confidence_raw < ML_CONFIDENCE_THRESHOLD:
                 # If even after averaging, confidence is low, and ML wasn't confident.
                 # Consider the highest heuristic as a "best guess" but with low confidence.
                 if max_heuristic_score > 0:
                    final_type = max(heuristic_scores, key=heuristic_scores.get)
                    final_confidence = max_heuristic_score / (sum(WEBSITE_TYPES[final_type]) * 2) if sum(WEBSITE_TYPES[final_type]) > 0 else 0.0 # Example simple scaling
                    final_confidence = min(0.49, final_confidence) # Ensure it's explicitly below 0.5 if not confident
                 else:
                     return {"url": url, "type": "unknown", "confidence": 0.0}

        return {"url": url, "type": final_type, "confidence": round(float(final_confidence), 2)}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error accessing website {url}: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing {url}: {str(e)}")

@app.post("/classify", response_model=dict)
async def classify_website_type(website: Website):
    """
    Classify the type of a website based on its URL.
    Returns the predicted website type and confidence score.
    """
    result = classify_website(website.url)
    return result

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """
    Serve the React-based UI for the website type classifier.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Website Type Classifier</title>
        <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.2/dist/axios.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.23.6/babel.min.js"></script>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center">
        <div id="root"></div>
        <script type="text/babel">
            function App() {
                const [url, setUrl] = React.useState('');
                const [result, setResult] = React.useState(null);
                const [error, setError] = React.useState(null);
                const [loading, setLoading] = React.useState(false);

                const handleSubmit = async () => {
                    setError(null);
                    setResult(null);
                    setLoading(true);

                    try {
                        const response = await axios.post('/classify', { url });
                        setResult(response.data);
                    } catch (err) {
                        setError(err.response?.data?.detail || 'An error occurred while classifying the website.');
                    } finally {
                        setLoading(false);
                    }
                };

                return (
                    <div className="max-w-lg w-full mx-auto p-6 bg-white rounded-lg shadow-lg">
                        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">Website Type Classifier</h1>
                        <div className="mb-4">
                            <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
                                Enter Website URL
                            </label>
                        <input
                                type="text"
                                id="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="e.g., https://www.example.com"
                                className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>
                        <button
                            onClick={handleSubmit}
                            disabled={loading || !url}
                            className={`w-full py-3 rounded-lg text-white font-semibold transition-colors ${
                                loading || !url ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                            }`}
                        >
                            {loading ? 'Classifying...' : 'Classify Website'}
                        </button>
                        {result && (
                            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                                <h2 className="text-lg font-semibold text-gray-800">Result</h2>
                                <p className="text-gray-600">
                                    <span className="font-medium">URL:</span> {result.url}
                                </p>
                                <p className="text-gray-600">
                                    <span className="font-medium">Type:</span> {result.type.charAt(0).toUpperCase() + result.type.slice(1)}
                                </p>
                                <p className="text-gray-600">
                                    <span className="font-medium">Confidence:</span> {(result.confidence * 100).toFixed(0)}%
                                </p>
                            </div>
                        )}
                        {error && (
                            <div className="mt-6 p-4 bg-red-50 rounded-lg">
                                <p className="text-red-600">{error}</p>
                            </div>
                        )}
                        <p className="mt-6 text-center text-sm text-gray-500">
                            Powered by FastAPI & xAI
                        </p>
                    </div>
                );
            }

            ReactDOM.render(<App />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)