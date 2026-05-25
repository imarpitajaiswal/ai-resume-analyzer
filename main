import os
import re
import logging
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- APP INITIALIZATION ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-123')

# --- CONFIG ---
UPLOAD_FOLDER = os.path.join('data', 'resumes')
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# --- DATASETS ---
ATS_KEYWORDS = {"python": 15, "machine learning": 20, "sql": 15, "cloud": 12, "aws": 15}
JOB_DESCRIPTIONS = {
    "ML Engineer": "machine learning python tensorflow deep learning data science",
    "Data Analyst": "sql excel python data visualization pandas statistics",
    "Software Developer": "java python javascript web development backend api"
}
SKILLS_DB = {"python", "java", "sql", "aws", "docker", "flask", "react", "machine learning"}

# --- UTILS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.lower()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).lower().strip()

def extract_skills(text):
    found = {skill for skill in SKILLS_DB if re.search(r'\b' + re.escape(skill) + r'\b', text)}
    return sorted(list(found))

def calculate_ats_score(text):
    score = sum(ATS_KEYWORDS.get(kw, 0) for kw in ATS_KEYWORDS if kw in text)
    return min(int(score / 5), 100)

def match_jobs(text):
    docs = [text] + list(JOB_DESCRIPTIONS.values())
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(docs)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return sorted([{"role": k, "match_score": round(s*100, 2)} for k, s in zip(JOB_DESCRIPTIONS.keys(), scores)], 
                  key=lambda x: x["match_score"], reverse=True)

# --- ROUTES ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume" not in request.files: return redirect(url_for("home"))
    file = request.files["resume"]
    if file and allowed_file(file.filename):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(path)
        
        text = clean_text(extract_text_from_pdf(path))
        return render_template("results.html", 
                               skills=extract_skills(text), 
                               ats_score=calculate_ats_score(text), 
                               job_matches=match_jobs(text))
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
