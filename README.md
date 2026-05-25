# AI-Resume-Analyzer

## 🎯 The Why
An intelligent recruitment tool that leverages Natural Language Processing (NLP) to bridge the gap between candidates and job descriptions by automating resume screening and ATS scoring.

## 🚀 Key Features
* **Automated Parsing**: Extracts text from PDF resumes accurately.
* **ATS Scoring**: Calculates a keyword-based score to determine resume optimization.
* **Job Matching**: Uses TF-IDF and Cosine Similarity to find the best role matches.
* **Skill Extraction**: Identifies technical competencies automatically.

## 🛠 Tech Stack
| Category | Technology |
| :--- | :--- |
| **Backend** | Flask (Python) |
| **AI/NLP** | Scikit-Learn (TF-IDF), PyPDF2 |
| **Data Processing**| Regex, NumPy |

## ⚙️ How to Run
1. **Install requirements**:
   `pip install flask PyPDF2 scikit-learn`
2. **Run the application**:
   `python app.py`
3. **Access the interface**:
   Open `http://127.0.0.1:5000` in your browser.

## 🔮 Future Scope
* **OpenAI API Integration**: Transition from keyword matching to Large Language Model (LLM) based evaluation for better context understanding.
* **Database Persistence**: Move from a local file system to a PostgreSQL database for historical candidate tracking.
