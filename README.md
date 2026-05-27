# 📄 AI Resume Analyzer (NLP & Machine Learning)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/UI-Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

A lightweight, machine-learning-powered web application that analyzes PDF resumes against target tech roles. 

**Why this architecture?** Instead of relying on expensive, high-latency external LLM APIs, this system leverages native **Natural Language Processing (NLP)** techniques and vector space mathematics to provide immediate, deterministic, and mathematically grounded feedback on a candidate's resume optimization.

---

## ✨ Key Features

* **Zero-Latency Execution:** Runs entirely on local machine learning models with zero dependency on third-party LLMs.
* **Instant Document Parsing:** Extracts, cleans, and normalizes raw text from complex PDF binaries instantly using RegEx.
* **Smart ATS Scoring:** Evaluates resume viability utilizing a weighted, domain-specific keyword extraction algorithm.
* **Precision Role Alignment:** Calculates exact mathematical match percentages against specific job descriptions (e.g., Machine Learning Engineer, Data Analyst, Software Developer).

---

## 🧠 Core ML Pipeline & Architecture

This application utilizes vector space modeling for highly
