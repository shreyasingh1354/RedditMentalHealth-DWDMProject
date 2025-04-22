# Reddit Mental Health Analysis

## 📋 Overview
This project analyzes Reddit posts from mental health subreddits to identify patterns, trends, and insights that could help understand mental health discussions online. Using advanced natural language processing, machine learning techniques, and data visualization, we classify posts into different mental health categories and extract meaningful insights from user discussions.


## ✨ Features
- **Data Collection**: Gathered posts from various mental health subreddits
- **Data Pre-processing**: Clean, normalize, and prepare text data for analysis
- **Feature Engineering**: Extract meaningful features from post text using NLP techniques
- **Semantic Sampling**: Intelligently sample large datasets while preserving their semantic and temporal distribution
- **Text Classification**: Categorize posts into different mental health topics
- **Advanced Visualization**: Analyze patterns, trends, and distributions in the dataset
- **Model Evaluation**: Rigorous assessment of classification performance

## 🗂️ Mental Health Categories
Our analysis focuses on several key mental health topics including:
- Drug and Alcohol related discussions
- Existential/Life concerns
- Psychological Factors
- Temporal/Situational issues

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.11+
- Required libraries:
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  sentence-transformers
  tqdm
  nltk
  spacy
  ```

### 💻 Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/reddit-mental-health-analysis.git
cd reddit-mental-health-analysis
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```


## 🔄 Analysis Process
1. **📥 Data Collection**: Reddit posts from mental health subreddits
2. **🧹 Pre-processing**: Cleaning text, handling missing values, normalization
3. **🔍 Feature Extraction**: TF-IDF, embeddings, linguistic features
4. **🏗️ Model Building**: Training classification models
5. **📏 Evaluation**: Performance metrics, confusion matrices, validation
6. **📊 Visualization**: Distribution analysis, temporal trends, topic modeling

## 💡 Insights
The project aims to discover:
- Common linguistic patterns in mental health discussions
- How different mental health topics correlate
- Temporal trends in mental health discussions
- Effectiveness of different classification approaches


## 👥 Contributing
Contributions to improve the analysis or extend the project are welcome! Please feel free to submit a pull request or open an issue.

---

**⚠️ Important Note**: This project is for research purposes only and should not be used for diagnosing mental health conditions. Always consult with healthcare professionals for mental health concerns.