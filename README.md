# Consumer Complaint Sentiment Insights and Classification System

## Context & Motivation
The Consumer Financial Protection Bureau (CFPB) collects thousands of consumer complaints across financial services. Understanding the emotional intensity of these complaint narratives (Neutral, Negative, Extreme Negative) enables us to:
- Detect systemic issues in financial services  
- Improve customer-service strategies at institutions  
- Inform policy decisions and enforcement priorities  
- Prioritize high-risk cases for investigation  

##  Problem Statement
How can we leverage NLP and deep learning to build a scalable, production-ready pipeline that classifies consumer-complaint narratives into three sentiment levels:  
1. **Neutral**  
2. **Negative**  
3. **Extreme Negative**  

---

## Project Workflow & Sprints

|          | Objective                                                                                       | Status               |
|----------|-------------------------------------------------------------------------------------------------|----------------------|
| **1**    | Load & explore raw dataset; perform basic structural cleaning                                    | ✅ Complete          |
| **2**    | Clean & normalize complaint text; engineer initial “weak” sentiment labels                       | ✅ Complete          |
| **3**    | Exploratory Data Analysis on product, company, sentiment & interaction features (validate H1–H5) | ✅ Complete          |
| **4**    | Train baseline BiLSTM deep-learning model for sentiment classification                            | ✅ Complete          |
| **5**    | Test & demonstrate the trained BiLSTM model (predictions, confidence, interpretability)           | ⏳ In Progress / TODO |
| **6**    | Develop interactive Tableau dashboards for stakeholder insights (geography, sentiment, product)   | ⏳ TODO              |
| **7**    | Wrap up: organize GitHub repo, document structure, summarize results & findings                   | ⏳ TODO              |

---

##  Hypotheses Explored

| ID  | Hypothesis                                                                                         |
|-----|----------------------------------------------------------------------------------------------------|
| H1  | Products like **Credit Reporting** and **Debt Collection** exhibit higher Extreme-Negative rates   |
| H2  | Longer narratives correlate with higher emotional intensity                                        |
| H3  | Certain companies show systemic patterns of Extreme-Negative sentiment                             |
| H4  | Trigger keywords (e.g., “fraud”, “lawsuit”) strongly correlate with Extreme-Negative sentiment     |
| H5  | **Timely response** has only a minor effect on sentiment compared to the quality of the experience |

---

##  Tools & Technologies

- **Language & Notebooks:** Python 3.8 (via Pyenv), Jupyter Notebooks  
- **Data & Visualization:** pandas, NumPy, Matplotlib, Seaborn  
- **NLP Preprocessing:** NLTK, SpaCy, TextBlob  
- **Modeling:** TensorFlow / Keras (BiLSTM)  
- **Dashboarding:** Tableau  
- **Version Control & Deployment:** Git & GitHub  

---
