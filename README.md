## PROJECT TITLE –
# Fake News Detection Using Machine Learning.



## PROJECT DESCRIPTION – 
The Fake News Detection Using Machine Learning project is a Python-based application that identifies whether a given news article is FAKE or REAL using Machine Learning and Natural Language Processing (NLP).



## OBJECTIVE –
The main objective of the Fake News Detection Using Machine Learning project is to develop an intelligent system that can automatically identify whether a given news article is FAKE or REAL.



## TECHNOLOGIES USED –
Programming Language: Python

                   - Data Handling: Pandas, NumPy

                   - Machine Learning: Scikit-learn

                   - Feature Extraction: TF-IDF Vectorizer

                   - Classification Algorithm: Logistic Regression

                   - Data Visualization: Matplotlib, Seaborn

                   - Model Saving and Loading: Joblib

                   - Web Framework: Streamlit

## TEXT PREPROCESSING: 
                      -Natural Language Processing (NLP)



## PROJECT STRUCTURE – 

fake-news-detection/
│
├── dataset/
│
├── 1_load_dataset.py
├── 2_preprocessing.py
├── 3_visualization.py
├── 4_feature_extraction.py
├── 5_model_training.py
├── 6_model_evaluation.py
├── 7_prediction.py
├── 8_app.py
│
└── fake_news_model.pkl



## 🔄 Project Workflow:

The Fake News Detection system follows these steps:

1. **Dataset Loading:** Load the fake and real news dataset.
2. **Data Preprocessing:** Clean the news text by removing unnecessary characters and extra spaces.
3. **Data Visualization:** Visualize the distribution of fake and real news.
4. **Feature Extraction:** Convert text into numerical features using TF-IDF Vectorization.
5. **Model Training:** Train the Logistic Regression model using the processed dataset.
6. **Model Evaluation:** Evaluate the trained model using performance metrics.
7. **Prediction:** Predict whether the entered news article is FAKE or REAL.
8. **Web Application:** Display the prediction through the Streamlit interface.




## INSTALLATION STEPS – 

Step 1: Install Python
        Download and install Python from https://www.python.org/downloads/.

Step 2: Clone the Repository
        git clone https://github.com/charani-maraka/fake-news-detection.git

Step 3: Navigate to the Project Folder
        cd fake-news-detection

Step 4: Install Required Libraries
        pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn

*After successfully installing the required libraries, the project is ready to run.



## HOW TO RUN ▶️ – 

Step 1: Open the Project
        Open the project folder in Visual Studio Code.

Step 2: Open the Terminal
        Go to Terminal → New Terminal.

Step 3: Run the Streamlit Application
        Execute the following command:
        streamlit run 8_app.py

Step 4: Open the Web Application
        After executing the command, it will automatically open the following URL in your browser:http://localhost:8501

Step 5: Predict News
        - Enter a news article in the input area provided in the dataset.
        - Click the **Predict News** button.
        - View the prediction result displayed by the application.

The application predicts whether the entered news article is **FAKE or REAL**.



## OUTPUT – 
*What happens when someone enters a news article?
->When a user enters a news article into your Fake News Detection web application, the system analyzes the text using a trained Machine Learning  model and predicts whether the news is FAKE or REAL.
