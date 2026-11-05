# Household Electricity Consumption Prediction

A Machine Learning project that predicts household electricity consumption using multiple regression algorithms and compares their performance to identify the most accurate model.

## Project Overview

Household electricity consumption varies based on several factors, making accurate prediction valuable for understanding energy usage.

This project explores machine learning techniques to predict **Global Active Power (kW)** and evaluates different regression models to determine the best-performing approach.

## Objectives

- Analyze household electricity consumption data.
- Perform data preprocessing and exploratory data analysis.
- Train and evaluate multiple regression models.
- Compare model performance using evaluation metrics.
- Develop an interactive Streamlit application for electricity consumption prediction.

## Machine Learning Models

The following algorithms were implemented and compared:

- K-Nearest Neighbors (KNN)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| KNN | 0.099432 | 0.147579 | 0.972872 |
| Linear Regression | 0.024379 | 0.039647 | 0.998042 |
| Decision Tree | 0.028024 | 0.047449 | 0.997196 |
| Random Forest | 0.020223 | 0.033080 | 0.998637 |

### Best Performing Model

**Random Forest Regressor**

- R² Score: 99.86%
- MAE: 0.020223
- RMSE: 0.033080

Random Forest achieved the best overall performance among the evaluated models, with the lowest MAE and RMSE and the highest R² score.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Household-Electricity-Consumption-Prediction/
│
├── Electricity_Consumption_Prediction_(2).ipynb
├── app.py
├── README.md
├── LICENSE
└── .gitignore
```

## Streamlit Application

The project includes an interactive Streamlit application for exploring electricity consumption predictions.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/rosee31/Household-Electricity-Consumption-Prediction.git
```

Navigate to the project directory:

```bash
cd Household-Electricity-Consumption-Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

Run the application:

```bash
streamlit run app.py
```

## Key Learnings

- Data preprocessing and exploratory data analysis
- Regression model implementation
- Model evaluation and performance comparison
- Understanding prediction accuracy through regression metrics
- Building an interactive ML application using Streamlit

## Future Improvements

- Deploy the Streamlit application online.
- Explore additional machine learning models.
- Improve prediction performance through hyperparameter tuning.
- Enhance the dashboard with additional visualizations.

## Author

**Rose Babu**

Data Science | Machine Learning | Data Analytics

GitHub: [@rosee31](https://github.com/rosee31)

## Acknowledgements

Grateful to **Jasmin Ma’am at Luminar Technolab** for her guidance and support throughout my Machine Learning journey.

