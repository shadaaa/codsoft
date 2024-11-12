# Titanic Survival Prediction

### Project Overview
The Titanic Survival Prediction project is a machine learning model built to predict if a passenger would survive based on features such as age, gender, and title. The model is trained using the Titanic dataset and utilizes a Random Forest Classifier.

### Project Structure
- `titanic_model.py`: Contains the code for loading, preprocessing the data, training the model, and predicting survival based on a passenger's name.


### Prerequisites
- Python 3.x
- Libraries: `pandas`, `sklearn`, `numpy`

### Setup and Usage
1. Install Dependencies:
    pip install pandas scikit-learn numpy

3. How It Works:
    - The function `predict_survival` takes the passenger's name, extracts the title, gender, and age, encodes the features, and uses the trained model to predict survival.
