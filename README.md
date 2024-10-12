# GST Hackathon Hybrid Model

## Description
This project implements a hybrid machine learning model using Support Vector Machine (SVM) and Gradient Boosting for predicting GST outcomes based on a provided dataset. The dataset consists of training and testing matrices, and the goal is to construct a predictive model that accurately estimates the target variable for new, unseen inputs.

## Problem Statement
Given a dataset \(D\), the objective is to construct a predictive model \(F_\theta(X) \rightarrow Y_{\text{pred}}\) that accurately estimates the target variable \(Y_i\) for new, unseen inputs \(X_i\).

### Dataset Description
- **Dtrain**: A matrix of dimension \(R(m \times n)\) representing the training data.
- **Dtest**: A matrix of dimension \(R(m1 \times n)\) representing the test data.
- **Ytrain**: Corresponding target variable with matrix dimension \(R(m \times 1)\).
- **Ytest**: Corresponding target variable with matrix dimension \(R(m1 \times 1)\).

## Features
- Hybrid model combining SVM and Gradient Boosting.
- Data preprocessing including feature scaling and label encoding.
- Prediction script to generate results based on new input data.

## Tech Stack
- **Programming Language**: Python
- <span style="background-color: #f0db4f; color: #323330; padding: 5px; border-radius: 3px;">Python</span>
<span style="background-color: #00d8ff; color: #323330; padding: 5px; border-radius: 3px;">scikit-learn</span>
<span style="background-color: #150458; color: #ffffff; padding: 5px; border-radius: 3px;">pandas</span>
<span style="background-color: #00b300; color: #ffffff; padding: 5px; border-radius: 3px;">joblib</span>
<span style="background-color: #fcbf1e; color: #323330; padding: 5px; border-radius: 3px;">numpy</span>
<span style="background-color: #007acc; color: #ffffff; padding: 5px; border-radius: 3px;">Visual Studio Code</span>
<span style="background-color: #f14e32; color: #ffffff; padding: 5px; border-radius: 3px;">Git</span>
- **Libraries**:
  - **scikit-learn**: For machine learning algorithms and data preprocessing.
  - **pandas**: For data manipulation and analysis.
  - **joblib**: For model serialization (saving and loading).
  - **numpy**: For numerical operations and array handling.
- **Development Environment**:
  - **Visual Studio Code**: For coding and debugging.
- **Version Control**: Git

## Installation
To run this project, ensure you have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name

2. Install the required packages:
   pip install -r requirements.txt

## Usage 
1. Prepare your input data in a CSV file named input_data.csv

2. Run the prediction script:
   python predict.py
   
3. View the predictions in the console output.

## Project Structure 
/project-root
├── predict.py          # Script for making predictions
├── input_data.csv      # Sample input data for predictions
├── requirements.txt     # Required Python packages
├── .gitignore          # Files and directories to ignore in git
└── README.md           # Project documentation

## License 
This project is licensed under the MIT License. See the LICENSE file for details.
