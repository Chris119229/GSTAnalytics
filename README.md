# GST Hackathon Hybrid Model

## Description
This project implements a hybrid machine learning model using Support Vector Machine (SVM) and Gradient Boosting for predicting GST outcomes based on a provided dataset. The dataset consists of training and testing matrices, and the goal is to construct a predictive model that accurately estimates the target variable for new, unseen inputs.

## Problem Statement
Given a dataset \(D\), the objective is to construct a predictive model \(F_\theta(X) \rightarrow Y_{\text{pred}}\) that accurately estimates the target variable \(Y_i\) for new, unseen inputs \(X_i\).

### Dataset Description
- **Xtrain**: A matrix of dimension \(R(m \times n)\) representing the training data.
- **Xtest**: A matrix of dimension \(R(m1 \times n)\) representing the test data.
- **Ytrain**: Corresponding target variable with matrix dimension \(R(m \times 1)\).
- **Ytest**: Corresponding target variable with matrix dimension \(R(m1 \times 1)\).

## Features
- Hybrid model combining SVM and Gradient Boosting.
- Data preprocessing including feature scaling and label encoding.
- Prediction script to generate results based on new input data.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - **scikit-learn**: For machine learning algorithms and data preprocessing.
  - **pandas**: For data manipulation and analysis.
  - **joblib**: For model serialization (saving and loading).
  - **numpy**: For numerical operations and array handling.
- **stacks**:
  - ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-00B300?style=flat&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/VS%20Code-007ACC?style=flat&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
- **Development Environment**:
  - **Visual Studio Code**: For coding and debugging.
- **Version Control**: Git

## Steps to Execute
To run this project, ensure you have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GSTAnalytics.git
   cd repository-name

2. Create a virtual environment
    ```bash
    python -m venv venv

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate

   - On macOS/Linus:
     ```bash
     source venv/bin/activate

4. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage 
1. Prepare your input data in a CSV file named input_data.csv

2. Run the prediction script:
   ```bash
   python predict.py
   
3. View the predictions in the console output.


## Project Structure
- **/project-root**
  - ├── model.py            # Script for training the model
  - ├── predict.py          # Script for making predictions
  - ├── input_data.csv      # Sample input data for predictions
  - ├── requirements.txt     # Required Python packages
  - ├── .gitignore          # Files and directories to ignore in git
  - └── README.md           # Project documentation

## Checksum Value 
  ```bash
- F:\>python checksum.py "GSTAnalytics.zip" --algorithm sha256
SHA256 Checksum: a436d8f6e7cb0947e0918aff7b0906b0f4a6cfe4e9f83252af4cefaf9631814c
