# Heart Disease Prediction

## Overview
This project is a machine learning-based heart disease prediction system. It uses a dataset with various medical attributes to predict the likelihood of heart disease in patients. By employing machine learning models, the system aims to assist healthcare professionals in making informed decisions about a patient's heart health.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models and Approach](#models-and-approach)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Project Structure
The project contains the following major components:
- `PDS_practicals.ipynb`: The Jupyter notebook containing code for data preprocessing, model training, and evaluation.
- `dataset/`: The folder containing the heart disease dataset used for training and evaluation (unzip the `abc173.zip` file to access the data).
- `README.md`: The file you're currently reading.
- `requirements.txt`: A list of Python dependencies required to run the project.

## Dataset
The dataset used for this project contains medical attributes relevant to heart disease diagnosis. Some key features include:
- Age
- Sex
- Resting blood pressure
- Cholesterol levels
- Maximum heart rate achieved
- Presence of angina
- ST depression induced by exercise

The dataset is included in the repository under the `dataset/` folder. Unzip the `abc173.zip` file to access it. You can also download it directly from [Heart Disease Dataset](path/to/dataset).

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/VaradLad2/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate # On Windows: `env\Scripts\activate`
   ```

## Usage

1. Open the Jupyter notebook to run the code:
   ```bash
   jupyter notebook PDS_practicals.ipynb
   ```

2. Follow the steps in the notebook to preprocess the dataset, train the models, and evaluate the results.

3. Adjust the parameters in the code (e.g., test size, model hyperparameters) to experiment with the models.

## Models and Approach

This project uses several machine learning models to predict heart disease, including:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

### Preprocessing:
- Handling missing values
- Normalizing data
- Splitting the data into training and test sets

### Model Training and Evaluation:
- Cross-validation for performance evaluation
- Accuracy, precision, recall, and F1-score metrics used for model assessment

## Results
The best performing model achieved an accuracy of **X%** (replace with your actual result). Detailed performance metrics and confusion matrices for each model can be found in the notebook.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open a pull request or issue.
