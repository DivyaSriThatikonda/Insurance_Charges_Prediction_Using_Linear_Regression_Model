# Insurance_Charges_Prediction_Using_Linear_Regression_Model

**Insurance Cost Prediction Project**

**Project Overview**

The Insurance Cost Prediction Project aims to develop a predictive model to estimate the insurance charges for individuals based on a variety of demographic, health, and lifestyle factors. By leveraging a comprehensive dataset, the project seeks to uncover the key drivers of insurance costs and provide actionable insights to optimize premium setting and risk assessment.

**Dataset Description**

The dataset used in this project contains information about individuals' demographics, health metrics, and insurance charges. The key features in the dataset include:

age: The age of the individual.

sex: The gender of the individual.

bmi: Body Mass Index, a measure of body fat.

children: Number of children covered by the insurance.

smoker: Smoking status of the individual.

Claim_Amount: The total amount claimed by the individual.

past_consultations: Number of past medical consultations.

num_of_steps: Average number of steps taken daily.

Hospital_expenditure: Total hospital expenditure.

NUmber_of_past_hospitalizations: Number of past hospitalizations.

Anual_Salary: Annual income of the individual.

region: Geographic region where the individual resides.

charges: The target variable representing the insurance charges.

**Objective**

The primary objective of this project is to predict the insurance charges for individuals using a linear regression model. The goal is to understand how different factors such as age, BMI, smoking status, and past health history influence the insurance costs.

## Installation

To run the project in your Jupyter Notebook or Google Colab, follow these steps:

1. **Clone the repository**:
   - If using Jupyter Notebook:
     ```bash
     !git clone https://github.com/DivyaSriThatikonda/Insurance_Charges_Prediction_Using_Linear_Regression_Model.git
     ```
   - If using Google Colab:
     - Open a new Colab notebook.
     - Execute the following code cell:
       ```python
       !git clone https://github.com/DivyaSriThatikonda/Insurance_Charges_Prediction_Using_Linear_Regression_Model.git
       ```
2. **Install dependencies**:
   - Run the following code cell in the notebook to install the required libraries:
     ```python
     !pip install numpy pandas matplotlib seaborn scikit-learn
     ```

3. **Access the dataset**:
   - The `insurance.csv` dataset is already included in the repository. 

4.**Tools used**:
- **Numpy**: Used for numerical computations and array manipulation.
- **Pandas**: Used for data manipulation, analysis, and cleaning.
- **Seaborn**: Used for data visualization and statistical plotting.
- **Scikit-learn**: Used for building and evaluating the linear regression model.
-**Descriptive Statistics**: Used to summarize and analyze the dataset, including measures such as mean, median, standard deviation, and correlation coefficients.

**Results and Performance**

The machine learning model, employing linear regression, was utilized to predict insurance charges for individuals. The model's performance was evaluated using the R2 score, which measures the proportion of the variance in the target variable that is predictable from the independent variables. The obtained R2 score of 84% signifies that the model explains 84% of the variability in insurance charges based on the features included in the analysis. This demonstrates the model's capability to accurately estimate insurance costs for individuals, providing valuable insights for insurance pricing and risk assessment.







