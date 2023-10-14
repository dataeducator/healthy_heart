# Healthy Hearts: Predicting Heart Disease with Random Forests
![healthy_heart](https://github.com/dataeducator/healthy_heart/assets/107881738/d2c08d7a-1947-4426-ba81-cd2abc7911dd)

# Business Understanding
#### __Problem Statement:__
In diagnostics, timely and accurate disease detection is crucial to improving patient health outcomes. Zephyr Healthcare recognizes the potential of machine learning techniques in achieving this goal. The challenge lies in developing a robust predictive model that harnesses the power of machine learning algorithms, including random forests, to predict heart disease, focusing on achieving high recall rates.

#### __Stakeholder:__
Zephyr Healthcare Solutions

#### __Business Case:__  
As a newly appointed head of the data analytics team at Zephyr Healthcare Solutions, my team has been tasked with enhancing the company's diagnostic capabilities through advanced predictive modeling techniques for Heart Disease.

# Data Understanding
### __Data Description:__
The Heart Failure Prediction dataset is a collection of clinical and demographic features created by combining five heart datasets to predict the likelihood of heart failure.
#### Features
| Feature           | Description                                                         |
|-------------------|---------------------------------------------------------------------|
| `Age`             | Age of the patient [years]                                          |
| `Sex`             | Sex of the patient [M: Male, F: Female]                              |
| `ChestPainType`   | Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic] |
| `RestingBP`       | Resting blood pressure [mm Hg]                                       |
| `Cholesterol`     | Serum cholesterol [mm/dl]                                           |
| `FastingBS`       | Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]      |
| `RestingECG`      | Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality, LVH: showing probable or definite left ventricular hypertrophy] |
| `MaxHR`           | Maximum heart rate achieved [Numeric value between 60 and 202]        |
| `ExerciseAngina`  | Exercise-induced angina [Y: Yes, N: No]                                |
| `Oldpeak`         | Oldpeak = ST [Numeric value measured in depression]                   |
| `ST_Slope`        | The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping] |
| `HeartDisease`    | Output class [1: heart disease, 0: Normal]                             |

### Data Exploration
#### __Obtaining  Dataset for Prediction with Machine  Learning__
1. __Create or Log in to Your Kaggle Account:__
    If you do not already have a Kaggle account, create one. If you have an account log in.
2. __Access Your Account Settings:__
  - Click on your profile picture in the top right corner of the Kaggle website.
  - Select __`Account`__ from the dropdown menu.
    
3. __Navigate to the API Section:__
  - Scroll down to the __`API`__ section on the account page.

4. __Create New API Token:__
  - Click on the __`Create New API Token`__ button. This will trigger the download of a file named `kaggle.json`.
5. __Move API Token to Google Drive(We will be using Google Colab)__
 - We will be using Gogle Colab. Please upload the `kaggle.json` file to a folder called Kaggle in your Google Drive. This will allow you to access the Kaggle API from your Colab notebooks.

# Data Preparation
#### __Display basic statistics__

First, we will instantiate the `ExploreData` class and pass the cleaned dataset `cleaned_data` as a parameter to initialize the class. Next, we will use the `examine_structure()` method to gather initial insights into the dataset's structure, including basic statistics, data shape, datatypes, and missing values.
![DataPreparation 1](https://github.com/dataeducator/capstone/assets/107881738/3f49a78a-326e-459b-9967-7ad42aabb514)

![DataPreparation2](https://github.com/dataeducator/capstone/assets/107881738/bd11ce1a-75ff-4337-ad6a-b2bc304f10f4)

Then we look at correlations
![DataPreparation3](https://github.com/dataeducator/capstone/assets/107881738/81392697-1502-498f-9a30-1dfb0ed0f0e2)

Then we create visualizations to look at patterns in the data.
![DataPreparation4](https://github.com/dataeducator/capstone/assets/107881738/7ef8de9c-c292-46b7-87d8-ad909fcd26aa)
![DataPreparation5](https://github.com/dataeducator/capstone/assets/107881738/9e9cab3f-1ada-4eba-9061-33dc0aa75e2c)
#### __Preliminary Insights:__
* Sex: __Male patients__ are more likely (approximately 61%) to have heart disease than female patients (approximately 26%).
* ChestPainType: Patients with __ASY type of chest pain__ are more likely to have heart disease than patients with other types of chest pains. (ASY: Asymptomatic Chest Pain, sometimes described as silent due to a lack of intensity compared to a heart attack)
* RestingECG- Patients with Resting electrocardiograms that show __ST-T wave abnormality (labeled ST)__ are more likely to have heart disease
* ExerciseAngina: Patients with this feature __(ExerciseAngina) labeled `yes`__ are more likely to have heart disease
* ST_slope: Patients with this feature __(ST_slope) labeled `Flat` or `Down`__ are more likely than patients with this feature labeled `Up`.
* FastingBS: Patients with __fasting blood sugar > 120 mg/dL__ are more likely to have heart disease.
* Age: Patients __over 50__ are more likely to have heart disease
* RestingBP: difficult to quantify with data (higher resting blood pressure >100 bpm slightly more likely to have heart disease)
* Cholesterol: difficult to quantity with data
* MaxHR: Patients with lower maximum heart rate are more likely to have heart disease ( values <100)
* Oldpeak: Patients with higher old peak values are more likely to have heart disease (values > 1)
  
# Modeling

This code section focuses on leveraging the binary_classifier object to analyze heart disease data. The binary_classifier is a crucial component that facilitates the classification task. It encapsulates a set of models, including Logistic Regression, Decision Trees, Random Forest, and XGBoost, each with its distinct approach to predicting heart disease. These models collectively form a versatile toolkit for classification tasks.

The process begins with data preprocessing, where the DataFrame 'df' containing the heart disease dataset is prepared for analysis. This involves scaling the features and organizing the target variable, resulting in X_scaled and y_binary. Subsequently, the data is partitioned into training and testing sets using the train_test_split method. This ensures that the models are evaluated on independent data to assess their generalization capabilities.

Next, the models are trained using the preprocessed data. For instance, the code demonstrates training a Logistic Regression model. The binary_classifier object provides a convenient interface for this task. Following training, key details about the model are printed using the print_model_information method. This step offers valuable insights into the chosen model's characteristics and parameters, aiding in the interpretation of results.

Our baseline Results were as follows:


![Modeling1](https://github.com/dataeducator/healthy_heart/assets/107881738/06cfdead-7aea-490e-89f8-554da2008f60)


Our final model had the following results based on recall: 
| Model                 | Recall   |
|-----------------------|----------|
| Logistic Regression   | 0.862    |
| K-Nearest Neighbors   | 0.868    |
| Support Vector Machine | 0.865    |
| Decision Trees        | 0.811    |
| Random Forest         | 0.871    |
| XGBoost               | 0.857    |
| Neural Network        | 0.869    |


# Evaluation

Given the feature importance provided for each model, we can infer which features are deemed important across different algorithms for classifying a patient as having heart disease. Considering that the Random Forest model is the most performative after parameter tuning, let's highlight the features that appear to be consistently influential across different models:

# Deployment
1. __Web Application Development:__ Work in progress click [here](https://healthy-heart.streamlit.app/)
  - Build a user-friendly web application that allows users to interact with your heart disease classification model. Consider using frameworks like Flask, Streamlit, or FastAPI for efficient development.

2. Github Repo: minimum viable product
# Recommendations
1. __Use the Random Forest Model__: Given its performance across metrics, including a recall > 0.85, consider prioritizing the Random Forest model for heart disease classification.

2. __Feature Importance Insights__: Focus on `ExerciseAngina`, `ChestPainType`, and `RestingBP` as they consistently appear as top model influencers across various models.

3. __SHAP force plot Interpretation__: Leverage the SHAP for the plot generated for the random forest model. It provides valuable insights into how individual features impact the classification of a chosen instance. 

4. __Consider Logistic Regression and Neural Network__: While the Random Forest Model is the most performative, Logistic Regression is known for being easier to understand, and considering we will be using this in a clinical setting, we should consider using models like Neural Networks that have been readily adopted in the medical community for similar tasks.

# Future Work
1. __Advanced Feature Engineering:__

  - Explore more sophisticated feature engineering techniques like interaction terms, polynomial features, or dimensionality reduction methods like PCA. These can help extract more meaningful information from the data.

2. Parallel Computing with XGBoost:

  - Leverage parallel processing capabilities offered by XGBoost. Utilize multi-threading or distributed computing techniques to expedite model training, especially when dealing with large datasets.

3. __User Feedback and Iterative Improvement:__
    - Gather feedback from web application users to identify improvement areas. Use this feedback to refine the model and the application interface iteratively.

4. __Optimization for GPU Usage:__
  - Fine-tune models to make optimal use of the T4 GPU. Experiment with different settings and configurations to maximize GPU performance.


Please review my full analysis in [my notebook](https://github.com/dataeducator/healthy_heart/blob/main/notebook.ipynb ) or ([my presentation](https://github.com/dataeducator/healthy_heart/blob/main/presentation.pdf )).
Feel free to contact me __Tenicka Norwood__ at tenicka.norwood@gmail.com if you have more questions.

# Repository Structure

***
<pre>
   .
   └──notebook/
      ├── README.md                                            Overview for project reviewers  
      ├── notebook.ipynb                                       Documentation of Full Analysis in Jupyter Notebook
      ├── presentation.pdf                                     PDF version of Full Analysis shown in a slide deck                                   
      ├── requirements/                                        Includes requirements to deploy the Streamlit app and instructions to obtain dataset
      ├── models/                                              Includes models of scaled X_train and random forest for web app deployment
      ├── images/                                              Includes a folder of images for the project
      ├── scripts/                                             Includes Python code for the Streamlit web app
      └── .gitignore                                           Specifies intentionally untracked files
