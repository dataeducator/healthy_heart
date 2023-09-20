# Heart Disease Classification
![national-cancer-institute-z8ofh6Zkn4k-unsplash](https://github.com/dataeducator/capstone/assets/107881738/2e13f05d-1344-41a1-9a9e-30175fa86572)

# Business Understanding
#### __Problem Statement:__
In the realm diagnostics, timely and accurate diseease detection is crucial to improving patient health outcomes. Zephyr Healthcare recognizes the potential of machine learning techniques in achieving this goal. The challenge lies in developing a robust predictive model that harnesses the power of neural networks or ensemble methods to identify cardiac conditions with a focus on achieving high recall rates.

#### __Stakeholder:__
Zephyr Healthcare Solutions

#### __Business Case:__  
As a newly appointed lead of the data analytics team at Zephyr Healthcare Solutions,my team has been tasked with enhancing the company's diagnostic capabilities through advanced predictive modeling techniques.

# Data Understanding
### __Data Description:__
The Heart Failure Prediction dataset is a collection of clinical and demographic features that was created by combining five heart datasets aimed at predicting the likelihood of heart failure.
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
 - We will be using Gogle Colab. Please upload the `kaggle.json` file to a folder called kaggle your Gogle Drive. This will allow you to access the Kaggle API from your Colab notebooks.

# Data Preparation
#### __Display basic statistics__

First, we will instantiate the `ExploreData` class and pass the cleaned dataset `cleaned_data` as a parameter to initialize the class. Next we will use the `examine_structure()` method to gather initial insights into the datasets structure, including basic statistics, data shape, datatypess and any missing values.
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
* RestingBP: difficult to quanitify with data (higher resting blood pressure >100 bpm slightly more likely to have heart disease)
* Cholesterol: difficult to quantity with data
* MaxHR: Patients with lower maximum heart rate are more likely to have heart disease ( values <100)
* Oldpeak: Patients with higher old peak values are more likely to have heart disease (values > 1)
  
# Modeling
# Evaluation
# Deployment
# Repository Structure
***
<pre>
   .
   └──notebook/
      ├── README.md                                            Overview for project reviewers  
      ├── notebook.ipynb                                       Documentation of Full Analysis in Jupyter Notebook
      ├── presentation.pdf                                     PDF version of Full Analysis shown in a slide deck                                   
      ├── setup.yml                                            Includes instructions to obtain the dataset
      └── .gitignore                                           Specifies intentionally untracked files
