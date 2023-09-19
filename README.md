# Heart Failure Prediction
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
| Feature                 | Data Type |Description                          |
|----------------------------|-----------|----------------------------------|
| `age`                      | Numeric   | Age of the patient in years.                                        |
| `anaemia`                  | Binary    | Indicates whether the patient has anaemia (0 for no, 1 for yes).    |
| `creatinine_phosphokinase` | Numeric   | Level of creatinine phosphokinase enzyme in the blood (mcg/L).      |
| `diabetes`                 | Binary    | Indicates whether the patient has diabetes (0 for no, 1 for yes).   |
| `ejection_fraction`        | Numeric   | Percentage of blood leaving the heart at each contraction.          |
| `high_blood_pressure`      | Binary    | Indicates whether the patient has high blood pressure (0 for no, 1 for yes). |
| `platelets`                | Numeric   | Platelets count in the blood (kiloplatelets/mL).                    |
| `serum_creatinine`         | Numeric   | Level of serum creatinine in the blood (mg/dL).                     |
| `serum_sodium`             | Numeric   | Level of serum sodium in the blood (mEq/L).                         |
| `sex`                      | Binary    | Gender of the patient (0 for female, 1 for male).                   |
| `smoking`                  | Binary    | Indicates whether the patient is a smoker (0 for no, 1 for yes).    |
| `time`                     | Numeric   | Follow-up period in days.                                          |
| `DEATH_EVENT`              | Binary    | Indicates whether the patient experienced a death


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
