# Machine learning-based forecast of Helmet-CPAP therapy failure in Acute Respiratory Distress Syndrome patients

<br>

## Authors
Riccardo Campi<sup>a</sup>, Antonio De Santis<sup>a</sup>, Paolo Colombo<sup>b</sup>, Paolo Scarpazza<sup>b</sup>, Marco Masseroli<sup>a</sup>

<sup>a</sup>Department of Electronics Information and Bioengineering, Politecnico di Milano, Piazza L. Da Vinci 32, Milano, MI, 20133, Italy<br>
<sup>b</sup>Azienda Socio-Sanitaria Territoriale (ASST) della Brianza, Via Santi Cosma e Damiano 10, Vimercate, MB, 20871, Italy

Email addresses: riccardo.campi@mail.polimi.it (Riccardo Campi), antonio.desantis@polimi.it (Antonio De Santis), paolo.colombo@asst-brianza.it (Paolo Colombo), paolo.scarpazza@asst-brianza.it (Paolo Scarpazza), marco.masseroli@polimi.it (Marco Masseroli)

<br>

## Abstract
### Background and Objective:
Helmet-Continuous Positive Airway Pressure (H-CPAP) is a non-invasive respiratory
support that is used for the treatment of Acute Respiratory Distress Syndrome (ARDS), a severe medical condition
diagnosed when symptoms like profound hypoxemia, pulmonary opacities on radiography, or unexplained respiratory
failure are present. It can be classified as mild, moderate or severe. H-CPAP therapy is recommended as the initial
treatment approach for mild ARDS. Even though the efficacy of H-CPAP in managing patients with moderate-to-severe
hypoxemia remains unclear, its use has increased for these cases in response to the emergence of the COVID-19 Pandemic.

Using the electronic medical records (EMR) from the Pulmonology Department of Vimercate Hospital, in this study we
develop and evaluate a Machine Learning (ML) system able to predict the failure of H-CPAP therapy on ARDS patients.

### Methods:
The Vimercate Hospital EMR provides demographic information, blood tests, and vital parameters of
all hospitalizations of patients who are treated with H-CPAP and diagnosed with ARDS. This data is used to create
a dataset of 720 records and 38 features. Different ML models such as Random Forest, XGBoost, SVM, and Logistic
Regression with LASSO are iteratively trained in a cross-validation fashion. We also apply a classification threshold
calibration method and a feature selection algorithm to improve predictions quality and reduce the number of features.

### Results and Conclusions:
The Random Forest and XGBoost models proved to be the most effective. They
achieved final accuracies of 93.06% and 92.13% respectively. In terms of F1-score, the models scored 89.21% and 87.59%
respectively. These models are trained with 13 features for Random Forest and 16 features for XGBoost. The PaO2/FiO2
Ratio, C-Reactive Protein, and O2 Saturation resulted as the most important features, followed by Heartbeats, Respiratory
Rate, eGFR, Creatinine, and D-Dimer, in accordance with the medical scientific literature.

<br>

## Reproducibility

| File/folder        | Description                                                                    |
|:------------------ |:------------------------------------------------------------------------------ |
| /main.ipynb        | code used to clean the datasets and train the models                           |
| /functions.py      | accessory functions and classes                                                |
| /models_vimercate/ | folder containing the trained models                                           |
| /datasets/         | folder containing the dataset "dataset_vimercate.csv" used to train the models |
| /images/           | folder containing the obtained images                                          |
