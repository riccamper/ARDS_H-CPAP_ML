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
Helmet-Continuous Positive Airway Pressure (H-CPAP) is a non-invasive respiratory support that is used for the treatment of Acute Respiratory Distress Syndrome (ARDS), a severe medical condition diagnosed when symptoms like profound hypoxemia, pulmonary opacities on radiography, or unexplained respiratory failure are present.
It can be classified as mild, moderate or severe.
H-CPAP therapy is recommended as the initial treatment approach for mild ARDS.
Even though the efficacy of H-CPAP in managing patients with moderate-to-severe hypoxemia remains unclear, its use has increased for these cases in response to the emergence of the COVID-19 Pandemic.

Using the electronic medical records (EMR) from the Pulmonology Department of Vimercate Hospital, in this study we develop and evaluate a Machine Learning (ML) system able to predict the failure of H-CPAP therapy on ARDS patients.

### Methods:
The Vimercate Hospital EMR provides demographic information, blood tests, and vital parameters of all hospitalizations of patients who are treated with H-CPAP and diagnosed with ARDS.
This data is used to create a dataset of 622 records and 38 features, with 70-30% split between training and test set.
Different ML models such as Support Vector Machines (SVM), XGBoost, Neural Network, Random Forest, and Logistic Regression are iteratively trained in a cross-validation fashion.
We also apply a feature selection algorithm to improve predictions quality and reduce the number of features.

### Results and Conclusions:
The SVM and Fully Connected Neural Network models proved to be the most effective, achieving final accuracies of 95.19% and 94.65%, respectively. In terms of F1-score, the models scored 88.61% and 87.18%, respectively. Additionally, the SVM and XGBoost models performed well with a reduced number of features (23 and 13, respectively).
The PaO2/FiO2 Ratio, C-Reactive Protein, and O2 Saturation resulted as the most important features, followed by Heartbeats, White Blood Cells, Arterial Sodium, Ionized Calcium, Arterial Potassium, and D-Dimer, in accordance with the clinical scientific literature.

<br>

## Reproducibility

| File/folder        | Description                                                                    |
|:------------------ |:------------------------------------------------------------------------------ |
| /main.ipynb        | code used to clean the datasets and train the models                           |
| /functions.py      | accessory functions and classes                                                |
| /models_vimercate/ | folder containing the trained models                                           |
| /datasets/         | folder containing the dataset "dataset_vimercate.csv" used to train the models |
| /images/           | folder containing the obtained images                                          |

<br>

## Institutional Review Board Statement
Human participants were involved in this research; the study was conducted in accordance with the Declaration of Helsinki. Our study was approved by the local institution, Vimercate Hospital, ASST-Brianza, according to the legal requirements concerning observational studies (Resolutions 0000573 27/07/2021 and 0000133 22/02/2023).

## Informed Consent Statement
Due to the nature of the present observational study and data anonymization, the patients' consent to participate was not required, as declared by the ASST Brianza Ethics Committee.
