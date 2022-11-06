# Fetal Health Classification Project

## Introduction

This is a mid-term project for the 2022 ml-zoomcamp held by Alexey Grigorev. 

The goal of this project is to classify fetal health based on records extracted from Cardiotocograms (CTGs), i.e. instruments measuring _e.g._ fetal heart rate, fetal movements, _etc._, via ultrasound pulses. 
Fetal health has been classified by expert obstetritians into 3 classes:
- Normal (Class 1)
- Suspect (Class 2)
- Pathological (Class 3)

The goal of this project is to build a multiclass classification model allowing to categorize each fetus' health into one of the above three classes. This is achieved using a Random Forest classifier.

## Dataset

Data were retrieved [here](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) from Kaggle, and are based on:

Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318

## File description

- `notebook.ipynb` contains code performing data preparation and cleaning, EDA, feature importance analysis, parameter tuning and model selection. It also contains code for training the final model and saving it with BentoML.
- `train.py`, as requested in the assignment, contains code for training the final model and saving it with BentoML.
- `service.py` contains code for loading the model and serving it via BentoML.
- `bentofile.yaml` contains a list of dependencies.
