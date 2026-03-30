# ShadowReliability

ShadowReliability is a reliability aware text classification framework that combines a main classifier with a shadow model designed to predict when the main classifier may be wrong.

This work is centered on a healthcare administrative request routing case study. Instead of only returning a predicted label, the system also estimates prediction risk and supports safer decision making through selective review.


## Project Goal

Most text classification systems focus on accuracy alone. In practice, however, an accurate classifier can still make high confidence mistakes on ambiguous, difficult, or shifted examples.

The goal of ShadowReliability is to address that gap by building a second layer model that predicts failure risk from the behavior of the primary classifier. This allows the system to move beyond simple classification and toward reliability aware decision support.

## Live Demo

Hugging Face demo: [ShadowReliability Demo](https://huggingface.co/spaces/thejosht/shadowreliability-demo)

## Problem Setting

This project uses a multi class healthcare administrative request classification task with the following labels:

- Appointment Scheduling
- Billing and Insurance
- Prescription Refill
- Referral Request
- Medical Records and Forms
- Portal and Account Access

The objective is to classify each request correctly while also identifying which predictions appear safe and which may require manual review.


## Method Overview

ShadowReliability uses a two layer design:

### Main Classifier
The primary model is a **TF-IDF with Logistic Regression** classifier trained to predict the administrative request label.

### Auxiliary Models
To enrich reliability signals, two additional models are used:

- Support Vector Machine (SVM)
- Naive Bayes (NB)

These models are not the final production facing classifier. Instead, they provide additional uncertainty and disagreement information used by the shadow layer.

### Shadow Model
A separate shadow model is trained to estimate whether the main classifier is likely to fail on a given input.

The shadow model uses features derived from model behavior, including:

- maximum confidence
- probability margin
- prediction entropy
- disagreement across models
- number of unique predicted labels
- majority vote size
- text length features

This creates a reliability layer that predicts risk rather than class.

## Core Idea

Instead of asking only:

**“What class does this request belong to?”**

the system also asks:

**“How likely is it that this prediction is wrong?”**

That second question is the core of the project.

This allows the system to support:

- prediction risk scoring
- selective prediction
- manual review recommendations
- more trustworthy use of classifier outputs

## Results Summary

Key project results include:

- Clean In Domain Accuracy: 99.4%
- Test Challenge Accuracy: 86.1%
- Selective Accuracy at ~80% Coverage: 91.4%
- Shadow Model ROC-AUC: 0.844
- Shadow Model AP: 0.502

These results suggest that the shadow model captures meaningful signal about prediction risk and can help distinguish safer predictions from riskier ones.

## App Demo

The Streamlit app allows a user to:

- enter a healthcare request
- view the predicted class
- view the classifier confidence
- view shadow model risk
- inspect class probability breakdown
- inspect feature importance and project summary metrics

## Project Structure

```text
ShadowReliability/
│
├── app/
│   └── app.py
├── artifacts/
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

## How to run (in terminal)

1. Install dependencies

pip install -r requirements.txt

2. Run the app

streamlit run app/app.py


## Key Contribution

The main contribution of this project is the addition of a shadow reliability layer on top of a standard text classifier.

Rather than treating all predictions equally, the system estimates which predictions are more likely to fail. This makes the project more aligned with real world deployment concerns, where uncertainty and failure awareness matter in addition to raw accuracy.


## Limitations

Current limitations include:
	•	the shadow model depends on the quality of the base model derived features
	•	evaluation is tied to the current dataset and class structure
	•	results may not fully generalize to other domains without retraining
	•	the current interface is still a prototype and can be extended further


## Future Improvements

Possible next steps:
	•	stronger out of domain detection
	•	richer calibration analysis
	•	improved interpretability views
	•	multi dataset evaluation


## Research Context

This work is related to research on selective classification, model calibration, and prediction trustworthiness.

Selective classification studies how a model can keep safer predictions while rejecting or deferring riskier ones. That idea is relevant here because ShadowReliability is designed to support review decisions rather than treating every prediction as equally safe.

Calibration research studies whether model confidence scores are actually trustworthy. This is important for this project because a model’s top probability alone does not always mean the prediction is truly reliable.

The project is also connected to work on classifier trust and failure prediction, where a second mechanism is used to estimate whether a model’s output should be trusted. ShadowReliability applies those ideas in a practical text classification setting by using a shadow model to estimate prediction risk.

### References

- Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*.
- Jiang, H., Kim, B., Guan, M., & Gupta, M. (2018). *To Trust Or Not To Trust A Classifier*.
- Luo, Y., Liu, J., et al. (2021). *Learning to Predict Trustworthiness with Steep Slope Loss*.