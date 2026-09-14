# Gnasher Group

**Team Members:** Irene Ramos

**Tier Selection:** Tier 1 — Tier 1 fits the scope because the project uses one pretrained computer vision model for a single image-classification task.

## Problem Statement

Many people have difficulty identifying their dog's breed group, especially when breeds have similar physical features, such as French Bulldogs and Pugs. This can make it harder for pet owners, adopters, veterinary professionals, and shelter professionals to understand a dog's general characteristics and needs.

## Solution Overview

Gnasher Group will create an AI-powered computer vision application that analyzes a single uploaded dog photograph and predicts which of the seven AKC breed groups the dog belongs to. The application will return the predicted AKC breed group along with a confidence score.

**Flow:** `Dog photograph` → `Image preparation` → `EfficientNet-B0 model` → `AKC breed group + confidence score`

## Technical Approach

- **CV Technique:** Image Classification
- **Model:** EfficientNet-B0
- **Framework:** PyTorch / Torchvision
- **Environment:** Google Colab

**Why:** EfficientNet-B0 is an efficient, lightweight image-classification model that supports transfer learning using pretrained ImageNet weights. Its relatively small size makes it appropriate for this Tier 1 project and practical for training and testing with free Google Colab GPU resources.

## Data Plan

- **Source:** Stanford Dogs Dataset
- **Full Dataset Size:** 20,580 dog images from 120 breeds
- **Planned Subset:** Approximately 7,000 balanced images, with a maximum of about 1,000 images per AKC group
- **Labels:** Sporting, Hound, Working, Terrier, Toy, Non-Sporting, and Herding
- **Data Split:** 70% training, 15% validation, and 15% testing
- **Dataset Link:** http://vision.stanford.edu/aditya86/ImageNetDogs/

The original breed labels will be mapped into the seven AKC breed groups for classification.

## Success Metrics

- **Primary:** Achieve at least 85% classification accuracy on unseen test images.
- **Secondary:** Maintain an average prediction time of under 1 second per image using a Google Colab GPU.
- **Evaluation:** Use a confusion matrix and review 3–5 incorrect predictions to identify common failure cases.

## Milestone Plan

| Phase | Goal | Target Timeline |
| :--- | :--- | :--- |
| 🧭 Blueprint | Proposal submission and repository creation | Week 10 |
| 🔌 First Working Demo | Run pretrained model on sample dog photographs | Week 11 |
| 🛠 Make It Yours | Prepare labels, train model, and add prediction logic | Weeks 12–13 |
| 📈 Improve and Measure | Test accuracy, prediction speed, and failure cases | Week 14 |
| 🎥 Package and Present | Complete final demo, README, video, and presentation slides | Week 15 |

## Risks and Plan B

1. **Risk:** Similar-looking breeds may confuse the model.  
   **Plan B:** Show the top two predictions and provide an alert when model confidence is low.

2. **Risk:** Class imbalance may negatively affect classification accuracy.  
   **Plan B:** Use dataset balancing, class weights, and image augmentation to improve model performance.

## Compute & Estimated Cost

- **Environment:** Google Colab GPU
- **Framework:** PyTorch / Torchvision
- **Dataset:** Stanford Dogs Dataset
- **Project Tools:** GitHub and Google Drive
- **Estimated Cost:** $0

## Data Source & Acquisition Plan

The project will use the **Stanford Dogs Dataset**, which contains 20,580 images representing 120 dog breeds. The original breed labels will be mapped into the seven AKC breed groups: Sporting, Hound, Working, Terrier, Toy, Non-Sporting, and Herding.

A balanced subset of approximately 7,000 images will be used, with a maximum of about 1,000 images per group. The dataset will be divided into 70% training, 15% validation, and 15% testing data.

**Dataset Link:** http://vision.stanford.edu/aditya86/ImageNetDogs/

### Directory Layout

```text
data/
├── raw/       # Original dog images from the dataset
└── processed/ # Images organized and labeled into the seven AKC groups
```

**License / Compliance Note:** The Stanford Dogs Dataset will be used for educational and academic purposes. The project will follow the dataset's stated usage terms and properly acknowledge the original dataset source.