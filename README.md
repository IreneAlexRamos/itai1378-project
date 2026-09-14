\# Gnasher Group



\*\*Team Members:\*\* Irene Ramos  

\*\*Tier Selection:\*\* Tier 1 — Tier 1 fits the scope because the project will use a pretrained computer vision model and focus on image classification without building a large custom model from scratch.



\## Problem Statement

Many people have difficulty identifying what type of dog they own or encounter, especially when breeds have similar physical features. This can make it harder for owners to understand expected behavior, care needs, and breed characteristics, and can also create confusion in veterinary and animal-care environments.



\## Solution Overview

Gnasher Group will create a computer vision application that analyzes an uploaded dog image and predicts which of the seven AKC breed groups the dog belongs to. The application will return the predicted group and basic information that helps the user better understand the dog's general characteristics.  

\*\*Flow:\*\* `Dog image` → `Pretrained image classification model` → `Image preprocessing and classification` → `Predicted AKC group and result display`



\## Technical Approach

\- \*\*CV Technique:\*\* Image Classification

\- \*\*Model:\*\* ResNet50

\- \*\*Framework:\*\* PyTorch / Torchvision  

\*Why:\* ResNet50 is a well-established pretrained image-classification model that can be fine-tuned using transfer learning. It is suitable for Google Colab and should provide good accuracy without requiring the group to train a large neural network from scratch.



\## Data Plan

\- \*\*Source:\*\* Public dog-image dataset, such as the Stanford Dogs Dataset, with breed labels mapped into the seven AKC groups

\- \*\*Approx. Size:\*\* Approximately 20,000 dog images available, with a smaller balanced subset used if needed for Colab limits

\- \*\*Labels:\*\* Sporting, Hound, Working, Terrier, Toy, Non-Sporting, and Herding



\## Success Metrics

\- \*\*Primary:\*\* Achieve at least 80% classification accuracy across the seven AKC groups

\- \*\*Secondary:\*\* Produce a prediction in under 3 seconds per uploaded image in Google Colab



\## Milestone Plan

| Phase | Goal | Target Timeline |

| :--- | :--- | :--- |

| 🧭 Blueprint | Proposal submitted | Week 5 (10-wk) / Week 10 (16-wk) |

| 🔌 First Working Demo | Pretrained model runs on sample dog images | Week 6 / Week 11 |

| 🛠 Make It Yours | Dataset mapped to seven AKC groups and classification logic integrated | Weeks 7-8 / Weeks 12-13 |

| 📈 Improve and Measure | Evaluate accuracy and improve model performance | Week 9 / Week 14 |

| 🎥 Package and Present | Demo video, final notebook, and documentation completed | Week 10 / Week 15 |



\## Risks and Plan B

1\. \*\*Risk:\*\* Some AKC groups may have more available images than others, which could cause class imbalance. → \*\*Plan B:\*\* Use a balanced subset, image augmentation, or class weighting during training.

2\. \*\*Risk:\*\* Google Colab GPU limits or long training times could interrupt model training. → \*\*Plan B:\*\* Reduce the dataset size, use fewer training epochs, freeze more pretrained model layers, or run inference using the pretrained model on a smaller sample.



\## Compute \& Estimated Cost

\- \*\*Environment:\*\* Google Colab (Free tier)

\- \*\*Estimated Cost:\*\* $0.00



\# Data Source \& Acquisition Plan



\- \*\*Primary Source:\*\* Stanford Dogs Dataset — a public dataset containing images of 120 dog breeds. The breed labels will be mapped into the seven AKC groups: Sporting, Hound, Working, Terrier, Toy, Non-Sporting, and Herding.



\- \*\*Dataset Link:\*\* http://vision.stanford.edu/aditya86/ImageNetDogs/



\- \*\*Directory Layout:\*\*

&#x20; ```text

&#x20; data/

&#x20; ├── raw/       # Original dog images from the dataset

&#x20; └── processed/ # Resized, organized, and labeled images for the seven AKC groups

