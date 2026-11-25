# Gnasher Group

Irene Ramos · HCC ITAI 1378 · Tier 1

Gnasher Group predicts an AKC group or returns Uncertain from a single dog photo. V2 trains a seven-output EfficientNet-B0 in PyTorch and Torchvision. The project preserves its original [Midterm Blueprint](docs/proposal.pdf) and [V1 prototype](prototypes/v1/README.md).

**Flow** · Photo → RGB and EfficientNet transforms → seven scores → leading group or Uncertain → top two candidates.

## Run the final demo

Open the [inference-only notebook](notebooks/Gnasher_Group_Inference.ipynb) in [Colab](https://colab.research.google.com/github/IreneAlexRamos/itai1378-project/blob/main/notebooks/Gnasher_Group_Inference.ipynb). Upload the original v2_run04 `best.pt` or point the notebook to your Drive copy. The threshold ships in the repository and the notebook checks the checkpoint SHA-256 before loading. Then upload one dog photo. The checkpoint is currently private. A fresh CPU check with the verified checkpoint reproduced the two benchmark examples in the recording guide. Course reviewers can inspect the saved [training notebook](notebooks/Gnasher_Group_Training.ipynb) and use the [V1 backup](prototypes/v1/README.md). See the [recording guide](docs/demo-recording.md).

**Narrated 3–5 minute demo** · Pending recording and signed-out access check. Add the live URL here and to the final slides.

## Data and engineering

Stanford Dogs supplied 20,580 photos across 120 breeds. The run selected 7,000 images from the AKC group mapping after exact-byte duplicate handling. A custom group-stratified 70/15/15 split used 4,900 training, 1,050 validation and 1,050 test examples. V2 trained the head for 3 epochs and fine-tuned the last 3 feature blocks for 5. Validation selected the checkpoint and a 0.50 abstention threshold. See [dataset details](data/README.md), [run provenance](results/v2_run04/README.md), and the [AI usage log](docs/AI_usage_log.md).

## Blueprint targets and measured results

| Target | Saved result |
|---|---|
| At least 85% accuracy on unseen test photos | V2 reached 90.19% top-1 and 89.43% final-decision accuracy on 1,050 photos held out from this project's fine-tuning. ImageNet pretraining overlap remains possible. |
| Average prediction in under one second on Colab GPU | Average model forward was 8.12 ms on Tesla T4, 30 measured passes after five warmups. Full upload-to-decision timing was not measured. |
| Confusion matrix and review of 3–5 mistakes | [Decision matrix](results/v2_run04/confusion_with_uncertain.png) and [failure review](results/v2_run04/README.md#failure-review). |

| Paired test measure | V1 | V2 |
|---|---:|---:|
| Top-1 accuracy | 97.14% | 90.19% |
| Final accuracy including abstentions | 94.38% | 89.43% |
| Coverage | 96.00% | 97.81% |
| Abstentions | 42 | 23 |

V2 answered more often, with lower accuracy on the shared benchmark. The comparison does not isolate why. The pretrained breed vocabulary aligns with this dataset, and potential pretraining overlap and near-duplicates limit interpretation. Image-level splitting can place photos of the same breed in multiple splits. The seven-output model has no non-dog class and its scores can be high on a wrong input. V1's original hand-picked 6/7 dog demo and CPU benchmark are separate from this paired test.

## What changed

V1 used unmodified ImageNet scores mapped from breeds to groups as a working baseline. V2 returned to the Blueprint’s Stanford Dogs training plan and made threshold selection and test evaluation explicit. The final comparison preserves both runs. [Earlier project notes](https://github.com/IreneAlexRamos/itai1378-project/blob/2e5492c/README.md) describe the original progression.

## Presentation

- [Final 12-slide PDF](docs/presentation.pdf)
- [Editable PowerPoint](docs/FP_Ramos_Irene_ITAI1378.pptx)
- [Ten-minute speaking script](docs/presentation-script.md)
