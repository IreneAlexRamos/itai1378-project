# Saved V2 run · v2_run04

Saved on October 13, 2026. Run configuration identifies source commit `5402e9f2a5d7ad6ab1eccc5fe373f399417aaf49`, seed 42, Torch 2.11.0+cu130 and Torchvision 0.26.0+cu130. The run configuration and test report reference the same manifest hash. The threshold and report reference the same checkpoint hash. The threshold is 0.50, selected on validation. The public inference notebook checks the checkpoint bytes before loading it.

- [Paired predictions](test_predictions.csv) and [test report](test_report.json) cover 1,050 rows. From the V2 predictions, 947 top-1 labels and 939 final decisions matched.
- [Train/validation history](history.csv), [curves](learning_curves.png), and [test decision matrix](confusion_with_uncertain.png) are saved run outputs. Last and best validation top-1 was 91.14%.
- [Run configuration](run_config.json), [manifest summary](manifest_metadata.json), [threshold](threshold.json), and [GPU timing](speed_report.json) document the run.

## Failure review

| Saved image label | Expected | V2 decision | Leading score |
|---|---|---|---:|
| n02100583_11710.jpg | Sporting | Hound | 59.5% |
| n02090622_5025.jpg | Hound | Working | 91.5% |
| n02107683_3817.jpg | Working | Herding | 61.7% |
| n02095889_4774.jpg | Terrier | Working | 74.3% |
| Cocker spaniel n02102318_11704.jpg | Sporting | Uncertain | 49.4% Non-Sporting |

The cocker spaniel shows the intended abstention behavior. The Borzoi is an accepted error despite its high Working score. The notebook does not establish whether pose, background, label mapping or another factor caused these mistakes. Watch for both failure modes during a live demo.

A separate saved notebook output shows Titus as Herding (80.37%) and Toy second (16.19%). Titus has no confirmed breed label and remains outside accuracy counts. This audit checked saved results without rerunning model inference.

The [public breed-label table](../../data/stanford_breed_to_group_review.csv) replaces the reviewer-name field with a role. Its bytes therefore differ from the manifest's original label hash. The original remains in Drive. Review status and exclusions are preserved.
