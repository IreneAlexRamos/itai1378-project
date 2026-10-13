# Gnasher Group

**Student:** Irene Ramos  
**Course:** ITAI 1378  
**Tier:** 1 — pretrained image-classification prototype

## Problem and implemented solution

Gnasher Group uses a dog photograph to suggest one of seven AKC groups:
Sporting, Hound, Working, Terrier, Toy, Non-Sporting, and Herding.

The notebook uses pretrained ImageNet EfficientNet-B0. A mapping module
sums supported breed scores into AKC group scores while preserving
unsupported-dog and non-dog scores separately.

If the highest group score is below 50%, the final decision is
“Uncertain” and displays the top two group candidates for inspection.
These candidates are not accepted classifications. Group scores are not calibrated guarantees of correctness.
The system does not reliably detect cats or identify mixed-breed ancestry.

**Flow:** photograph → preprocessing → EfficientNet-B0 →
breed-to-group aggregation → suggested group or Uncertain

## Blueprint — original proposal

This section preserves the original plan in [docs/proposal.pdf](docs/proposal.pdf).
It describes proposed work, not completed training or evaluation.

- **Tier and reason:** Tier 1 (Core Project): one lightweight EfficientNet-B0
  model for one image-classification task, using PyTorch/Torchvision and free
  Google Colab resources.
- **Data:** [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  contains 20,580 images across 120 breeds. The plan was to build a
  breed-to-AKC-group table, target approximately 7,000 images with a maximum
  of 1,000 per group, and use a 70/15/15 train/validation/test split.
  Approximately 7,000 is a target, not a guaranteed count after exclusions.
- **Success targets:** at least 85% accuracy on unseen test images; average
  prediction time under one second per image on a Colab GPU; a confusion
  matrix and review of 3–5 incorrect predictions.
- **Risks and responses:** similar-looking breeds → show the top two group
  candidates when confidence is low; class imbalance → capped sampling,
  with class weights and augmentation planned for training. Ambiguous
  breed names need explicit labeling decisions rather than guesses.
- **Resources and budget:** Google Colab GPU, PyTorch, Stanford Dogs,
  GitHub, and Google Drive; estimated cost $0.

| Course week | Original milestone |
|---|---|
| 10 | Submit proposal and create repository |
| 11 | Run pretrained model on sample photographs |
| 12–13 | Prepare labels, train model, and add prediction logic |
| 14 | Test accuracy, speed, and failure cases |
| 15 | Complete demo, README, video, and presentation |

### Make It Yours — breed-to-group labeling

The customization is the breed-to-AKC-group labeling and the veterinary
motivation, not simply running a pretrained model. The planned Stanford Dogs
workflow is:

1. Inventory all 120 source breed identifiers and preserve their exact names.
2. Resolve aliases to a canonical breed/variety using official AKC breed pages.
3. Record the source identifier, canonical name, AKC group, source URL, and
   review status in a table; explicitly exclude unresolved or unsupported labels.
4. Review every row, check for missing and duplicate identifiers, then apply the
   reviewed labels to images before capped sampling and split creation.

The implemented prototype instead maps **ImageNet output labels**, in
[gnasher_group_mapping.py](gnasher_group_mapping.py). Its current assignments
are exported in [data/imagenet_breed_to_group.csv](data/imagenet_breed_to_group.csv).
That export is not a completed Stanford Dogs labeling table or evidence of
independent human review. See [data/README.md](data/README.md) for aliases,
exceptions, sources, and remaining labeling work.

## Run in Google Colab

1. Download notebooks/Gnasher_Group_Final.ipynb and gnasher_group_mapping.py
   from this repository.
2. Upload the notebook to Google Colab.
3. Upload the mapping file and your test photograph through Colab’s
   Files sidebar into /content.
4. Set image_path to the photograph’s exact path.
5. Set expected_group from the known breed before testing.
6. Run the cells in order.

The first model load downloads pretrained weights and needs internet
access. The current implementation runs on CPU.

The saved notebook displays the Border Collie example and the results
recorded during the original session. A fresh runtime starts a new
results dictionary; it does not automatically reload previous CSV rows.

## Files

- notebooks/Gnasher_Group_Final.ipynb — implementation, saved outputs,
  reflections, photograph sources, and limitations
- gnasher_group_mapping.py — breed-to-group mapping and score aggregation
- data/imagenet_breed_to_group.csv — current ImageNet label assignments and exclusions
- results/gnasher_test_results.csv — eight recorded photograph results
- requirements.txt — required Python packages
- docs/AI_usage_log.md — AI assistance and student testing log
- docs/proposal.pdf — original blueprint

## Measured results

Seven dog photographs, one representing each group, and one cat
photograph were tested.

- Correct final dog-group predictions: 6 of 7 (85.7%).
- Dog abstentions: 1 of 7, the husky.
- Cat outcome: Uncertain; this was not a cat classification.
- Average CPU model forward pass: 0.0991 seconds over 30 measured
  runs after five warm-up runs.
- Timing excludes image loading, preprocessing, and group mapping.

The husky’s leading supported group was Working at 20.3%.
Most of its score went to the unsupported ImageNet label “Eskimo dog,”
so the system returned Uncertain.

## Changes from the blueprint

The blueprint proposed Stanford Dogs fine-tuning, approximately
7,000 images, and a 70/15/15 training/validation/testing split.

The implemented prototype instead uses pretrained EfficientNet-B0
without additional training, followed by breed-to-group mapping.
The proposed dataset preparation and training split were not implemented.

## Limitations and target comparison

These are eight hand-picked demonstration photographs, not a large
independent benchmark. Poppy and Goober were used during development
of the 50% threshold. Possible overlap between internet photographs
and model pretraining data is unknown. Internet-photo breed labels
were not independently verified.

The original 85% unseen-image accuracy target is therefore not
established. CPU forward-pass timing was below one second, but GPU
performance and total application response time were not measured.

Future work includes a larger held-out evaluation, threshold validation,
mixed-breed and non-dog testing, and fine-tuning a seven-group classifier.

## Demonstration and presentation

The narrated demonstration video and final presentation will be linked
here after they are completed.

## Sources and AI assistance

Photograph sources and technical references are documented in the
notebook. Internet photographs remain subject to their owners’ rights.

ChatGPT helped generate starter code, the mapping module, debugging
guidance, and documentation drafts. I ran the notebook, selected and
tested photographs, recorded outputs, and edited the project.
See docs/AI_usage_log.md for details.
