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
“Uncertain.” Group scores are not calibrated guarantees of correctness.
The system does not reliably detect cats or identify mixed-breed ancestry.

**Flow:** photograph → preprocessing → EfficientNet-B0 →
breed-to-group aggregation → suggested group or Uncertain

## Run in Google Colab

1. Download Gnasher_Group_Final.ipynb and gnasher_group_mapping.py
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

- Gnasher_Group_Final.ipynb — implementation, saved outputs,
  reflections, photograph sources, and limitations
- gnasher_group_mapping.py — breed-to-group mapping and score aggregation
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
