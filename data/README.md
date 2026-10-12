# Gnasher Group — Demonstration Data

## Photographs used

The implemented prototype was tested on eight photographs:
seven dogs, one representing each AKC group, and one cat.

Poppy and Goober are personal photographs. The other six
photographs came from internet sources listed in the notebook.
Internet-photo breed labels were not independently verified.

## Evaluation limitations

Poppy and Goober were used during development of the 50% threshold.

These photographs are a small, hand-picked convenience sample,
not an independent benchmark. Possible overlap between internet
photographs and the model's pretraining data is unknown.

## Changes from the original plan

No Stanford Dogs training subset or 70/15/15 split was created.
The implemented system uses pretrained ImageNet EfficientNet-B0
without additional training.

The original training plan is retained in docs/proposal.pdf.

## Results and sources

- Recorded results: results/gnasher_test_results.csv
- Photograph sources and limitations:
  notebooks/Gnasher_Group_Final.ipynb

Internet photographs remain subject to their owners' rights.

## Testing another photograph

Upload your photograph and gnasher_group_mapping.py to Colab's
/content folder. Set image_path to the photograph's exact path,
then run the prediction, mapping, and decision cells.
