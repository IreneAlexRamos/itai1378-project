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

## Stanford Dogs source and original labeling plan

Dataset link: [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/).
The original plan used its 20,580 photographs and 120 breed labels to create
seven AKC group labels, with at most 1,000 photographs per group and a
70/15/15 split. That dataset preparation was not completed.

The labeling procedure appears in the main README's **Make It Yours** section.
A future Stanford Dogs table must preserve the dataset's exact breed identifiers,
add canonical breed names, AKC groups, source URLs, and review status, and
document exclusions. Do not treat ImageNet label spellings as an automatic
replacement for the dataset identifiers.

## Implemented ImageNet mapping

[imagenet_breed_to_group.csv](imagenet_breed_to_group.csv) is an export of
the assignments and explicit exclusions already present in
[gnasher_group_mapping.py](../gnasher_group_mapping.py).
It contains 114 mapped labels and 4 excluded dog labels.
This is a labeling artifact for the current prototype, not a Stanford Dogs
training manifest. The reference column points to AKC group directories;
it does not certify that every row has been individually checked.

Examples of the naming decisions in the existing code:

| ImageNet label | Interpreted breed/variety | Current group |
|---|---|---|
| French bulldog | French Bulldog | Non-Sporting |
| pug | Pug | Toy |
| Boston bull | Boston Terrier | Non-Sporting |
| toy poodle | Toy Poodle | Toy |
| miniature poodle / standard poodle | Miniature / Standard Poodle | Non-Sporting |

Official references:
[French Bulldog](https://www.akc.org/dog-breeds/french-bulldog/),
[Pug](https://www.akc.org/dog-breeds/pug/),
[Boston Terrier](https://www.akc.org/dog-breeds/boston-terrier/),
and [AKC breed groups](https://www.akc.org/dog-breeds/).

Ambiguous or unsupported labels—`toy terrier`, `kelpie`, `Appenzeller`,
and `Eskimo dog`—remain in the unsupported-dog bucket. In particular,
the code does not assume that “Eskimo dog” means American Eskimo Dog.

The aggregation checks duplicate mapped labels, missing expected model labels,
and probability totals. Those checks validate code structure and score
accounting; they do not establish that each taxonomy assignment is correct.
ChatGPT assisted with the original mapping. Irene still needs to review the
exported assignments and sources before treating the table as verified
student labeling.
