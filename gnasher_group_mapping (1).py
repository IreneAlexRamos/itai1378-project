"""Gnasher checkpoint 2: ImageNet breed scores -> seven AKC group scores.

ChatGPT assisted with this mapping and its checks. Review it before using it.
This is postprocessing of pretrained outputs, NOT a trained seven-class head.
Scores are raw softmax mass, not calibrated confidence or breed ancestry.

Taxonomy references (AKC group directories, including subsequent pages):
https://www.akc.org/dog-breeds/sporting/
https://www.akc.org/dog-breeds/hound/
https://www.akc.org/dog-breeds/working/
https://www.akc.org/dog-breeds/terrier/
https://www.akc.org/dog-breeds/toy/
https://www.akc.org/dog-breeds/non-sporting/
https://www.akc.org/dog-breeds/herding/

Keys below preserve Torchvision's exact ImageNet label spelling.
Examples of aliases: Boston bull = Boston Terrier; Japanese spaniel = Japanese
Chin; Blenheim spaniel = Cavalier King Charles Spaniel; Brabancon griffon is
the smooth-coated Brussels Griffon; Mexican hairless = Xoloitzcuintli.
Cocker-spaniel varieties are both Sporting. Poodle varieties are handled
separately: Toy -> Toy; Miniature and Standard -> Non-Sporting.
"""

import math

BREEDS_BY_GROUP = {
    "Sporting": [
        "Weimaraner", "flat-coated retriever", "curly-coated retriever",
        "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
        "German short-haired pointer", "vizsla", "English setter", "Irish setter",
        "Gordon setter", "Brittany spaniel", "clumber", "English springer",
        "Welsh springer spaniel", "cocker spaniel", "Sussex spaniel", "Irish water spaniel",
    ],
    "Hound": [
        "Rhodesian ridgeback", "Afghan hound", "basset", "beagle", "bloodhound",
        "bluetick", "black-and-tan coonhound", "Walker hound", "English foxhound",
        "redbone", "borzoi", "Irish wolfhound", "whippet", "Ibizan hound",
        "Norwegian elkhound", "otterhound", "Saluki", "Scottish deerhound", "basenji",
    ],
    "Working": [
        "giant schnauzer", "standard schnauzer", "kuvasz", "komondor", "Rottweiler",
        "Doberman", "Greater Swiss Mountain dog", "Bernese mountain dog", "boxer",
        "bull mastiff", "Tibetan mastiff", "Great Dane", "Saint Bernard", "malamute",
        "Siberian husky", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed",
    ],
    "Terrier": [
        "Staffordshire bullterrier", "American Staffordshire terrier", "Bedlington terrier",
        "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier",
        "Norwich terrier", "wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier",
        "Airedale", "cairn", "Australian terrier", "Dandie Dinmont", "miniature schnauzer",
        "Scotch terrier", "soft-coated wheaten terrier", "West Highland white terrier",
    ],
    "Toy": [
        "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu",
        "Blenheim spaniel", "papillon", "Italian greyhound", "Yorkshire terrier",
        "silky terrier", "miniature pinscher", "affenpinscher", "pug", "Pomeranian",
        "Brabancon griffon", "toy poodle",
    ],
    "Non-Sporting": [
        "Boston bull", "Tibetan terrier", "Lhasa", "schipperke", "French bulldog",
        "dalmatian", "chow", "keeshond", "miniature poodle", "standard poodle",
        "Mexican hairless",
    ],
    "Herding": [
        "groenendael", "malinois", "briard", "Old English sheepdog", "Shetland sheepdog",
        "collie", "Border collie", "Bouvier des Flandres", "German shepherd",
        "EntleBucher", "Pembroke", "Cardigan",
    ],
}

# Deliberately not forced into a group. These are still dog-class outputs.
# Exclusion is a conservative project decision, not a claim that a breed is
# unrecognized by every kennel organization.
UNSUPPORTED_DOG_LABELS = {
    "toy terrier": "Generic label; do not assume a specific AKC breed/variety.",
    "kelpie": "No seven-group assignment used in this proof of concept.",
    "Appenzeller": "No seven-group assignment used in this proof of concept.",
    "Eskimo dog": "Ambiguous label; do not assume American Eskimo Dog.",
}

BREED_TO_GROUP = {}
for group, breeds in BREEDS_BY_GROUP.items():
    for breed in breeds:
        if breed in BREED_TO_GROUP:
            raise ValueError(f"Duplicate breed: {breed}")
        BREED_TO_GROUP[breed] = group


def aggregate_group_scores(labels, probabilities):
    """Sum all scores, preserving unsupported-dog and non-dog mass separately.

    The seven group scores plus two excluded buckets sum to approximately one.
    Do not renormalize the group scores: that would hide non-dog evidence.
    This function ranks groups but does not decide when to abstain.
    """
    scores = [float(value) for value in probabilities]
    if len(labels) != len(scores) or len(set(labels)) != len(labels):
        raise ValueError("Expected one score per unique model label.")
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("Scores must be finite and nonnegative.")
    if not math.isclose(sum(scores), 1.0, abs_tol=1e-5):
        raise ValueError("Expected softmax probabilities summing to one.")
    expected = set(BREED_TO_GROUP) | set(UNSUPPORTED_DOG_LABELS)
    missing = expected - set(labels)
    if missing:
        raise ValueError(f"Model label set does not match mapping: {sorted(missing)}")

    groups = {group: 0.0 for group in BREEDS_BY_GROUP}
    unsupported = 0.0
    non_dog = 0.0
    for label, score in zip(labels, scores):
        if label in BREED_TO_GROUP:
            groups[BREED_TO_GROUP[label]] += score
        elif label in UNSUPPORTED_DOG_LABELS:
            unsupported += score
        else:
            non_dog += score
    return {
        "groups": groups,
        "unsupported_dog": unsupported,
        "non_dog": non_dog,
        "ranked_groups": sorted(groups.items(), key=lambda item: item[1], reverse=True),
    }
