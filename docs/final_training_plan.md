# Gnasher Group — final training plan and learning progression

## What is preserved

The prototype snapshot was copied exactly from commit `fc9f9f19cffcc3fea3bd76ce69add40f7662f87d`.
It contains the working notebook, mapping module, and eight-photo CSV.
Its demonstration rate and CPU timing remain Version 1 evidence.

The existing final slide deck describes Version 1. Keep it as a prototype
presentation; revise the final deck only after the Version 2 results exist.

## What Version 2 adds

1. Student-reviewed breed-to-AKC-group labeling from actual Stanford Dogs IDs.
2. A maximum of 1,000 unique images per group with a custom 70/15/15 split.
3. A seven-output classification head trained on the reviewed labels.
4. Fine-tuning of the last three feature blocks with a smaller learning rate.
5. Validation-only checkpoint and confidence-threshold selection.
6. Both versions evaluated on the identical test photographs.
7. Learning curves, a seven-group confusion matrix and an eight-column
   decision matrix that retains Uncertain, failure examples, and timing.

The original 85% and GPU speed targets remain targets, not promised results.

## Irene's steps in Colab

Open [Gnasher_Group_Training.ipynb](../notebooks/Gnasher_Group_Training.ipynb)
and select a GPU. Run one section at a time:

1. Set up the source and mount Drive.
2. Download Stanford Dogs (or upload the original images.tar if its host fails).
3. Inspect the suggested label table. Verify each breed/variety with an AKC
   source. Correct labels and aliases; explain exclusions. Mark rows reviewed
   only after doing the checks, with your name in reviewed_by.
4. Upload the reviewed CSV using the notebook's import switch. It validates
   the uploaded table before replacing the saved one.
5. Create the fixed manifest and inspect actual group/split counts.
6. Train both stages; checkpoints and history are saved to Drive each epoch.
7. Inspect training/validation curves and lock the threshold.
8. Evaluate on test once, inspect actual mistakes, and measure speed.
9. Save the notebook with outputs. Keep best.pt and other large files in Drive.
   Copy the reviewed labeling table and small reports/plots to GitHub when ready.

If interrupted, best.pt and last.pt can still be loaded for inference. This
initial workflow does not implement optimizer-state resume: a new training run
gets a new experiment name and preserves the earlier run's artifacts.

## Honest comparison

The old 6/7 demonstration result is not a comparable independent benchmark.
Use the new paired report for both models on the same test split. Report:

- Top-1 accuracy on all test images.
- Final-decision accuracy, counting abstentions in the denominator.
- Coverage, number of abstentions, and accuracy among accepted decisions.
- Per-group scores and actual errors.
- Actual hardware and timing scope.

The 0.50 V1 rule is retained. V2 selects among declared thresholds on validation.
That is a comparison of the two systems, not an isolated experiment on training
alone. Full top-1 accuracy is also reported without their abstention rules.

Stanford Dogs uses ImageNet images. Holding examples out of this fine-tuning
does not prove they were absent from the pretrained EfficientNet's data.
The split is custom, not the dataset's official train/test protocol.
Exact-byte duplicates are removed; near duplicates may remain.
A seven-group model has no cat class and can confidently misclassify non-dogs.

Do not choose new settings using the final test score and then call reuse of
that score an unbiased confirmation. Record exploratory iterations honestly;
a fresh independent evaluation is needed after test-informed tuning.

## What to say about progression

- “My first version taught me image loading, pretrained inference, mapping,
  uncertainty, and recording results.”
- “For the final version I reviewed dataset labels, trained a task-specific
  output layer, fine-tuned part of the network, and compared both systems on
  the same held-out images.”
- State the second sentence only after you perform that work.
- Explain one actual labeling decision, one observation from the learning
  curves, and one real failure. Improvement is something to measure, not assume.

## Validation completed before the real run

Offline tests exercise human-review gates, unknown-label exclusions, exact
duplicate isolation, conflicting-label rejection, reproducible stratification,
and protection against changed images/settings. Notebook and module syntax are
checked. The local environment has no PyTorch/GPU, so actual model execution,
Stanford downloading/extraction, and end-to-end Colab training are not yet tested.

## Sources

- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [TensorFlow Datasets description](https://www.tensorflow.org/datasets/catalog/stanford_dogs)
- [AKC breed directory](https://www.akc.org/dog-breeds/)
- [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

