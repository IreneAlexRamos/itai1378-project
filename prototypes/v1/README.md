# Gnasher Group — Version 1 preserved prototype

Archived from repository commit `fc9f9f19cffcc3fea3bd76ce69add40f7662f87d` on 2026-10-13.
The notebook, mapping module, and eight-photo CSV are exact copies of that
working version. Its saved outputs are retained.

This version uses pretrained ImageNet EfficientNet-B0 and postprocessing.
No new model was trained. A provisional 50% rule can return Uncertain and
show two group candidates. The eight photographs are demonstration examples,
not an independent accuracy benchmark.

To run: open the notebook in Colab; upload this folder's mapping module and
a photograph into /content, set image_path and expected_group, and run in order.
A fresh runtime starts a new results table.

Version 2 is developed separately in notebooks/Gnasher_Group_Training.ipynb.
Original prototype files also remain at their existing repository paths,
so existing links continue to work.
