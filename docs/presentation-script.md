# Gnasher Group | 10-minute speaking script

Rehearse against a timer at your own pace. Spend 60–90 seconds on the demonstration excerpt, then allow brief pauses on the data and results slides. Use your own words wherever the script does not reflect your experience.

## 1. Gnasher Group

Hi, I’m Irene Ramos. My project is Gnasher Group, a Tier 1 computer vision project. I wanted to see whether one dog photo could help someone identify a likely American Kennel Club group. I’ll walk through what I built, show what the system does, and explain what I learned from comparing two versions.

## 2. The problem

I chose this problem because I’ve worked in animal care. Similar-looking dogs can be hard to tell apart, even when you see them regularly. I wanted a learning tool that could give users a starting point. An AKC group does not tell us an individual dog’s behavior or prove its breed.

## 3. What I built

Here is the pipeline. First I upload a photo and convert it to RGB. The model needs a consistent input, so the image goes through the expected crop, resize and normalization. EfficientNet produces seven scores, one per group. The notebook displays the highest group or Uncertain, plus two candidates so we can inspect what the model considered.

## 4. Demonstration

Here I show the actual inference notebook after loading the saved checkpoint. I upload one known-label example and look at the image and output. Then I show the cocker spaniel. Sporting was expected. The saved output led with Non-Sporting at 49.4 percent, so the system abstained. I want the viewer to see an input go through the workflow and watch the difference between a usable prediction and a result that needs review. [Show a 60–90 second excerpt or live demo here.]

## 5. How it works

EfficientNet-B0 started with ImageNet weights. In plain language, the model already knew useful visual patterns and I adapted it to this task. I replaced the last layer with seven outputs. I trained that head first, then allowed the last three feature blocks to update at a lower learning rate. I used validation data to choose a checkpoint and a threshold of fifty percent.

## 6. The data

Stanford Dogs started with 20,580 photos of 120 breeds. I used a breed-to-group table to turn breed labels into group labels. The final run took 7,000 images, evenly spread across seven groups. There were 4,900 train, 1,050 validation and 1,050 test images. The code checks exact duplicate bytes. Photos of a breed can still occur in multiple splits, and near duplicates are possible.

## 7. Blueprint targets

My Blueprint target was eighty-five percent accuracy. V2 reached 90.19 percent when I scored the highest candidate and 89.43 percent when Uncertain counted as a miss. Both use the test images held out from my fine-tuning. I can’t verify those images were unseen in ImageNet pretraining. The GPU forward pass averaged 8.12 milliseconds. That timing starts after the image is prepared. I didn’t measure the full upload-to-answer experience.

## 8. What changed

I first used an off-the-shelf model and summed its breed scores into groups. Then I trained Version 2 directly on seven groups. The most surprising result was that V1 was more accurate on the same test images. It reached 97.14 percent top-1 versus 90.19 percent for V2. V2 abstained on 23 images and V1 on 42. V2 gave an answer more often, while the older baseline produced fewer wrong decisions on the full test set. I can’t isolate the cause from this comparison.

## 9. Challenges and fixes

One challenge was assigning breed names to the appropriate AKC group. I kept aliases and exclusions visible so they can be reviewed. Another challenge was keeping comparisons consistent. I fixed the split, kept the threshold from validation, and tracked checkpoint and report hashes. Looking at the learning curves also helped me separate training performance from how the model did on validation.

## 10. Lessons and next steps

The biggest lesson was to compare versions on the same data and take mistakes seriously. Next I’d test on fresh photos with known labels, check near duplicates, and use a breed-disjoint split where possible. I’d time the whole prediction path and test photos outside the seven groups before suggesting broader use.

## 11. Where V2 struggled

These examples show two failure modes. The cocker spaniel got an Uncertain result because its leading score fell below the threshold. The Borzoi got a confident Working prediction despite its expected Hound label. That difference is why I report both accuracy and coverage, and why I would not treat the score as a guarantee. The matrix also helps identify which groups tend to get confused.

## 12. Evidence and links

The project repository links to the notebook, original Blueprint and saved run evidence. It includes the V1 prototype so someone can see the progression. I’ve credited the data, tools and AI assistance. Thank you. I’m happy to explain the split, model or results.
