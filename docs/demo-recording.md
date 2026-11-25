# Record the Gnasher Group demo

## Set up before recording

1. Download only `best.pt` from **My Drive → GnasherGroup → experiments → v2_run04**. Keep it outside the public repository.
2. In the [Stanford Dogs archive](https://vision.stanford.edu/aditya86/ImageNetDogs/images.tar), extract these exact files. Keep them local.
   - `Images/n02099267-flat-coated_retriever/n02099267_2269.jpg`
   - `Images/n02102318-cocker_spaniel/n02102318_11704.jpg`
3. Open [Gnasher_Group_Inference.ipynb in Colab](https://colab.research.google.com/github/IreneAlexRamos/itai1378-project/blob/main/notebooks/Gnasher_Group_Inference.ipynb). A CPU runtime works. Run **1. Load the checkpoint** and upload `best.pt` when prompted. Wait for `v2_run04 ready` and a locked threshold of 50%. A hash error signals the wrong or altered file. Re-download the original.
4. Run **2. Upload and classify one photo** and select `n02099267_2269.jpg`. Wait for Sporting and a score around 96.8%. Run only cell 2 again and select `n02102318_11704.jpg`. Wait for Uncertain and a warning. The top candidates should be Non-Sporting around 49.4% and Sporting around 49.1%. Both JPEG hashes and these outcomes matched in fresh CPU testing. Colab's upload widget itself still needs your rehearsal.
5. Before recording, connect the runtime, load the checkpoint, test your microphone, and capture only the notebook tab. In the recording, execute cell 2 on camera for both results. Narrate the displayed decisions as they happen.

## Give Patricia access to the checkpoint

In Drive, right-click **best.pt** itself → **Share** → enter Patricia's confirmed HCC email → choose **Viewer** → **Send**. Keep **General access** set to **Restricted**, and verify Viewer download is allowed. Send her the public Colab link as well. She can download this one file and upload it in cell 1. Sharing the parent folder is unnecessary. The demo video uses a separate link and access setting.

## Examples

| Use | File or saved output | Expected result |
|---|---|---|
| Known-label success | n02099267-flat-coated_retriever/n02099267_2269.jpg | Sporting; V2 saved Sporting at 96.76%. |
| Abstention | n02102318-cocker_spaniel/n02102318_11704.jpg | Expected Sporting. Saved model led with Non-Sporting at 49.4% and returned Uncertain. |
| Confident mistake | n02090622-borzoi/n02090622_5025.jpg | Expected Hound. Saved result Working at 91.5%. |
| Backup V1 | Archived border collie and husky output | Herding 95.24% for Border Collie; Uncertain for husky. State you switched versions. |

## Narration · about four minutes

**0:00–0:40** “This is Gnasher Group, my Tier 1 computer vision project. It suggests one of seven AKC groups from a dog photo. I’ve loaded the saved V2 model and its original 50 percent decision threshold. I can now upload an image and run a prediction without training anything.”

**0:40–1:45** Upload the success case and wait for the output. “The notebook displays the photo, leading group, model score and another candidate. The score describes the model’s preference among its seven outputs. I compared this with the expected group from the held-out dataset.”

**1:45–3:05** Upload the cocker spaniel. “This image is labeled Sporting. The model's leading prediction in the saved run was Non-Sporting, just under 50 percent, so it returned Uncertain. That threshold was selected on validation data. The model can also be confidently wrong. In the saved test, a Borzoi was marked Working even though its expected group was Hound.” Show the real output and say what it does.

**3:05–4:00** “On 1,050 photos held out from fine-tuning, V2 got 90.19 percent top-1 and 89.43 percent when abstentions counted as misses. The older mapping baseline had higher accuracy on these same photos. V2 answered more often. This is a dog-group classifier with no separate non-dog output. That limit matters when users upload something outside the intended inputs.”

## Upload and link

1. Trim dead time and confirm the video is between three and five minutes. Keep the upload, prediction and narrated failure visible.
2. Upload to YouTube as **Unlisted**, or to Google Drive with **Anyone with the link → Viewer** on the video file. Open the link signed out and play the video.
3. Replace the pending demo line in the root README with the actual clickable URL. Add the same live link to slide 4 in the editable PowerPoint. Re-export the PDF and replace docs/presentation.pdf with that export. Submit the named PDF in Canvas.
4. Follow Canvas for the section format. Online asynchronous students post a 10-minute recording link in the Canvas comments. On-campus students present live and keep this video ready as backup.

## Likely questions

- **What is transfer learning?** I started with EfficientNet-B0 weights that already encode visual patterns, fitted a new seven-output head, then fine-tuned the last three feature blocks.
- **How did you keep evaluation fair?** I used a fixed image-level split, chose the model and threshold on validation, and evaluated both versions on the same test images. Images may still overlap ImageNet pretraining.
- **Why did V1 win?** On this benchmark, the original ImageNet breed scores mapped effectively into the seven groups. Possible pretraining overlap is one explanation. I did not isolate the cause.
- **What is top-1 versus final accuracy?** Top-1 checks the leading group even when the score is low. Final accuracy counts every Uncertain as a miss.
- **Does score mean probability of correctness?** It is a softmax score among seven possible groups. It was not calibrated as probability of correctness.
- **Why 50 percent?** It was the lowest tested threshold that achieved the required validation coverage and accepted accuracy.
- **What does 8.12 ms measure?** One GPU model forward pass with an already prepared input on a Tesla T4. Uploading, decoding and preprocessing are outside that measurement.
- **Can it identify a cat or a mixed breed?** The model returns a dog group or Uncertain. It has no cat or mixed-ancestry class.
