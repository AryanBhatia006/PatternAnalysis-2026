# Siamese network for melanoma triage (ISIC, COMP3710 Hard tier)

DRAFT v2. Every [CHECK], [YOUR OBSERVATION] and [YOUR DECISION] marker needs you. Delete this line and every marker when done.

## 1. What this does

A skin clinic gets a lot of photos of skin spots. A reviewer has to decide which ones look harmless, which ones need a closer look, and which ones look dangerous (melanoma). This project builds a small model that gives each photo a score for how malignant it looks, then uses two cut-offs to sort photos into three groups: auto-clear, refer to a specialist, or flag. I compare two models on exactly the same test photos:

- A **baseline**: a plain convolutional network that looks at one photo and outputs one number.
- A **Siamese network**: a network that turns a photo into a short list of numbers (an embedding), trained so that photos of the same class end up close together and photos of different classes end up far apart.

The main question is whether the score can be trusted. Does it separate the cases it is sure about from the ones it is guessing, or does it stay confident when it is wrong? And does the extra complexity of the Siamese network pay off?

### How it works

Both models use the same four blocks of convolution, batch normalisation, ReLU and max-pooling (32, 64, 128 and 256 channels), then global average pooling. The baseline ends in one linear layer with one output and is trained with a weighted binary cross-entropy loss (the weight is the number of benign images divided by the number of malignant images in the training split). The Siamese network ends in a linear layer with 128 outputs, then each embedding is scaled to length 1. It is trained on pairs of photos with a contrastive loss: similar pairs (same class, label 0) are pulled together, and dissimilar pairs (label 1) are pushed apart until they are at least the margin (1.0) away. Because every embedding has length 1, the distance between two of them is always between 0 and 2.

To score a new photo, I embed 200 benign and 200 malignant reference photos from the training split. The score is the average distance to the 5 nearest benign references minus the average distance to the 5 nearest malignant references. A higher score means the photo looks more like the malignant references.

Figure: `figures/score_hist.png` shows the test scores of both models. Dashed lines are the two thresholds chosen on the validation split.

![score histograms](figures/score_hist.png)

## 2. Feasibility review (shown to my tutor at a prac)

[CHECK: paste the final text you actually showed your tutor, and write the date and any feedback they gave. This is the version from my notes; edit it to match what you presented.]

1. Need and criteria. A clinic reviewer wants help sorting skin photos into clear, flag, or send to a specialist. My question is whether the model's score can be trusted. Success means: (a) the baseline and the Siamese network are scored on the same test split with AUROC, PR-AUC, recall and balanced accuracy, with intervals; plain accuracy is not enough, because saying "benign" every time scores 91.8% on my data; (b) thresholds come from validation only, and I report the real miss rate in the auto-clear tier on test, with an interval; (c) I report each image source separately, because the sources have different malignant rates; (d) each model trains in under 2 hours and 4 GB on one GPU.
2. Model. A Siamese network trained from scratch with contrastive loss, compared with a baseline CNN. I kept the 14,061 images that have a usable ID (4,794 IDs, 8.5% malignant) and split 70/15/15 by ID.
3. Evidence so far. The split passed two separate overlap checks and the full runs finished.
4. Risks and fallback. HAM_ IDs have one image each, so the split is only guaranteed patient-clean for IP_ images. HAM images are about 21% malignant and IP images about 5%, so a model may partly recognise the source. I dropped about 10,400 images with no usable ID, and they held 77% of the malignant cases. Results are one seed and one split. If the Siamese network does not beat the baseline, I report that.

## 3. Dependencies and reproducibility

Laptop (Windows, CPU; used for the evaluation and the per-source analysis): Python 3.14.7, torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.5.2, pandas 3.0.6, matplotlib 3.11.2, pillow 12.3.0.

Lab GPU machine (UQ Virtual GPU Lab, Tesla T4; used for training): Python 3.10, torch 2.14.0+cu130, torchvision 0.29.0+cu130. The course environment on that machine was used as it was, and I did not record the pandas and matplotlib versions there.

Seeds and settings:
- Seed 42 for the split, the reference set, the bootstrap and training. The validation and test pairs use seeds 43 and 44 (seed plus 1 and seed plus 2).
- Two runs of the split with seed 42 gave identical tables. Seed 7 gave a different split (test set of 2,079 images with 182 malignant, instead of 2,151 with 176).
- Training uses the Adam optimiser with learning rate 0.001 and PyTorch's default betas (0.9, 0.999), no weight decay, no learning-rate schedule and no gradient clipping. Batch size 32. Margin 1.0. Both models were trained for 30 epochs (the default in `train.py` is 20, so `--epochs 30` was passed) with 3 data-loading workers. The checkpoint with the lowest validation loss is kept, and the test split is used once, at the end.
- `train.py` sets `torch.backends.cudnn.deterministic = True` and `benchmark = False`. On the lab machine cuDNN failed to load, so I disabled it with a small wrapper (below). The wrapper is not part of `train.py`. It makes convolutions slower and changes nothing else that I know of.

Commands (the data folder holds `cleaned_metadata.csv` and `combined_images/`):

```
python recognition/Siamese_Network_s5057318/train.py <data_dir> --model siamese --epochs 30 --num_workers 3
python recognition/Siamese_Network_s5057318/train.py <data_dir> --model baseline --epochs 30 --num_workers 3
python recognition/Siamese_Network_s5057318/predict.py <data_dir> --no_cudnn
python recognition/Siamese_Network_s5057318/check_metrics.py
python recognition/Siamese_Network_s5057318/verify_split.py
```

Run them from the repository root. `verify_split.py` has the data folder written at the top of the file (`DATA = r"C:\isic_data"`), so edit that line first.

On the lab machine the training runs went through this wrapper (saved as `run_train.py` in my home folder there, not committed):

```python
import sys, runpy, torch
torch.backends.cudnn.enabled = False  # cuDNN fails to load on this machine
model, epochs, workers = sys.argv[1], sys.argv[2], sys.argv[3]
repo = "/home/s5057318/PatternAnalysis-2026/recognition/Siamese_Network_s5057318"
sys.path.insert(0, repo)
sys.argv = ["train.py", "/home/s5057318/isic_data", "--model", model,
            "--epochs", epochs, "--num_workers", workers]
runpy.run_path(repo + "/train.py", run_name="__main__")
```

A checkpoint holds the model weights, the epoch, the validation loss and the arguments, but not the normalisation numbers, so `predict.py` recomputes them from the training split. Model weights and the dataset are not committed.

## 4. Data and preprocessing

Source: the 224 by 224 resized ISIC melanoma copy linked in the assignment spec (Kaggle, `roua91/isic-melanoma-dataset-resized-to-224x224`). It has 24,493 images and a file `cleaned_metadata.csv` with `isic_id`, `patient_id` and `target`.

The `patient_id` column is messy. Some values are placeholders like `dummy_123`, and 1,326 rows hold two IDs in quotes, for example `dummy_2932,HAM_0004431`. I kept only rows with a real (non-dummy) ID, which leaves:

| | Images | IDs | Malignant |
|---|---|---|---|
| Kept | 14,061 | 4,794 | 1,195 (8.50%) |
| Dropped | 10,432 | n/a | 4,071 (39.0%) |

So the rows I dropped hold 77.3% of all the malignant images in the file, and my 8.5% malignant rate is lower than the rate in the full file (21.5%). The kept IDs come from two sources: IP_ IDs (1,919 IDs over 11,186 images, about 5.8 images each) and HAM_ IDs (2,875 IDs over 2,875 images, exactly one image each).

Split: 70/15/15 by real ID, so all images with the same ID go to the same split. IDs are dealt out one at a time, IDs with malignant images first, to whichever split is furthest below its share of images and malignant images. Seed 42.

| Split | Images | IDs | Malignant |
|---|---|---|---|
| Train | 9,881 | 3,367 | 833 (8.43%) |
| Val | 2,029 | 711 | 186 (9.17%) |
| Test | 2,151 | 716 | 176 (8.18%) |

Checks: `check_no_id_overlap` in `dataset.py`, and a second script, `verify_split.py`, that rebuilds every ID from the raw csv and counts how many splits each ID appears in. Both give zero IDs in more than one split, and 14,061 images with 14,061 unique image IDs.

Important limitation: a HAM_ ID has one image, so it cannot stop the same patient's different lesions landing in different splits. I can only say the split is patient-safe for the IP_ images. I did not check how many HAM patients this affects.

Images are converted to RGB, resized to 224 by 224 and normalised with a mean and standard deviation computed on the training split only (mean 0.815, 0.613, 0.590 and standard deviation 0.151, 0.188, 0.209 for red, green and blue). Training images also get random horizontal and vertical flips and a random rotation of up to 15 degrees either way; validation and test images get no augmentation (checked by loading the same image twice and getting identical tensors). The rotation leaves black corners, which is why the smallest values in a training batch are about -5.4.

Pairs for the Siamese network: each epoch 9,880 pairs from the training split (the number is rounded down to a multiple of 4). Half are similar and half dissimilar. Similar pairs are split evenly between benign-benign and malignant-malignant, so the rare malignant images are shown much more often. Dissimilar pairs are one benign and one malignant image. Pairs are only built inside one split. Training pairs are redrawn each epoch; validation and test pairs are drawn once from a fixed seed and are identical on every run. The baseline simply shuffles the training images.

Example inputs. Eight training images with their label and ID (all eight happen to be HAM_ images; the data also contains IP_ images). The black corners come from the random rotation used on training images:

![training images](figures/grid.png)

Six validation pairs. Label 0 means a similar pair (same class) and label 1 a dissimilar pair:

![validation pairs](figures/pairs.png)

## 5. Results

Test split: 2,151 images, 176 malignant. A model that says "benign" for everything gets 0.9182 accuracy and finds no melanomas, so I judge models on AUROC, PR-AUC and recall. Intervals are 95% bootstrap intervals made by resampling IDs (1,000 resamples).

| | Baseline | Siamese |
|---|---|---|
| AUROC | 0.8668 [0.8375, 0.8951] | 0.8464 [0.8127, 0.8767] |
| PR-AUC | 0.4483 [0.3765, 0.5330] | 0.2918 [0.2412, 0.3651] |
| Siamese minus baseline, AUROC | [-0.0392, -0.0008] | |
| Siamese minus baseline, PR-AUC | [-0.2190, -0.0857] | |

The baseline is ahead on both. The AUROC gap is small and its interval only just excludes zero. The PR-AUC gap is large. This is one seed and one split, so I cannot say how much of the AUROC gap would survive a different seed.

### Triage rule

Both thresholds come from the validation split only. `t_low` is the highest threshold that still keeps at least 95% of the validation malignant images above it. `t_high` is the lowest threshold where validation precision reaches 0.5. A photo scoring below `t_low` is auto-cleared, one at or above `t_high` is flagged, and the rest go to a specialist.

| | Baseline | Siamese |
|---|---|---|
| t_low / t_high | -0.9492 / 1.2939 | -0.0770 / 0.0823 |
| Auto-clear: images | 1,041 (48.4%) | 1,012 (47.0%) |
| Auto-clear: malignant missed | 9 of 176 (5.1%) | 8 of 176 (4.5%) |
| Auto-clear miss rate, 95% interval | [2.71%, 9.43%] | [2.32%, 8.71%] |
| Refer: images / malignant | 918 / 83 | 1,075 / 146 |
| Flag: images / malignant | 192 / 84 | 64 / 22 |
| Flag tier malignant share | 43.8% | 34.4% |
| Recall at t_low / at t_high | 0.9489 / 0.4773 | 0.9545 / 0.1250 |
| Precision at t_low / at t_high | 0.1505 / 0.4375 | 0.1475 / 0.3438 |
| Balanced accuracy at t_low / at t_high | 0.7357 / 0.7113 | 0.7314 / 0.5519 |

Accuracy at the two thresholds (worked out from the tier counts above, not printed by my script): baseline 0.557 at `t_low` and 0.907 at `t_high`; Siamese 0.545 and 0.909. The assignment mentions about 0.8 accuracy, but with this much imbalance accuracy mostly reflects how many images are called benign, so I do not use it to rank the models.

### Does the score say how sure the model is?

Partly. The share of malignant images rises as the score rises, for both models. For the baseline it rises at every step (0%, 0%, 0.5%, 1.4%, 2.8%, 3.3%, 7.4%, 10.7%, 14.9%, 40.9% across ten equal groups of test images). For the Siamese model it rises with small dips (for example 4.7% then 3.3% then 4.7% in the middle groups) and ends at 34.4%. The top tenth of the baseline scores holds 88 of the 176 malignant images and the top tenth of the Siamese scores holds 74.

The two scores are on different scales, so I compare each model's mistakes with its own correct answers, using the gap between a score and the threshold:

| | Baseline | Siamese |
|---|---|---|
| Missed melanomas: median gap below `t_low` | 0.52 | 0.071 |
| Correctly cleared benign images: median gap below `t_low` | 1.95 | 0.127 |
| Ratio | 0.27 | 0.56 |
| Benign images wrongly flagged: median gap above `t_high` | 0.71 | 0.0145 |
| Melanomas correctly flagged: median gap above `t_high` | 1.25 | 0.0072 |
| Ratio | 0.57 | 2.0 |

For the baseline, the further a score is from the threshold, the more likely it is right: its misses sit close to the line, and its false flags score lower than its true flags. For the Siamese model the same holds weakly on the clearing side (ratio 0.56), but in the flag tier its false flags score higher than its true flags (ratio 2.0), so a higher Siamese score did not mean a more reliable flag. These are medians over small groups (9 and 8 misses, 22 correct flags), and I did not test them statistically. The baseline does make a few confident mistakes: `ISIC_3778192` scores -3.09 against a clearing threshold of -0.95, a gap of 2.14, which is as deep as a typical correctly cleared benign image.

### By image source

The IP_ and HAM_ images have very different malignant rates in the test split (IP 89 of 1,721, which is 5.2%; HAM 87 of 430, which is 20.2%), so a model could do well by recognising the source instead of the lesion. I report them separately with the same thresholds.

| | IP (1,721 images, 89 malignant) | HAM (430 images, 87 malignant) |
|---|---|---|
| Always-benign accuracy | 0.9483 | 0.7977 |
| Baseline AUROC / PR-AUC | 0.8075 / 0.1998 | 0.8915 / 0.6691 |
| Siamese AUROC / PR-AUC | 0.7849 / 0.1675 | 0.8410 / 0.4479 |
| AUROC bootstrap, baseline | [0.7613, 0.8518] | [0.8589, 0.9220] |
| AUROC bootstrap, Siamese | [0.7290, 0.8357] | [0.8027, 0.8739] |
| Siamese minus baseline AUROC | [-0.0571, 0.0086] | [-0.0853, -0.0197] |
| Auto-clear malignant missed (baseline / Siamese) | 9 of 89 (10.1%) / 8 of 89 (9.0%) | 0 of 87 / 0 of 87 |

All 9 melanomas the baseline cleared, and all 8 the Siamese model cleared, are IP_ images, so for IP_ images roughly 1 melanoma in 10 would be missed, double the overall figure. On IP_ images I cannot separate the two models (the interval for the AUROC difference includes zero); on HAM_ images the baseline is clearly better. I do not know why. Both models score higher on HAM_ images, but HAM_ is also the more malignant source, so I compare AUROC, which does not depend on how common melanoma is, and I cannot rule out that some of the HAM_ result comes from recognising how those photos look.

### Resources

| | Baseline | Siamese |
|---|---|---|
| Parameters | 389,153 | 421,792 |
| Training time, 30 epochs (T4) | 1,076 s (about 18 min) | 2,370 s (about 39.5 min) |
| Peak GPU memory during training | 1,085 MB | 2,147 MB |
| Best epoch (by validation loss) | 25 | 27 |
| Test inference, GPU (batch of 64) | 0.91 ms per image | 0.92 ms per image |
| Peak GPU memory, test inference | 837.4 MB | 837.4 MB |

The Siamese model also needs a one-off step of embedding the 400 reference images (0.47 s on the T4). Both models show the same inference memory, probably because the batch of 64 images dominates it; I did not test this. Training curves are in `figures/baseline_loss.png` and `figures/siamese_loss.png` (from the T4 runs). The two training losses are different quantities and cannot be compared with each other.

![baseline loss](figures/baseline_loss.png)
![siamese loss](figures/siamese_loss.png)

The training numbers come from the training logs, and the GPU inference numbers come from running `predict.py` on the lab T4. The metrics, tiers and per-source tables come from running `predict.py` on my laptop CPU. An earlier run of the same script on the T4 gave identical overall metrics and tier counts; the per-source tables were only produced on the laptop.

## 6. Failure cases

`figures/failures.png` shows, for each model, the three malignant test images with the lowest scores and the two benign test images with the highest scores.

![failure cases](figures/failures.png)

[YOUR OBSERVATION: for each case below, look at the image yourself and write what you see in your own words. The notes are my first impressions from the picture, and I cannot judge whether a lesion is malignant. Anything about why the model failed is a guess to be tested, not a finding.]

1. `ISIC_3778192` (IP_8972853, malignant). Both models score it as benign (baseline -3.087, below its clearing threshold of -0.949 by 2.14; Siamese -0.199, below its threshold of -0.077 by 0.122, the second largest gap among its 8 misses). First impression: blurry, pink, no sharp lesion edge. Possible trigger: the lesion is hard to see.
2. `ISIC_6526489` (IP_5550558, malignant, baseline -2.170). First impression: a dark brown lesion with a dark centre and small dots around it. Possible trigger: I do not know; this one looks like it has features I would expect to score as malignant.
3. `ISIC_6285925` (IP_5399626, malignant, Siamese -0.169). First impression: a small, pale lesion on light skin with hairs and ruler-like marks. Possible trigger: small lesion, low contrast.
4. `ISIC_0031993` (HAM_0006887, benign, baseline 4.834, the baseline's highest-scoring benign image). First impression: a red textured surface with hairs and no clear lesion. Possible trigger: strong red colour.
5. `ISIC_6712494` (IP_6017204, benign, baseline 4.481). First impression: purple ink marks on pale skin in a close-up with other objects. Possible trigger: ink marks.

The two models share five misses: `ISIC_3778192`, `ISIC_6352565`, `ISIC_6516710`, `ISIC_1159891` and `ISIC_8207974` are cleared by both, which suggests these images look benign to both models and not that the errors are random. One ID can also appear more than once (`IP_8702124` has two images among the baseline's misses and the same two among the Siamese model's), which is why the bootstrap resamples IDs.

A way to test the trigger ideas: count how many of the false flags (108 benign images in the baseline's flag tier, 42 in the Siamese flag tier) have ink, rulers or strong colour, and compare with the rate in all benign images. I did not do this.

## 7. Recommendation to the project manager

[YOUR DECISION: the facts below are mine; the yes or no and the wording are yours. Change it if you disagree.]

Can we use this model? Not as it is, and if we did, use the baseline and not the Siamese network.

- Neither model is safe enough to clear photos without a human. Both would auto-clear about 47 to 48% of photos, which saves a lot of review, but they would miss about 1 melanoma in 20 overall and about 1 in 10 among IP_ images. That miss rate comes from the validation rule, and its 95% interval on test is roughly 2.3% to 9.4%, so it is not a guarantee.
- The Siamese network costs about 2.2 times the training time and about twice the training memory, takes the same time per image, and does not beat the baseline (PR-AUC 0.29 against 0.45). It catches fewer melanomas in the flag tier (22 of 176, against 84), and its score is a less reliable sign of confidence than the baseline's. On this data the extra complexity is not justified.
- What I would fix before any deployment: test on photos from other clinics and on the full data including the images I dropped; use a split that is guaranteed patient-safe for all images; check whether the model is recognising the image source; repeat with more seeds; and look at whether a stricter `t_low` (for example one aiming for 99% recall on validation) still auto-clears enough photos to be worth it.
- A sensible way to use it today: as a sorting aid, where every photo in the refer and flag tiers goes to a person, and auto-cleared photos are still sampled and checked.

## 8. Limitations

- One seed and one split. The bootstrap intervals only measure how much the test set could vary, not how much the training could vary. The seed 7 split moved the test set from 176 to 182 malignant images, which shows how much one split can matter.
- I dropped 10,432 images with no usable ID, and they held 77% of the malignant images. The kept data is less malignant than the full file, so the numbers here are not clinic numbers.
- HAM_ IDs have one image each, so the split is only patient-safe for IP_ images.
- Image source may act as a shortcut (see the by-source table).
- Thresholds were picked on 186 validation malignant images, so they are noisy.
- Validation and test pairs come from a small number of malignant images used many times.
- cuDNN was disabled on the lab machine for training, through a wrapper that is not committed.

## 9. Artificial Intelligence Usage Disclosure

[CHECK AND REWRITE IN YOUR OWN WORDS. This is a draft of what happened, based on the conversation. Add details from your own notes and correct anything that is wrong.]

Tools: Claude Code (in the terminal) for writing and running code, and the Claude chat assistant for planning, writing the prompts I gave Claude Code, reading outputs, suggesting checks, and drafting this README.

What the AI wrote:
- Claude Code wrote `inspect_data.py`, `dataset.py`, `modules.py`, `train.py`, `predict.py`, `make_grid.py`, `check_modules.py`, `check_loss.py`, `check_pairs.py` and `check_metrics.py`.
- Claude chat wrote `check_loader.py` and `verify_split.py`, the lab wrapper `run_train.py`, the commands I ran on Rangpur and the lab machine, and a first draft of the feasibility review and of this README.

How I checked it: [CHECK each line is true for you.]
- Hand-calculated expected values for the contrastive loss (for example distance 0.3 gives 0.045 for a similar pair and 0.245 for a dissimilar pair), and for the metric functions (AUROC 0.75 and PR-AUC 0.8333 on a four-point example).
- Showed that the loss test fails when the two terms of the loss are swapped (7 of 10 checks failed, then all passed after restoring the file).
- Checked the pair labels of six validation pairs against the raw csv, and that each pair only uses images from one split.
- Ran a separate script that rebuilds every ID from the raw csv and counts splits per ID (zero overlaps), ran the seed experiment, and confirmed the same seed gives the same split.
- Checked that validation and test images get no augmentation, that the normalisation uses the training split only, and that thresholds are chosen on validation only.
- Compared the overall metrics from the laptop CPU run and the lab GPU run.
- Checked that the tier counts and per-source counts add up to the totals, and that `modules.py` contains no numpy.
- Asked Claude Code to compare this README with the code line by line. It found seven mistakes (a wrong command, a wrong claim about where cuDNN was disabled, two wrong statements about the score deciles and the Siamese model's confidence, a miscounted list of shared misses, a wrong description of one score gap, and a wrong statement about seeds), which are fixed in this version.

Things that went wrong along the way: [YOUR NOTES: for example, debug checkpoints and plots from earlier runs left in the output folder, the cuDNN failure on the lab machine, unpacking an incomplete upload, git commands run in the wrong folder, and anything else you remember.]

The commit log has 15 commits from 18 October to 27 October (before this README). [CHECK: update the count when you finish.]

## 10. References

- Hadsell, Chopra and LeCun (2006), Dimensionality reduction by learning an invariant mapping (contrastive loss). [CHECK full reference]
- Koch, Zemel and Salakhutdinov (2015), Siamese neural networks for one-shot image recognition (from the assignment spec).
- ISIC 2020 challenge data and the Kaggle resized copy used here. [CHECK: add the dataset reference and the Kaggle dataset page.]
- HAM10000 dataset (the HAM_ images). [CHECK: add the reference.]
