# Experiment logs

Text logs, loss plots and job scripts from the experiments I ran after the tutor feedback (see section 5b of the main README). No model files and no data are included. Paths inside the logs are Rangpur paths.

Folders (each holds predict.txt, training logs and loss plots where they exist):
- `ctrl_code`: control run of the original settings on Rangpur.
- `exp_code/outputs`: training variants (`aug`, `lr3e4`, `ep60`, `ep100`, `ep100lr3e4`), the 5-nearest-photos settings tried on the 60-epoch model (`ep60_k*`), and `diagnose.txt` for the control and 60-epoch models.
- `exp2_code/outputs/pre`: gray-world colour correction.
- `exp6_code/outputs`: scoring variants on the control and 60-epoch models, margin 2.0 (`m20`) and same-source pairs (`same`).
- `exp7_code/sweeps`: the sweep over reference-set sizes and k. `exp7_code/outputs`: 100-epoch runs for seeds 7 and 123, baselines at learning rate 0.0003 for seeds 42, 7 and 123, and the lab-trained seeds re-scored.
- `seeds_lab`: logs from the lab T4 runs for seeds 7 and 123.
- `exp1.sh` to `exp11.sh`, `ctrl.sh`: the Slurm job scripts I submitted.

Selection was done on the validation split only. Test numbers are in the same files but were not used to choose anything.
