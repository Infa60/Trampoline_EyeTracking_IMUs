# Introduction
This repository is linked to a research project on the visual strategies of trampolinists. The aim of the project is to identify differences in visuomotor behavior between elite and subelite athletes during the execution of four simple acrobatics: 4-/, 41/, 42/ and 43/. The scientific article related is [not submitted yet] (DOI to come). We measured the kinematics with IMUs (Xsens) and the visual startegies with a wearable eye-tracking (Pupil invisible). 

# Let's jump into the code!
The Xsens data was exported in HD with 'no level' mode and was extracted from the .mvnx with codes available in the ![xsens_data_unpack](xsens_data_unpack) folder. The Pupil data was manually labled (with codes available in the ![trampoline_bed_labeling](trampoline_bed_labeling) folder) to identify characteristic points on the trampoline bed and the timing of the eye blinks. The gaze orientation in the gymnasium reference frame was reconstructed using a vector based approach. Multiple metrics were extracted from the reconstructed gaze orientation projectec on the gymnasium (with codes available in the ![metrics](metrics) folder). The metrics were represented graphically and compared with statistical tests (with codes availbale in the ![analysis](analysis) folder).

Briefly, the codes allow to compare the elite and subelite groups in terms of:
_Primary analysis_
- Number of fixations
- Fixations duration
- Quiet-eye duration
- Neck movements amplitude
- Eye movements amplitude

_Exploratory analysis_
- Spreading of the heatmap of the gaze on the trampoline bed (90th percentile)
- Temporal evolution of the symmetrized projected gaze orientation
- Dwell time in the following area of interest: trampoline, trampoline bed, walls, ceiling, and themself
- Proportion of the acrobatic when the athlete exibits the following characteristic neck and eye movements: anticipatory, compensatory, spotting, movement detection, or blink.

If you use the code available here, please cite:
```bibtex
@misc{Charbonneau2023,
  author = {Charbonneau, Eve and Begon, Mickael and Romeas, Thomas},
  title = {Gaze strategies of elite and sub-elite trampolinists during varying difficulty levels of twisting somersaults.},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/EveCharbie/Trampoline_EyeTracking_IMUs}}
}
```

# Requirements
In order to run the analysis, you need to install the following python packages:
```bash
conda install -c conda-forge numpy matplotlib scipy pandas tqdm biorbd opencv ipython itertools casadi quaternion requests pprint seaborn tkinter
```
