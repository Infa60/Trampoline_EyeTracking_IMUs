# Trampoline_EyeTracking_IMUs

This repository is linked to a research project on the visual strategies of trampolinists. The aim is to identify differences in visuomotor behavior between elite and subelite athletes during the execution of 4-/, 41/, 42/ and 43/. The scientific article related is [not submitted yet] (DOI to come).

We measured the kinematics with IMUs (Xsens) and the visual startegies with a wearable eye-tracking (Pupil invisible). The Xsens data is exported in HD with 'no level' mode and is extracted from the .mvnx with codes available in xsens_data_unpack. The Pupil data is manually labled (with codes available in trampoline_bed_labeling) to identify characteristic points on the trampoline bed.

Codes in metrics allow to extrack the metrics compared in the article such as:
_Primary analysis_
- Fixation number and duration
- Quiet eye duration
- Neck and eye movement amplitude
- Neck and eye max amplitude
_Exploratory analysis_
- Spreading of the heatmap of the gaze on the trampoline bed (90th percentile)
- Temporal evolution of the projected visual orientation
- Proportion of the acrobatic when the athlete look at the trampoline, the walls the ceiling and himself
- Proportion of the acrobatic when the athlete is doing anticipatory movements, compensatory movements, spotting, movement detection or blink.

If you are using part of this code please cite:
{title={Gaze strategies of elite and sub-elite trampolinists during varying difficulty levels of twisting somersaults}, author={Charbonneau, Eve and Begon, Mickael and Romeas, Thomas.}, year={2022}}

Please feel free to email eve.charbonneau.1@umontreal.ca if you have any questions regarding this study or code :)
