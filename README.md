# DeepSpeed-Task2
AlphaPilot gate detector using localization as regression

# TODO:
  * Pre-crop images using a state-of-the-art detector such as yolo, inception, mobilenet, etc
  * Data augmentation
  * Get someone who knows about CNNs to review network architecture
  * Test on different colorspaces

# How to use the scripts:
  * (almost) all scripts are configured with a few variables at the top.
  * When a script is provided a json file, that determines which images it will load. All images must be in the directory specified.
  * If an OUTDIR is specified, you must mkdir it first.
  * Python3 everything
  * Use tscherli/alphatraining docker image. Should be configured for all three nodes now.
  * Use "sbatch runTask2Training.py" to train, don't train in an interactive session.
  * Follow the cluster tutorial for more info.

# Scripts (All located in Task2 folder)
  * Train/current-best.py: current best model with training code. Must be configured with model output path and model parameters.
  * Train/basic<n>.py: Training scripts to mess with. If you beat current-best, overwrite it. These are called by runTask2Training<n>.sh.
  * Starter-scriptsv2/generate-results.py: Inferencing script. Must be given path to model.
  * Starter-scriptsv2/generate-submission.py: Submission generator script. Call this to run inferencing. Must be configured with image source, (training or leaderboard) and the json labels output file.
  * drawboxes.py: Draws boxes on the image. Configured with ground truth and prediction json files. Set "NOGT" to run without ground truth specified.
  * scorer-scriptsv2/score-detections.py: comes up with a map score for your detection. There is some corruption in goodlabels so for whatever reason you can only metric against training_labelsv2.json.
  
# Other info:
  * .h5 models and the images themselves are ommited due to size.
  * Login to the cluster to actually use this. Don't try to use locally unless you already have keras/tensorflow/cuda working.
  * Talk to Tom for cluster login info
