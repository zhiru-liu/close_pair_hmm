import os

HMM_BLOCK_SIZE = 10
HMM_MIN_SEQ_LEN = 100 # if the sequence is too short, we don't want to make any inferences

# Default location of the bundled reference priors, resolved relative to this
# package so the install works on any machine / editable install. These are the
# LiuGood2024 QP-SNV priors shipped with the package (see priors/README.md).
# Any inference run can override this by passing `prior_path=...` to the model
# or pipeline, e.g. to use a run-specific prior generated into its own work folder.
HMM_PRIOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'priors')
HMM_PRIOR_BINS = 40
