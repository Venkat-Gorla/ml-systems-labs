import os
from sagemaker.sklearn.estimator import SKLearn

role = os.getenv("SAGEMAKER_ROLE_ARN")
if role is None:
    raise ValueError("SAGEMAKER_ROLE_ARN environment variable is not set")

estimator = SKLearn(
    entry_point="sagemaker_train.py",
    source_dir="src",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
)

estimator.fit({
    "train": "s3://ml-systems-labs-spam-detection-vegorla/input/train/spam.csv"
})
