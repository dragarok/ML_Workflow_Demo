#!/usr/bin/env python3

import shutil
import tensorflow_cloud as tfc

gcp_bucket = 'tfc-cml'

# Move file to src folder for embedding inside docker
shutil.move("Full_Features.csv", "../../../src/Full_Features.csv")
print("Moved file to src")
tfc.run(
    entry_point='../../../src/tfc-cloud.py',
    requirements_txt='../../../src/requirements-tfcloud.txt',
    distribution_strategy="auto",
    chief_config=tfc.MachineConfig(
            cpu_cores=8,
            memory=30,
            accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=2),
    docker_image_bucket_name=gcp_bucket,
)
# Move file to original place again
print("Moved file back from src to Folder no 1")
shutil.move("../../../src/Full_Features.csv", "Full_Features.csv")