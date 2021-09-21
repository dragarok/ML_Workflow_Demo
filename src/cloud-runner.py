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
            accelerator_count=1),
    docker_image_bucket_name=gcp_bucket,
)
# Move file to original place again
print("Moved file back from src to Folder no 1")
shutil.move("../../../src/Full_Features.csv", "Full_Features.csv")
# TODO Look if a better way is available

# TODO Watching job status updates
# projectName = 'your_project_name'
# projectId = 'projects/{}'.format(projectName)
# jobName = 'your_job_name'
# jobId = '{}/jobs/{}'.format(projectId, jobName)

# request = ml.projects().jobs().get(name=jobId)

# response = None

# try:
#     response = request.execute()
# except errors.HttpError, err:
#     # Something went wrong. Handle the exception in an appropriate
#     #  way for your application.
