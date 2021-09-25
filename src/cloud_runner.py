#!/usr/bin/env python3

import shutil
import tensorflow_cloud as tfc
from git import Repo
from tfc_cloud import run_model_training
import subprocess
import sys


gcp_bucket = 'tfc-cml'

print("Moved file to src")
tfc.run(
    entry_point='../../../src/tfc_cloud.py',
    requirements_txt='../../../src/requirements-tfcloud.txt',
    chief_config=tfc.MachineConfig(
            cpu_cores=8,
            memory=30,
            accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=1),
    docker_image_bucket_name=gcp_bucket,
)
token = ''
branch_name = ''
github_username = ''
repo_main_url = ''
# token = os.getenv('GIT_TOKEN')
# branch_name = os.getenv('BRANCH_NAME')
git_url = "https://" + github_username + ":" + token + "@" + repo_main_url
repo = Repo.clone_from(git_url, 'cloned_repo')
repo.git.checkout(branch_name)
print("Cloned the repo")
# subprocess.run(['pyenv', 'activate dvc_tfc'])
# ret = subprocess.run([command], capture_output=True, shell=True)
# print(ret)
pull_cmd_dvc = 'cd cloned_repo; dvc pull --run-cache'
process = subprocess.Popen(pull_cmd_dvc, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
for line in iter(process.stdout.readline, b''):
    sys.stdout.write(line.decode(sys.stdout.encoding))
print("Pulled the data")
run_model_training()



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
