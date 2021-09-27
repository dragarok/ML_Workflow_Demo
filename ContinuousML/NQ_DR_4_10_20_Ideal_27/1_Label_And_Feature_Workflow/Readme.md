# Feature Importance Readme
- [x] Build out dvc workflow for determining feature importance on Full_features.csv
  - [x] dvc.yaml
    - [x] dep: 
      - [x] ./src/Feature_Importance.py script for feature reduction
      - Full_Features.csv
      - [x] Params.yaml
        - Number of features: 300
    - [-] out:
      - [x] ./2_Training_Workflow/Feature_Importance.csv or txt
        - A single column with featrure column name (Feature_1 through Feature_2973) 
      - [-] Feature_Importance_Output.md
        - Using CML, add plotly visuals, etc to Markdown
- [x] Setting up GCP remote dvc repository
  - [x] Create repo in name of new gmail id
    - Alok_Roger@gmail.com
      - $300 budget to play with
      - Host on dvc remote on GCP gsutil  
- [x] .github/workflows/cml.yaml Globally...need to create a GitHub Actions workflow that calls dvc repro 
# Links 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zi6cdD2aPM9rf3bcUOPWdnp1GVaCfjsN?usp=sharing) Notebook exploring rudimentary feature selection techniques
