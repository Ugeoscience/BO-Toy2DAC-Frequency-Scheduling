BO-Toy2DAC Frequency Scheduling

This repository contains the reproducibility materials for the manuscript: "Automatic frequency scheduling for multi-scale full-waveform inversion via Bayesian optimization"

The project implements a Bayesian-optimization framework for automatic frequency-schedule design in frequency-domain full-waveform inversion (FWI). Candidate frequency schedules are encoded by five interpretable parameters and evaluated through complete multi-scale FWI runs using Toy2DAC. A Gaussian-process surrogate with a log-Expected-Improvement acquisition function is used to guide the search toward improved schedules.

Repository structure:

BO-Toy2DAC-Frequency-Scheduling/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── config/
│   ├── bo_search_bounds.yaml
│   ├── expert_schedule.yaml
│   ├── random_search_seeds.yaml
│   └── grid_search.yaml
│
├── scripts/
│   ├── run_bo_search.py
│   ├── run_expert_schedule.py
│   ├── run_random_search.py
│   ├── run_grid_search.py
│   ├── run_noise_tests.py
│   ├── run_starting_model_test.py
│   └── compute_metrics.py
│
├── toy2dac_templates/
│   ├── fdfd_input_template
│   ├── toy2dac_input_modeling_template
│   ├── toy2dac_input_inversion_template
│   └── acquisition_geometry/
│
├── schedules/
│   ├── expert_schedule.csv
│   ├── bo_best_schedule.csv
│   ├── random_search_schedules.csv
│   └── grid_search_schedules.csv
│
├── results/
│   ├── bo_history.csv
│   ├── random_search_results.csv
│   ├── grid_search_results.csv
│   ├── noise_results.csv
│   ├── starting_model_results.csv
│   └── cost_logs.csv
│
├── figures/
│   ├── make_figure_1.py
│   ├── make_figure_2.py
│   └── ...

    
Requirements
The Bayesian-optimization controller was developed in Python. The main Python dependencies are:
  Python 3.11
  NumPy
  SciPy
  pandas
  matplotlib
  scikit-learn
  PyTorch
  BoTorch
  GPyTorch
  PyYAML
  
Install the Python dependencies using: "pip install -r requirements.txt" or create an equivalent conda environment.

External software

  The FWI simulations are performed using Toy2DAC, a 2D acoustic frequency-domain FWI code developed by the SEISCOPE Consortium.
  Toy2DAC is not redistributed in this repository. Users should obtain Toy2DAC directly from the SEISCOPE distribution page and follow its licensing terms and installation instructions.
  Toy2DAC distribution page: "https://seiscope2.osug.fr/sites/seiscope2.osug.fr/IMG/tgz/toy2dac_v2.6_2019_05_24-2.tgz"
  After installing Toy2DAC, update the executable path in the configuration files or scripts before running the experiments.

Data

  The Marmousi synthetic velocity model is publicly available through the SEG Wiki: "https://wiki.seg.org/wiki/Dictionary:Marmousi_model"
  
  The Toy2DAC distribution also includes Marmousi example files that can be used as the basis for the numerical setup.
  Large generated files, including synthetic data or full recovered velocity models, are not stored directly in this repository.
  

Reproducibility notes

  The Bayesian-optimization workflow uses fixed random seeds for the Latin-Hypercube warm start, random-search baselines, and noise realizations. 

Citation

  If you use this repository, please cite the associated manuscript: "Mehmet Ali Uge, Automatic frequency scheduling for multi-scale full-waveform inversion via Bayesian optimization, submitted."
  The final citation will be updated after publication.

License:

  The Python scripts and configuration files developed for this study are released under the BSD 3-Clause License. See the LICENSE file for details.
  Toy2DAC is external software and is subject to its own licensing terms.
