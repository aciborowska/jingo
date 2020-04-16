# Just-in-time Adaptable Bug Localization - JINGO

This repository contains the code and data associated with the following research paper:
`
A. Ciborowska, M.J. Decker, K. Damevski, "Just-in-Time Adaptable Bug Localization Based on 
Changesets" _IEEE Transactions on Software Engineering_, 2020. _submitted_
`

# Requirements
* Python3.6
* git
* Packages listed in _requirements-3.txt_ file: `pip install -r requirements.txt`

# Running the project
To run the project with default options simply use the following command:

`python3.6 main.py [options]`

### Options:
* `--datasets` - path to datasets: _datasets/corley_ or _datasets/bench4bl_. Default: _datasets/bench4bl_.
* `--name` - name of project to run the experiment on. If not provided, run experiments for all projects in the dataset.
* `--model-type` - type of BL model to use. Use _joined_ to run JINGO, _changesets_ to run Corley et al.[1]. Default: _joined_.
* `--topics` - number of topics for the bug report model and the changeset model. Format: [bug_topics, changeset_topics]. Default: [50, 100].
* `--decays` - decays for for the bug report model and the changeset model. Format [bug_decay, changeset_decay]. Default: [0.75, 1.0].
* `--gamma` - boosting parameter for JINGO prediction. Default: 1.0.
* `--omega` - boosting for the number of fixed issues to observe to train T matrix. Default: 1.0.
* `--save-model` - save model after training (flag). Default: off.
* `-v` - enable verbose output (flag). Default: off.

All options are defined at the top of _main.py_.


#### Example:
Run JINGO with the following configuration:
* dataset - _datasets/corley_
* project - BookKeeper
* the bug report model - 25 topics, 0.5 decay
* the changeset model - 200 topics, 0.85 decay
* omega = 1.0
* gamma = 5.0

`python3.6 main.py --datasets datasets/corley --name bookkeeper --num-topics 25 200 --decays 0.5 0.85 --omega 1.0 --gamma 5.0`

### Results
Results for each project are stored in _results/project_name_ folder.

### References

