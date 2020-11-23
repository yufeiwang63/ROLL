source ~/.bashrc
conda activate ROLL
export PYTHONPATH=${PWD}:$PYTHONPATH
export PJHOME=${PWD}
alias viskit='python rllab/viskit/frontend.py'
alias pullseuss='python chester/pull_result.py seuss'