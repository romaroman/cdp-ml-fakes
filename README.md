## Machine learning attack on copy detection patterns: are 1x1 patterns cloneable?

The research was supported by the [SNF](http://www.snf.ch) project No. 200021_182063. 
##

The public repositoty for a paper ["Machine learning attack on copy detection patterns: are 1x1 patterns cloneable?"](https://arxiv.org/abs/2110.02176) 

Nowadays, the modern economy critically requires reliable yet cheap protection solutions against product counterfeiting for the mass market. Copy detection patterns (CDP) are considered as such a solution in several applications. It is assumed that being printed at the maximum achievable limit of a printing resolution of an industrial printer with the smallest symbol size $1\times1$, the CDP cannot be copied with sufficient accuracy and thus are unclonable. In this paper, we challenge this hypothesis and consider a copy attack against the CDP based on machine learning. The experimental results based on samples produced on two industrial printers demonstrate that simple detection metrics used in the CDP authentication cannot reliably distinguish the original CDP from their fakes under certain printing conditions. Thus, the paper calls for a need of careful reconsideration of CDP cloneability and search for new authentication techniques and CDP optimization facing the current attack.

## Data

Data and comprehensive description can be found [here](http://sip.unige.ch/projects/snf-it-dis/datasets/indigo-base/).

## Reqirements

The most important packages are listed in `env.yml`. If you use conda you can create a new environment with this list of packages by
    $ conda env create -f env.yml

# Estimation
## Train

    $ python train_estimator.py --config_path configuration.yml --type Dtt_Dt --lr 0.0001 --epochs 100 --is_stochastic True --is_debug False

## Test

    $ python test_estimator.py --config_path configuration.yml --symbol_size 8 --target_symbol_size 1 --type Dtt_Dt --lr 0.00001 --epoch 100 --is_symbol_proc True --thr 0.5 --is_debug False

# Authentication

## Metrics

    $ python metrics.py path/to/templates --bsize 684 --dens 50 --cpus 6 --debug False

## SVMs

    $ python svms.py metrics.csv --cpus 6

## Citation
R. Chaban, O. Taran, J. Tutt, T. Holotyak, S. Bonev and S. Voloshynovskiy, "Machine learning attack on copy detection patterns: are 1x1 patterns cloneable?" in Proc. IEEE International Workshop on Information Forensics and Security (WIFS), Montpellier, France 2021. 
  
    @inproceedings { Chaban2021wifs,
        author = { Chaban, Roman and Taran, Olga and Tutt, Joakim and Holotyak, Taras and Bonev, Slavi and Voloshynovskiy, Slava },
        booktitle = { IEEE International Workshop on Information Forensics and Security (WIFS)},
        title = { Machine learning attack on copy detection patterns: are 1x1 patterns cloneable? },
        address = { Montpellier, France },
        month = { December },
        year = { 2021 }
    }
