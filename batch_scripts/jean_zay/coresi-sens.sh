#!/bin/bash

# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

#SBATCH -A hbu@v100
#SBATCH -C v100-16g
#SBATCH --job-name=coresi-sens
#SBATCH --nodes=1
# nombre total de taches (= nombre de GPU ici)
#SBATCH --ntasks-per-node=1
# nombre de GPU (1/4 des GPU)
#SBATCH --gres=gpu:1
# nombre de coeurs CPU par tache (2/4 du noeud 4-GPU)
#SBATCH --cpus-per-task=20
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
# hyperthreading desactive
#SBATCH --hint=nomultithread
# temps maximum d'execution demande (HH:MM:SS)
#SBATCH --time=20:00:00
#SBATCH --output=coresi-sens%j.out
#SBATCH --error=coresi-sens%j.out

module purge
module load pytorch-gpu/py3/2.3.0

cd coresi_python
envsubst < config.templated.yaml > "config.yaml"
python -m coresi.main -c config.yaml --sensitivity -v
