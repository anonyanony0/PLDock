# PLDock

## Neural Protein Ligand Docking: Dataset and Evaluation

## Project Page
[PLDock](https://anonyanony0.github.io/PLDock/)

PLDock is a comprehensive neural protein-ligand docking dataset with a tool for training and evaluating machine learning-based protein-ligand docking models. It contains real scenario-based protein-ligand docking tasks, splits, baselines and metrics, with the goal of training and evaluating machine learning-based protein-ligand docking models. PLDock provides more than 70,000 protein-ligand complex structures, more than 150,000 protein-ligand affinity data, 3 typical tasks, 5 types of structured data splits and 9 evaluation metrics.

## Environment
* Base environment
```
python=3.8
pytorch=1.11
cudatoolkit=11.3
rdkit
openbabel
biopython
biopandas
dgllife
mpi4y
torchdrug==0.1.2
```

It is recommended to use conda to manage the environment

```bash
# Install conda
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh

# Clone PLDock from gethub
git clone https://github.com/anonyanony0/PLDock
cd PLDock

# Create environment by conda from 'environment.yml' here:
conda env create --name PLDock -f environment.yml
conda activate PLDock

# Install PLdock
pip3 install -e .
```
Or Install PLdock by pip
```bash
# Clone project:
git clone https://github.com/anonyanony0/PLDock

cd PLDock

# Install
pip3 install -e .

# Install external:
pip3 install -r requirements.txt
```
* Soft Requirements
For Tankbind, gcc>=5.3 and torchdrug==0.1.2 are need. Torchdrug==0.1.3 may cause molecular feature dimension error.
## How to use PLDock

### Data

Preprocessed PLDock dataset with features is available from [google drive](https://drive.google.com/drive/folders/1_WSo3_ceuSFPHK_hVrFtcXqK0C9LoGnQ). 

You can download it and unzip to your directory.

### Extract features
We have release structures and features in .dill files.\ 
However, you can all also extract feratures from raw structures.
```python
# import data_feature function
from PLDock.data.data import data_feature
# Setting your directory of data
input_dir = '/root/data/structure_refined'
output_dir = '/root/data/structure_features'
#hhsuite_db:hh-suite database can get from https://github.com/soedinglab/hh-suite
hhsuite_db = '/data/install/hhsuitdb/dbCAN/'
num_cpus = 50 # Multi processes 
data_feature(input_dir, output_dir, hhsuite_db, num_cpus)
```
### Extract features from your dataset
If you want to extract features from your own dataset, you should create a directory containing the ligand and receptor file for each complex. PLdock accepts ligand files of the formats .mol2/.sdf  with 'ligand' in their names or protein files of the formats .pdb  with 'protein' in their names. Like:
```
Your_directory
└───1a1b
    │   1a1b_protein.pdb
    │   1a1b_ligand.sdf
└───1a2b
    │   1a2b_protein.pdb
    │   1a2b_ligand.mol2
...
```

### Load data
Load structures with features from .dill files you download:
```python
# import dataloader
from PLDock.data.data import data_feature
data_dir = '/root/data/structure_features'
# 'data_list' is a file with a name of structure file in each line 
data_list = '/root/data/structure_list.txt'
dataloader(data_dir,data_list)
# Or you can just set the directory
```

### Evaluation
To evaluate on the binding pose prediction task
```python
from PLDock.evaluation.binding_pose import binding_pose
#core: a list file with a pdb code in each line titled by'#code'
core='/root/PLDock/CoreSet.dat'
#score: a dir of files with a pose code and a docking score in each line titled by'#code' and 'score'
score='/root/PLDock/PLDock/X-Score'
#rmsd: a dir of files with a pose code and a rmsd in each line titled by'#code' and 'rmsd'
rmsd='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/decoys_docking'
# docking_out: output file
dp = binding_pose('docking_out')
dp.calcul(core,rmsd,score)
```
To evaluate on the binding affinity prediction task

```python
from PLDock.evaluation.binding_affinity import scoring,ranking
#core: a list file with a pdb code in each line titled by'#code'
core='/root/PLDock/CoreSet.dat'
#rscore: a file with a pdb code and a docking score in each line titled by'#code' and 'score'
rscore='/root/PLDock/X-Score.dat'
dpr = scoring('docking_out')
dpr.scoring(core,rscore)
dp = ranking('ranking_out')
dp.ranking(core,score)
```
To evaluate on the virtual_screening task
```python
from PLDock.evaluation.binding_pose import binding_pose
#core: a list file with a pdb code in each line titled by'#code'
core='/root/PLDock/CoreSet.dat'
#score: a dir of files with a pose code and a docking score in each line titled by'#code_ligand_num' and 'score'
score='/root/PLDock/PLDock/X-Score'
#traget: a file with a target code and ligand codes each line titled by'#T' and 'L1', 'L2' ...
target='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_screening/TargetInfo.dat'
dp = screening('screening_out')
dp.screening(core,score,target)
```
## Models
We provide [Equibind](https://github.com/HannesStark/EquiBind) as an example model because its code is cleaner relative to other models such as [GNINA](https://github.com/gnina/gnina). [TankBind](https://github.com/luwei0917/TankBind) is now also supported. We are adding support for other models.
### Equibind
#### Training 
Input data list, structure directory and output directory are needed.Other parameters supported by equbind, such as epoch and config, can also be specified in the function.The model file will be saved in the output_directory.
```python
from PLDock.models.Equibind.train import equibind_train
data_dir = '/root/PLDock/structure'
out_dir = '/root/PLDock/test_data'
equibind_train(train_names=data_dir+'/train.txt',val_names = data_dir+'/valid.txt',pdb_dir=data_dir,logdir=out_dir, num_epochs = 10)
```
#### To test your model on the test set. 
It is important to specify your test data list (test_names), structure directory (inference_path) and model directorys(models_dir). The RMSD, kabsch RMSD, centroid distance, and other indicators of the results will be saved in the output_directory.
```python
from PLDock.models.Equibind.inference import equibind_test
data_dir = '/root/PLDock/structure'
models_dir = '/root/PLDock/test_data'
out_dir = '/root/PLDock/test_out'
equibind_test(inference_path=data_dir,output_directory=out_dir,test_names=data_dir+'/test.txt',run_dir =models_dir)
```
#### Predict the results of a batch of new data
```python
from PLDock.models.Equibind.inference import equibind_test
# new data dir
new_data = '/root/PLDock/newdata/structure'
out_dir = '/root/PLDock/newdata/out'
out_dir = '/root/PLDock/newdata/models'
equibind_test(inference_path=data_dir,output_directory=out_dir,run_dir=models_dir)
```
### TankBind
#### predict binding poses and affinities
```python
from PLDock.models.Tankbind import predictor
# A list of pdbs
test_pdbs = ['3d4k']
# data directory
test_dir = '/root/PLDock/test'
# result directory
out_dir = os.path.join(test_dir, 'out')
# pdb structure directory
structure_dir = os.path.join(test_dir, 'structure')
# model file
modelFile = '/root/PLDock/saved_models/self_dock.pt'
# p2rank executable file
p2rank_path = '/root/PLDock/p2rank/prank'

batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
screening.run_precdict(test_pdbs, out_dir, structure_dir, modelFile,p2rank_path, batch_size, device)
```

####  high-throughput virtual screening
```python
from PLDock.models.Tankbind import screening
# A list of pdbs
test_pdbs = ['6dlo']
# a file of a list of compounds
compound_list = "/root/PLDock/HTVS/Mcule_10000.csv"
# data directory
base_pre = f"/root/PLDock/HTVS"
# result directory
out_dir = os.path.join(base_pre, 'out')
# pdb structure directory
structure_dir = os.path.join(base_pre, 'structure')
# model file
modelFile = '/root/PLDock/saved_models/self_dock.pt'

# p2rank executable file
p2rank_path = '/root/PLDock/p2rank/prank'

batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

screening.run_screen(test_pdbs, modelFile, base_pre, outdir, p2rank_path, compound_list, batch_size, device)
```
## Information contain author names such as LICENSE will be released after review.

