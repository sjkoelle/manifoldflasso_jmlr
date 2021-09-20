# manifoldflasso_jmlr
This repository contains code for generation of figures in the Manifold Coordinates with Physical Meaning’ (Koelle, Zhang, Meila, Chen).
The major steps are 1) configuring the virtual environment and 2) running the code.

1) Configuring the virtual environment
This part can be done prior to cloning the manifoldflasso_jmlr repository, and does not depend on being run in the same directory.

1.1) Create a virtual env with megaman dependencies

conda create -n manifold_env_april python=3.5 -y

source activate manifold_env_april

conda install --channel=conda-forge -y pip nose coverage cython numpy scipy scikit-learn pyflann pyamg h5py plotly

1.2) Install megaman

mkdir megaman_jmlr

cd megaman_jmlr

mkdir tmp

cd tmp/

git clone https://github.com/mmp2/megaman.git

cd megaman

python setup.py install

cd ../..

1.3) Install other packages

conda install pytorch torchvision -c pytorch -y

pip install -U matplotlib

conda install dill

conda install --channel=conda-forge --yes pyglmnet

conda install --channel=conda-forge --yes python-spams

conda install --channel=conda-forge --yes pathos

conda install --channel=conda-forge --yes autograd

conda install seaborn

1.4) (Optional) configure jupiter notebook
conda install -c anaconda ipython
conda install -c anaconda ipykernel
ipython kernel install --user --name=manifold_env_april2 --display-name=manifold_env_april2

2) Running the code
The RigidEthanol and SwissRoll examples simulate data, and so can be run without adding other files.  The Toluene, Ethanol, and Malonaldehyde examples require data to be added to the untracked_data/chemistry_data folder.  Data is available at xxx.

2.1) If running on a cluster
Use the .sh files e.g. 
sbatch swissroll_041420_nsel_50_nreps5.sh

2.2) If running locally
source activate manifold_env_april
python swissroll_041420_nsel_50_nreps5.py





