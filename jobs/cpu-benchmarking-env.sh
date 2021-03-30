echo "##vso[task.prependpath]$CONDA/bin"

conda env remove --yes -n tomopy-bench
rm -rf tomopy.github.io

conda env create --force -n tomopy-bench -f tomopy-bench.yaml 

conda list -n tomopy-bench

source activate tomopy-bench
git clone https://github.com/tomopy/tomopy
cd tomopy
python setup.py install
pytest

cd ..
git clone https://github.com/tomopy/tomopy.github.io

cd tomopy.github.io
TOMOPY_VERSION=`python -c 'import tomopy; print(tomopy.__version__)'`
BRANCH_NAME=$TOMOPY_VERSION"_CPU"
git checkout -B $BRANCH_NAME

cd ..
bash jobs/run-benchmarking-cpu-$1.sh 

git config --global user.name "Dreycen Foiles"
git config --global user.email "foilesdreycen@gmail.com"
git add *
git commit -m "Bi-monthly benchmark"
git push -uf https://$(PAT)@github.com/tomopy/tomopy.github.io $BRANCH_NAME

cd ..
conda deactivate
conda env remove --yes -n tomopy-bench
rm -rf tomopy.github.io
rm -rf tomopy