# Setup python environment

```bash
conda env create -f conda-env.yaml
conda activate coresi
```

# Run tests

```bash
make tests
```

CORESI can run on GPU or CPU by using either cupy or numpy. If not requested
explicitly, numpy is loaded and thus the CPU is used.

Set the environment variable ARRAY_MODULE to "cupy" to run on gpu. For example: 

```bash
ARRAY_MODULE="cupy" python src/main.py

# OR

export ARRAY_MODULE="cupy"
python src/main.py
```
