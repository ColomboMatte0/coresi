# Setup python environment

```bash
conda env create -f conda-env.yaml
conda activate coresi
```

# Run tests

```bash
make tests
```

CORESI can run on GPU or CPU.

Make use the environment variable `CUDA_VISIBLE_DEVICES` to tell which GPU
PyTorch should use.

```bash
CUDA_VISIBLE_DEVICES="1" python src/main.py
```

See the `config.yaml` file for configuring CORESI.
