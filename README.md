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

Supported flags:

```bash
python src/main.py --help
usage: main.py [-h] [-v] [-c CONFIG] [--sensitivity]

CORESI

options:
  -h, --help            show this help message and exit
  -v, --verbose         Enable debug output
  -c CONFIG, --config CONFIG
                        Path to the configuration file
  --sensitivity         Compute the sensitivity and quits
```

See the `config.yaml` file for configuring CORESI.
