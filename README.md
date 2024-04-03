# Setup python environment

```bash
conda env create -f conda-env.yaml
conda activate coresi
```

# Run

```bash
python3 -m coresi.main
# OR
coresi
```

# Display an image

```bash
python -m coresi.display_image --help
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
coresi --help
usage: main.py [-h] [-v] [-c CONFIG] [--sensitivity]

CORESI

options:
  -h, --help            show this help message and exit
  -v, --verbose         Enable debug output
  -c CONFIG, --config CONFIG
                        Path to the configuration file
  --sensitivity         Compute the sensitivity and quits
  --display             Display the reconstructed image
```

See the `config.yaml` file for configuring CORESI.

# Packaging

```bash
python3 -m build
```

This creates an installable wheel in the `dist/` folder.
