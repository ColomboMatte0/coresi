<!--
SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>

SPDX-License-Identifier: MIT
-->

# Installation

To install as a system library and executable "coresi"

```bash
pip install .
# or with the whell
pip install dist/coresi-0.0.X-py3-none-any.whl
```

# Setup python environment

```bash
conda env create -f conda-env.yaml
conda activate coresi
```

CORESI is known to work with pytorch version 2.4.1.

# Run

```bash
python3 -m coresi.main
# OR if coresi is installed
coresi
```

CORESI can run on GPU or CPU.

Make use the environment variable `CUDA_VISIBLE_DEVICES` to tell which GPU
PyTorch should use.

The batch_scripts folder contains scripts that can be used to run CoReSi on the
CREATIS and [Jean
Zay](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html) clusters.


```bash
CUDA_VISIBLE_DEVICES="1" python -m coresi.main
```

Supported flags:

```bash
coresi --help
usage: main.py [-h] [-v] [-c CONFIG] [--sensitivity] [--simulation] [--display]

CORESI - Code for Compton camera image reconstruction (default action)

options:
  -h, --help            show this help message and exit
  -v, --verbose         Enable debug output (default: False)
  -c CONFIG, --config CONFIG
                        Path to the configuration file (default: config.yaml)
  --sensitivity         Compute the sensitivity and quits (default: False)
  --simulation          Do a simulation and quit (default: False)
  --display             Display the reconstructed image after the reconstruction (default: False)
```

See the `config.yaml` file for configuring CORESI.

# Display an image

```bash
python -m coresi.display_image --help
usage: display_image.py [-h] -f FILE [-c CONFIG] [-s SLICE] [-a AXIS] [--cpp | --no-cpp]

CORESI - image display

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  File path to the volume to display (default: None)
  -c CONFIG, --config CONFIG
                        Path to the configuration file (default: config.yaml)
  -s SLICE, --slice SLICE
                        Slice number (default: 0)
  -a AXIS, --axis AXIS  Axis x, y or z (default: z)
  --cpp, --no-cpp       Use this if the file comes from the C++ version of CORESI (default: None)
```

# Run tests

```bash
make tests
```


# Packaging

```bash
python3 -m build
```

This creates an installable wheel in the `dist/` folder.
