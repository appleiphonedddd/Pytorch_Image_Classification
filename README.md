## Contents

###### Requirements

- **Operating System**: Ubuntu 24.04.03 LTS (Linux-based)
- **GPU**: NVIDIA GeForce RTX 3060 (or higher, CUDA-enabled)
- **CUDA Toolkit**: 12.x (compatible with your GPU driver)

###### Installation

1. Install Conda (If you have already installed this command or Anaconda , you can skip this step!!!!)

```sh
./install_miniconda.sh
```

### Deployment

1. Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate IC
```

2. Generate the dataset based on the data distribution you personally want to test, for example MNIST

```sh
cd dataset/

python generate_MNIST.py

```

### Extend new algorithms and datasets

- **New Dataset**: To add a new dataset, simply create a `generate_DATA.py` file in `./dataset` and then write the download code and use the [utils](dataset/utils) as shown in `./dataset/generate_MNIST.py` (you can consider it as a template):
  ```python
  # `generate_DATA.py`
  import necessary pkgs
  from utils import necessary processing funcs

  def generate_dataset(...):
    # download dataset as usual
    # pre-process dataset as usual
    X, y, statistic = separate_data((dataset_content, dataset_label), ...)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, statistic, ...)

  # call the generate_dataset func
  ```

### Author

611221201@gms.ndhu.edu.tw

Egor Alekseyevich Morozov
