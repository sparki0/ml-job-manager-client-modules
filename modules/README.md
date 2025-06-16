# Machine learning modules

Machine learning modules for analyzing spectral spectra.

# Overview

Contains following modules:

- **Active learning** - trains and applies CNN for spectra classfication.
- **Preprocessing** - transforms and scales spectra.
- **Dimensionality reduction** - visualize high-dimensional data in 2D using t-SNE.

This modules works with LAMOST DR2 spectra, if you want to use another spectra from other sources. You need to update preprocessing module, namely reading raw data from the file.

In active learning module CNN developed by Ing. Ondřej Podsztavek is used.

- [CNN source code](https://github.com/podondra/active-cnn).
- [Article](https://www.aanda.org/articles/aa/abs/2020/11/aa36090-19/aa36090-19.html).

# Tech Stack

`Tensorflow  2.16.2 [CUDA]`

`Scikit-learn 1.6.1`

# Running

Run the following commands in terminal:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
env/bin/python /directory_name/file_name config_path result_directory
```
