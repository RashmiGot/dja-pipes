# dja-pipes
Runs BAGPIPES on any JWST/NIRSpec PRISM spectrum from the DAWN JWST Archive (DJA)

Installation
============

Recommended install into a fresh conda environment:
```bash
conda create -n djapipes-python313 python=3.13 -y
conda activate djapipes-python313
```

Install `djapipes` and additional dependencies from git respositories:

```bash
pip install djapipes git+https://github.com/karllark/dust_attenuation.git 
```

Install fork of the ``bagpipes`` repo:
```bash
pip install git+https://github.com/RashmiGot/bagpipes.git#egg=bagpipes
```

There are occasionally problems installing some of the downstream dependencies in the latest Python versions, e.g., where the above returns an exception `ERROR: ... photutils-1.12.0.tar.gz does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.` This can be resolved with the workaround below:

```bash
wget https://files.pythonhosted.org/packages/10/b6/2ecd1ddebf269aa78103959a99ebb2c2ca9070f392cf10ac767fc4176b2a/photutils-1.12.0.tar.gz -O /tmp/photutils-1.12.0.tar.gz
tar xzvf /tmp/photutils-1.12.0.tar.gz -C /tmp/
pip install djapipes git+https://github.com/karllark/dust_attenuation.git  /tmp/photutils-1.12.0.
```

See also
--------
DJA: https://dawn-cph.github.io/dja/ <br>
BAGPIPES: https://bagpipes.readthedocs.io/en/latest/
