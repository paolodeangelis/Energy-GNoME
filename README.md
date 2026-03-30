
<h1 align="center"> Energy-GNoME </h1>
<p align="center">
  <a href="https://paolodeangelis.github.io/Energy-GNoME">
    <img src="https://raw.githubusercontent.com/paolodeangelis/Energy-GNoME/main/docs/assets/img/logo.png" width="250" alt="Energy-GNoMs">
  </a>
</p>

<p align="center">
  <strong>
    AI-Driven Screening and Prediction for Selected <a href="https://paolodeangelis.github.io/Energy-GNoME">Advanced Energy Materials</a>
  </strong>
</p>

<p align="center">
<a target="_blank" href="hhttps://www.nature.com/articles/sdata201618">
    <img src="https://custom-icon-badges.demolab.com/badge/data-FAIR-blue?logo=database\&logoColor=white" />
</a>
<a target="_blank" href="https://python.org">
    <img src="https://custom-icon-badges.demolab.com/badge/Python-3.10+-blue?logo=python\&logoColor=white" />
</a>
<a target="_blank" href="https://www.linux.org/">
    <img src="https://custom-icon-badges.demolab.com/badge/OS-Linux-orange?logo=linux\&logoColor=white" />
</a>
<a target="_blank" href=".github/CONTRIBUTING.md">
    <img src="https://custom-icon-badges.demolab.com/badge/contributions-open-color=4cb849?logo=code-of-conduct\&logoColor=white" />
</a>
</p>
<p align="center">
<a target="_blank" href="LICENSE">
    <img src="https://custom-icon-badges.demolab.com/badge/license-CC--BY%204.0-lightgray?logo=law\&logoColor=white" />
</a>
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a target="_blank" href="https://github.com/psf/black">
    <img src="https://custom-icon-badges.demolab.com/badge/code%20style-black-000000?logo=code\&logoColor=white" />
</a>
</p>
<p align="center">
<a target="_blank" href="https://github.com/paolodeangelis/Energy-GNoME/actions/workflows/deploy.yaml">
    <img src="https://results.pre-commit.ci/badge/github/paolodeangelis/Energy-GNoME/main.svg" />
</a>
<a target="_blank" href="https://github.com/paolodeangelis/Energy-GNoME/actions/workflows/deploy.yaml">
    <img src="https://github.com/paolodeangelis/Energy-GNoME/actions/workflows/deploy.yaml/badge.svg?branch=main" />
</a>
</p>
</p>
<p align="center">
<a target="_blank" href="https://doi.org/10.5281/zenodo.14338533"><img src="https://zenodo.org/badge/858064778.svg" alt="DOI"></a>
</p>

This repository contains the database, documentation, Python library (coming soon), and notebooks used to build the Energy-GNoME database.

The purpose of this repository is to enable reproducibility and, more importantly, to support the continuous integration of your data points for model training, as the database is designed as a *living* database.

For further details, refer to the associated article:

> De Angelis P., Barletta G., Trezza G., Asinari P., Chiavazzo E. "Energy-GNoME: A living database of selected materials for energy applications". *Energy and AI* 22, 100605, **2025**. doi: [10.1016/j.egyai.2025.100605](https://doi.org/10.1016/j.egyai.2025.100605).


## How to cite

If you find this project valuable, please consider citing the following published work:

> De Angelis P., Barletta G., Trezza G., Asinari P., Chiavazzo E. "Energy-GNoME: A living database of selected materials for energy applications". *Energy and AI* 22, 100605, **2025**. doi: [10.1016/j.egyai.2025.100605](https://doi.org/10.1016/j.egyai.2025.100605).


```bibtex

@article{DEANGELIS2025100605,
title = {Energy-GNoME: A living database of selected materials for energy applications},
journal = {Energy and AI},
volume = {22},
pages = {100605},
year = {2025},
issn = {2666-5468},
doi = {https://doi.org/10.1016/j.egyai.2025.100605},
url = {https://www.sciencedirect.com/science/article/pii/S2666546825001375},
author = {Paolo {De Angelis} and Giulio Barletta and Giovanni Trezza and Pietro Asinari and Eliodoro Chiavazzo},
keywords = {Energy materials, Artificial Intelligence, Machine Learning, Deep Learning, Thermoelectric, Battery, Perovskite},
abstract = {Artificial Intelligence (AI) in materials science is driving significant advancements in the discovery of advanced materials for energy applications. The recent GNoME protocol identifies over 380,000 novel stable crystals. From this, we identify over 38,500 materials with potential as energy materials forming the core of the Energy-GNoME database. Our unique combination of Machine Learning (ML) and Deep Learning (DL) tools mitigates cross-domain data bias using feature spaces, thus identifying potential candidates for thermoelectric materials, novel battery cathodes, and novel perovskites. First, classifiers with both structural and compositional features detect domains of applicability, where we expect enhanced reliability of regressors. Here, regressors are trained to predict key materials properties, like thermoelectric figure of merit (zT), band gap (Eg), and cathode voltage (ΔVc). This method significantly narrows the pool of potential candidates, serving as an efficient guide for experimental and computational chemistry investigations and accelerating the discovery of materials suited for electricity generation, energy storage and conversion.}
}

```

Additional articles to cite:

- **GNoME Database:** Additionally, please consider citing the foundational GNoME database work:

    > Merchant, A., Batzner, S., Schoenholz, S.S. *et al.* "Scaling deep learning for materials discovery". *Nature* 624, 80-85, **2023**. doi: [10.1038/s41586-023-06735-9](https://doi.org/10.1038/s41586-023-06735-9).

- **E(3)NN Model**: And the E(3)NN Graph Neural Network model

    > Chen Z., Andrejevic N., Smidt T. *et al.* " Direct Prediction of Phonon Density of States With Euclidean Neural Networks." *Advanced Science* 8 (12), 2004214, **2021**. [10.1002/advs.202004214](https://doi.org/10.1002/advs.202004214)

## Project Status

- [x] Databases:
    - [X] Cathodes
    - [x] Perovskites
    - [x] Thermoelectrics
- [x] Dashboards
    - [x] Cathodes
    - [x] Perovskites
    - [x] Thermoelectrics
- [ ] `energy-gnome` python library
    - [X] Data handlers Objects
    - [X] Model handlers Objects
    - [ ] CLI `e-gnome`
- [X] `jupyter` notebooks tutorials
    - [X] Perovskites
    - [X] Thermoelectrics
    - [X] Materials Project
    - [X] GNoME
    - [X] E(3)NN Regressor
    - [X] GBDT Regressor
    - [X] GBDT Classifier

Detailed **TODO** list:
- [`energy-gnome` API](devtools/conda-envs/TODO.md#api)
- [`energy-gnome` CLI](devtools/conda-envs/TODO.md#cli)
- [`energy-gnome` documentation](devtools/conda-envs/TODO.md#doc)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── final          <- The screened candidates, along with predictions on their properties of interest.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks demonstrating example usage of the `energy-gnome`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         energy_gnome and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── energy_gnome       <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes energy_gnome a Python module
    │
    ├── config.py      <- Store useful variables and configuration
    │
    ├── dataset        <- Scripts to handle data and features for modeling
    │
    ├── models         <- Scripts to handle ML models
```

--------

### [Contributing](.github/CONTRIBUTING.md)

⚠️ Work in Progress

We are actively working on improving testing and refining the API to support the seamless integration of new models and datasets. Our goal is to keep the project aligned with the latest advancements in computational materials science.

How You Can Contribute
While the contribution process is still under development, you’re welcome to get involved by:

Reviewing the contribution [guidelines](.github/CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

Forking the repository and creating a feature branch.

Adding your model or dataset (note: the test suite is still under construction).

Submitting a pull request for review.

For Larger Contributions
If you're interested in integrating new material descriptors or machine learning models (for regression or classification), we recommend:

Opening an issue to discuss your proposal, or

Contacting us directly at [paolo.deangelis@polito.it](mailto:paolo.deangelis@polito.it) for guidance and support.

We are happy to assist with integration and discuss potential research collaborations using the protocol or database.


<hr width="100%">
<div style="display: flex; justify-content: space-between; align-items: center;">
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0; height:35px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>
   <span style="float:right;">
    &nbsp;
    <a rel="small" href="https://areeweb.polito.it/ricerca/small/">
        <img style="border-width:0; height:35px" src="assets/img/logo-small.png" alt="SMALL site" >
    </a>
    &nbsp;
    <a rel="polito"href="https://www.polito.it/">
        <img style="border-width:0; height:35px" src="assets/img/logo-polito.png" alt="POLITO site" >
    </a>
</span>
</div>

<!-- [![CC BY 4.0][cc-by-image]][cc-by] -->

[cc-by]: http://creativecommons.org/licenses/by/4.0/

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[article-doi]: https://doi.org/10.1038/s41598-023-50978-5

[old-ff-doi]: https://doi.org/10.1021/acs.jpclett.7b00898

[enhancing-reaxFF-database-repository]: https://github.com/paolodeangelis/Enhancing_ReaxFF_DFT_database
