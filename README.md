
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

> De Angelis P., Trezza G., Barletta G., Asinari P., Chiavazzo E. "Energy-GNoME: A Living Database of Selected Materials for Energy Applications". arXiv, November 15, 2024. doi: [10.48550/arXiv.2411.10125](https://doi.org/10.48550/arXiv.2411.10125).


## How to cite

If you find this project valuable, please consider citing the following pre-print work:

> De Angelis P., Trezza G., Barletta G., Asinari P., Chiavazzo E. "Energy-GNoME: A Living Database of Selected Materials for Energy Applications". *arXiv* November 15, **2024**. doi: [10.48550/arXiv.2411.10125](https://doi.org/10.48550/arXiv.2411.10125).


```bibtex

@misc{deangelis_energy-gnome:_2024,
	title = {Energy-{GNoME}: {A} {Living} {Database} of {Selected} {Materials} for {Energy} {Applications}},
	shorttitle = {Energy-{GNoME}},
	url = {http://arxiv.org/abs/2411.10125},
	doi = {10.48550/arXiv.2411.10125},
	abstract = {Artificial Intelligence (AI) in materials science is driving significant advancements in the discovery of advanced materials for energy applications. The recent GNoME protocol identifies over 380,000 novel stable crystals. From this, we identify over 33,000 materials with potential as energy materials forming the Energy-GNoME database. Leveraging Machine Learning (ML) and Deep Learning (DL) tools, our protocol mitigates cross-domain data bias using feature spaces to identify potential candidates for thermoelectric materials, novel battery cathodes, and novel perovskites. Classifiers with both structural and compositional features identify domains of applicability, where we expect enhanced accuracy of the regressors. Such regressors are trained to predict key materials properties like, thermoelectric figure of merit (zT), band gap (Eg), and cathode voltage (\${\textbackslash}Delta V\_c\$). This method significantly narrows the pool of potential candidates, serving as an efficient guide for experimental and computational chemistry investigations and accelerating the discovery of materials suited for electricity generation, energy storage and conversion.},
	urldate = {2024-12-03},
	publisher = {arXiv},
	author = {De Angelis, Paolo and Trezza, Giovanni and Barletta, Giulio and Asinari, Pietro and Chiavazzo, Eliodoro},
	month = nov,
	year = {2024},
	note = {arXiv:2411.10125},
	keywords = {Condensed Matter - Materials Science, Condensed Matter - Other Condensed Matter, Computer Science - Machine Learning},
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
    - [ ] Data handlers Objects
    - [ ] Model handlers Objects
    - [ ] CLI `e-gnome`
- [ ] `jupyter` notebooks tutorials
    - [ ] Cathodes
    - [ ] Perovskites
    - [ ] Thermoelectrics

Detailed **TODO** list:
- [`energy-gnome` API](devtools/conda-envs/TODO.md#api)
- [`energy-gnom` CLI](devtools/conda-envs/TODO.md#cli)
- [`energy-gnom` notebooks](devtools/conda-envs/TODO.md#jupyter-notebooks)
- [`energy-gnom` documentation](devtools/conda-envs/TODO.md#doc)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
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
└── energy_gnome   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes energy_gnome a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

### [Contributing](.github/CONTRIBUTING.md)

We still working on the testing and of the API to allow easy continue integrtion of new models and databaset to allow the project to be up-todate with computetional materials state-of-the-art

Thus the contribution at the moment are limited, you can fork the project and try to add your model/database. First check the contribution [istructions](.github/CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

If you want to test our protocol with your new material descriptors and/or ML model (both for the regression or classification), pleas considete to write us an email to [paolo.deangelis@polito.it](mailto:paolo.deangelis@polito.it) for help and support in this integration or in general if you want collaborate on possible future reserch using our protocol/database.


### [Contributing](.github/CONTRIBUTING.md)

⚠️ **Work in progress**
We are still working on testing and refining the API to allow easy, continuous integration of new models and datasets, keeping the project up to date with state-of-the-art computational materials research.


#### How to help right now

1. Review the [contribution instructions](.github/CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
2. **Fork the repository** and create a feature branch.
3. Implement and test (⚠️ test suite not implemented yet) your integration.
4. Submit your pull request

#### Larger integrations

If you'd like to add **new material descriptors** or **ML/NN models** (regression or classification), open an issue first or email us at [paolo.deangelis@polito.it](mailto:paolo.deangelis@polito.it).
We're happy to guide you through the integration and explore future collaborations.


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
