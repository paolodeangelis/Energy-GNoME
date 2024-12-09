# About the Database

The Energy-GNoME database was developed to identify and predict materials suitable for energy applications, such as thermoelectrics, cathodes, and perovskites.
The process combines machine learning (ML) techniques with an iterative active learning approach, enabling continuous integration and refinement.

<div style="text-align: center;" markdown>
--8<-- "docs/assets/partial/dp_cardinality.md"
</div>
/// table-caption | <
    attrs: {id: tab_db_size}
Databases sizes[^1]
///

## Protocol overview

![Workflow](../assets/img/about_db/workflow_light.png#only-light)
![Workflow](../assets/img/about_db/workflow_dark.png#only-dark)
/// figure-caption
    attrs: {id: fig_protocol}
Protocol workflow
///

Here, we provide a brief overview of the protocol workflow illustrated in [Figure 1](#fig_protocol).

### 1. Defining the Energy Material Region

We hypothesize a high-dimensional feature space where an energy material region $E$ exists, containing materials suitable for specific energy applications. By leveraging existing datasets (e.g., [MP][3] database), we identify the intersection $M^E = M \cap E$, forming the initial training set for our models.

### 2. Two-Phase Workflow

The protocol comprises two phases:

- **Training Phase:** Train ML models to classify and predict material properties.
- **Prediction Phase:** Identifies promising materials within the GNoME database and predicts their properties.

#### Training Phase

- **Data Preparation:** The specialized energy database $M^E$ serves as the training set. Missing structural information leads to a conditional split:
  - **Structure Pipeline:** Graph-based representation, regressors use the E(3)NN models.
  - **Composition Pipeline:** Chemical descriptors-based representation, regressors use the GBDT models.
- **AI-Experts (Screening Models):** A committee of binary GBDT classifiers learns to identify materials similar to those in $M^E$ by delineating the boundary of $E$.

#### Prediction Phase

- **Screening:** Materials from the GNoME database ($G$) pass through the AI-experts to compute the likelihood of belonging to $E$. Crystals with $P(y \in M^E) > 0.5$ are retained for property prediction.
- **Regression:** Depending on the pipeline used, the materials are either converted to graphs (E(3)NNs) or descriptors (GBDTs) to predict their properties.
- **Energy-GNoME Database:** Candidates with predicted properties are stored for evaluation, refinement, and use by the community.

### 3. Iterative Active Learning

The protocol allows continuous improvement by integrating new experimental or computational data from the material science and engineering community.
This iterative cycle refines both the AI-expert classifiers and regressors, making Energy-GNoME a dynamic and *living* database.

## Learn More

For a deeper understanding of the protocol and the fundamental hypotheses behind it, we invite you to explore our detailed article:

--8<-- "docs/assets/partial/cite_article.md"
