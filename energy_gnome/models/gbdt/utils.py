from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import pandas as pd


def featurizing_structure_pipeline(
    data: pd.DataFrame, index: str = "formula", property_interest: str = None
) -> pd.DataFrame:
    """
    Featurizes the given dataset for composition and structure features.

    Args:
        data (pd.DataFrame): Input dataframe containing 'composition' and 'structure' columns.
        index (str, optional): Database columns to use as dataframe index.
        property_interest (str, optional): The property of interest to include in the output dataframe.

    Returns:
        pd.DataFrame: A dataframe with the featurized composition and structure data.
    """
    # Composition Featurizer
    composition_featurizer = MultipleFeaturizer(
        [
            cf.Stoichiometry(),
            cf.ElementProperty.from_preset("magpie"),
            cf.ValenceOrbital(),
            cf.IonProperty(fast=True),
            cf.ElementFraction(),
        ]
    )

    # Structure Featurizer
    structure_featurizer = MultipleFeaturizer(
        [
            sf.JarvisCFID(
                use_cell=True,  # Use structure cell descriptors (4 features, based on DensityFeatures
                # and log volume per atom).
                use_chem=True,  # Use chemical composition descriptors (438 features)
                use_chg=False,  # Use core charge descriptors (378 features)
                use_adf=False,  # Use angular distribution function (179 features x 2, one set of
                # features for each cutoff).
                use_rdf=False,  # Use radial distribution function (100 features)
                use_ddf=False,  # Use dihedral angle distribution function (179 features)
                use_nn=False,  # Use nearest neighbors (100 descriptors)
            )
        ]
    )

    # Featurize composition data
    featurized_data_comp = pd.DataFrame(
        composition_featurizer.featurize_many(data["composition"], ignore_errors=True),
        columns=composition_featurizer.feature_labels(),
        index=data[index],
    )

    # Featurize structure data
    featurized_data_struct = pd.DataFrame(
        structure_featurizer.featurize_many(data["structure"], ignore_errors=True),
        columns=structure_featurizer.feature_labels(),
        index=data[index],
    )

    # Combine the featurized data
    featurized_data = pd.concat([featurized_data_comp, featurized_data_struct], axis=1)

    # Optionally include the property of interest
    if property_interest:
        featurized_data[property_interest] = data[property_interest].values

    return featurized_data
