import os
import json
import re
import warnings
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from ase.formula import Formula
from ase.io import read
from pymatgen.core import Structure

root_dir = Path(__file__).parent.parent.parent

amu2g = 1.66053906e-24   # amu to grams
A2cm3 = 1e-24            # Å³ to cm³

def get_database_dict():
    """
    Load your different databases into a dict of DataFrames.
    Adjust these paths to match your actual folder structure.
    """
    db_types = [
        "cathodes/insertion",
        "perovskites",
        "thermoelectrics",
    ]
    sub_db = [
        ["Li", "Al", "Ca", "Cs", "K", "Mg", "Na", "Rb", "Y"],
        ["mixed_models", "pure_models"],
        ["300K", "430K", "560K", "690K", "820K", "950K"],
    ]

    db_dict = {}
    for i, db_t_ in enumerate(db_types):
        for sub_db_ in sub_db[i]:
            path_db = root_dir / f"data/final/{db_t_}/{sub_db_}/candidates.json"
            if path_db.exists():
                key = f"{db_t_}/{sub_db_}"
                db_dict[key] = pd.read_json(path_db)
    return db_dict

def get_cif_path(mid, db_name):
    """
    Returns the absolute path to the CIF file for the given material,
    depending on which sub-database (db_name) it came from.
    """
    if "thermoelectrics" in db_name:
        cif_path = root_dir / f"data/final/thermoelectrics/cif/{mid}.CIF"
    else:
        cif_path = root_dir / f"data/final/{db_name}/cif/{mid}.CIF"
    if not cif_path.exists():
        raise FileNotFoundError(f"{cif_path} NOT FOUND!")
    return cif_path

def get_spacegroup_link(symbol: str, number: int) -> str:
    """
    Generates the URL for a space group based on its symbol and International number.

    Args:
        symbol (str): Space group symbol (e.g., "Pnma").
        number (int): International number of the space group.

    Returns:
        str: URL pointing to the corresponding space group.
    """
    # Pad the International number to 3 digits
    padded_number = f"{number:03d}"

    # Default variation
    variation = "az1"

    # Adjust the variation based on the suffix in the symbol
    if "/c" in symbol:
        # Monoclinic with c-glide plane
        variation = "ay1"
    elif "/n" in symbol:
        # Monoclinic with n-glide plane
        variation = "my1"
    elif "/m" in symbol:
        # Monoclinic with m-glide plane
        variation = "ay1"
    elif "/a" in symbol:
        # Special case for certain a-glide planes
        variation = "ay1"

    # Construct the URL
    url = f"http://img.chem.ucl.ac.uk/sgp/large/{padded_number}{variation}.htm"

    return url

def convert_spacegroup_to_html(symbol: str) -> str:
    """
    Converts a space group symbol into an HTML representation.
    - Subscripts for digits following '_' or appearing individually.
    - Overlines for negative numbers (e.g., "-1").

    Args:
        symbol (str): Space group symbol (e.g., "P2_1/c", "P-1").

    Returns:
        str: HTML formatted space group.
    """
    # Apply subscripts only to digits directly preceded by '_'
    symbol = re.sub(r"_(\d)", r"<sub>\1</sub>", symbol)

    # Apply overline to parts starting with '-' (negative signs or symmetry elements)
    symbol = re.sub(
        r"-([a-zA-Z0-9])", r'<span style="text-decoration: overline;">\1</span>', symbol
    )

    # Return the formatted symbol
    return symbol

def get_spacegroup(symbol: str, number: int):
    spc_link = get_spacegroup_link(symbol, number)
    symbol_htm = convert_spacegroup_to_html(symbol)
    return f'<a href="{spc_link}" target="_blank">{symbol_htm} ({number})</a>'

### Properties
def add_cathode_properties(data, wion):
    properties_dict = {
        "Formation energy": {
            "Value[^val]": [r"{:.3f} eV/atom".format(data["Formation Energy (eV/atom)"].values[0])],
            "Model": ["GNoME"]
        },
        "Average voltage": {
            "Value[^val]": [
                r"{:.3f} &#xb1; {:.3f} V".format(
                    data["Average Voltage (V)"].values[0],
                    data["AI-experts confidence (deviation) (-)"].values[0]
                )
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Max volume expansion": {
            "Value[^val]": [
                r"{:.3f} &#xb1; {:.3f} %".format(
                    data["Max Volume expansion (-)"].values[0] * 100.0,
                    data["Max Volume expansion (deviation) (-)"].values[0] * 100.0
                )
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Stability charge": {
            "Value[^val]": [
                r"{:.3f} &#xb1; {:.3f} eV/atom".format(
                    data["Stability charge (eV/atom)"].values[0],
                    data["Stability charge (deviation) (eV/atom)"].values[0]
                )
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Stability discharge": {
            "Value[^val]": [
                r"{:.3f} &#xb1; {:.3f} eV/atom".format(
                    data["Stability discharge (eV/atom)"].values[0],
                    data["Stability discharge (deviation) (eV/atom)"].values[0]
                )
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Volumetric capacity": {
            "Value[^val]": [
                r"{:.3f} mAh/L".format(data["Volumetric capacity (mAh/L)"].values[0])
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Gravimetric capacity": {
            "Value[^val]": [
                r"{:.3f} mAh/g".format(data["Gravimetric capacity (mAh/g)"].values[0])
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Volumetric energy": {
            "Value[^val]": [
                r"{:.3f} Wh/L".format(data["Volumetric energy (Wh/L)"].values[0])
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        },
        "Gravimetric energy": {
            "Value[^val]": [
                r"{:.3f} Wh/kg".format(data["Gravimetric energy (Wh/kg)"].values[0])
            ],
            "Model": [f"E(3)NN ({wion}-cathode)"]
        }
    }

    # Handling the Note
    note = data["Note"].values[0]
    if len(note) > 0:
        foot = f"[^c{wion}]"
        note = f"[^c{wion}]: " + note.replace("R² < 0.5", r"R<sup>2</sup> < 0.5").replace(
            "AUC < 0.5", r"AUC < 0.5"
        )
    else:
        foot = ""

    return properties_dict, note

def add_perovskites_properties(data, model):
    """
    Collects predicted properties for perovskites and structures them as a dictionary of lists.
    """
    properties = {
        "Formation energy": {
            "Value[^val]": ["{:.3f} eV/atom".format(data["Formation Energy (eV/atom)"].values[0])],
            "Model": ["GNoME"]
        },
        "Band Gap": {
            "Value[^val]": ["{:.3f} &#xb1; {:.3f} eV".format(
                data["Average Band Gap (eV)"].values[0],
                data["Average Band Gap (deviation) (eV)"].values[0]
            )],
            "Model": [f"E(3)NN ({model})"]
        }
    }

    # Check if there's a note and format it properly
    note = data["Note"].values[0]
    if len(note) > 0:
        foot = f"[^p{model}]"
        note = f"[^p{model}]: " + note.replace("R²", r"R<sup>2</sup>")
        properties["Band Gap"]["Model"][0] += foot  # Append footnote to model name

    return properties, note

def add_thermoelectric_properties(data, wT):
    properties_dict = {
        "Formation energy": {
            "Value[^val]": [r"{:.3f} eV/atom".format(data["Formation Energy (eV/atom)"].values[0])],
            "Model": ["GNoME"]
        },
        "zT": {
            "Value[^val]": [
                r"{:.3f} &#xb1; {:.3f}".format(
                    data["Average zT (-)"].values[0],
                    data["Average zT (deviation) (-)"].values[0]
                )
            ],
            "Model": [f"GBDT ({wT} K)"]
        }
    }

    # Handling the Note
    note = data["Note"].values[0]
    if len(note) > 0:
        foot = f"[^t{wT}]"
        note = f"[^t{wT}]: " + note.replace("R²", r"R<sup>2</sup>")
    else:
        foot = ""

    return properties_dict, note

def make_cathode_info(wion, ai_experts_m, ai_experts_dev):
    return {
        "category": f"{wion}-cathode",
        "text": rf"""
    <div class="admonition info"><p class="admonition-title"> Possible {wion}-cathode</p><p>
        The material was identified by <a href="../../about_db/"><abbr title="Artificial Intelligence">AI</abbr> experts</a>
        as a potential cathode material for {wion}-ion batteries,
        with a probability of {ai_experts_m*100.0:.2f} &#xb1; {ai_experts_dev*100.0:.2f} %.
        <sup id="fnref2:val"><a class="footnote-ref" href="#fn:val">1</a></sup></p></div>
    """
    }

def make_perovskite_info(ai_experts_m_dict, ai_experts_dev_dict):
    model = list(ai_experts_m_dict.keys())[0]
    return {
        "category": f"perovskite",
        "text": rf"""
<div class="admonition info"><p class="admonition-title"> Possible perovskite</p><p>
    The material was identified by <a href="../../about_db/"><abbr title="Artificial Intelligence">AI</abbr> experts</a>
    as a potential perovskite material with a probability of {ai_experts_m_dict[model]*100.0:.2f} &#xb1; {ai_experts_dev_dict[model]*100.0:.2f} %.
    <sup id="fnref2:val"><a class="footnote-ref" href="#fn:val">1</a></sup></p></div>
"""
    }

def make_thermoelectric_info(ai_experts_m_dict, ai_experts_dev_dict):
    if len(ai_experts_m_dict) == 1:
        wT = list(ai_experts_m_dict.keys())[0]
        return {
        "category": f"thermoelectric",
        "text": rf"""
<div class="admonition info"><p class="admonition-title"> Possible thermoelectric</p><p>
    The material was identified by <a href="../../about_db/"><abbr title="Artificial Intelligence">AI</abbr> experts</a>
    as a potential thermoelectric material for the working temperature {wT} K, with a probability of {ai_experts_m_dict[wT]*100.0:.2f} &#xb1; {ai_experts_dev_dict[wT]*100.0:.2f} %.
    <sup id="fnref2:val"><a class="footnote-ref" href="#fn:val">1</a></sup></p></div>
"""
    }
    else:
        wT = list(ai_experts_m_dict.keys())
        temp_list = ", ".join([f"{t} K" for t in wT[:-1]])
        temp_list = temp_list + f" and {wT[-1]} K"
        info = {
        "category": f"thermoelectric",
        "text": rf"""
<div class="admonition info"><p class="admonition-title"> Possible thermoelectric</p><p>
    The material was identified by <a href="../../about_db/"><abbr title="Artificial Intelligence">AI</abbr> experts</a>
    as a potential thermoelectric material for working temperatures equal to {temp_list}, with a probability of """}
        info_ = []
        for wT_ in ai_experts_m_dict.keys():
            info_.append(
                rf"{ai_experts_m_dict[wT_]*100.0:.2f} &#xb1; {ai_experts_dev_dict[wT_]*100.0:.2f} % @ {wT_} K"
            )
        info["text"] += ", ".join(info_[:-1])
        info["text"] = info["text"] + f" and {info_[-1]}."
        info["text"] += rf"""<sup id="fnref2:val"><a class="footnote-ref" href="#fn:val">1</a></sup></p></div>"""

    return info

def main():
    db_dict = get_database_dict()

    # This dict will hold ALL material data:
    # materials_data[m_id] = {...info about that material...}
    materials_data = {}

    # 1) Build a map from m_id -> { "in_db": [db_key1, db_key2, ...], "cif": Path(...) }
    #    so we know where each material is located
    index_map = {}
    for dp_name_, df in db_dict.items():
        print("Reading material DB:", dp_name_)
        for m_id in tqdm(df["Material Id"].values, desc="materials analyzed"):
            # if m_id not in ["35ca0bcad2", "092187976b", "4ed36f8003", "ac2022f2d0", "bd64810b56", "40734d0790"]:
            #     continue
            if m_id not in index_map:
                index_map[m_id] = {"in_db": [], "cif": None}
            index_map[m_id]["in_db"].append(dp_name_)

    # 2) For each material, pick the first DB to find the CIF path
    #    (some materials appear in multiple DBs, but we only need one actual CIF)
    for m_id, info in tqdm(index_map.items(), desc="Building materials data"):
        # Use the first occurrence to get the CIF file location:
        cif_path = get_cif_path(m_id, info["in_db"][0])
        index_map[m_id]["cif"] = cif_path

        # Convert that CIF to various properties
        # (Mimicking your logic from the script)
        atoms = read(cif_path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            struct = Structure.from_file(cif_path)

        # Basic structural data
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        mass_amu = atoms.get_masses().sum()
        volume_A3 = atoms.get_volume()
        rho = (mass_amu * amu2g) / (volume_A3 * A2cm3)

        formula = Formula(atoms.get_chemical_formula())
        chemf_html = formula.reduce()[0].format("html")
        # Pymatgen space group
        sg_symbol, sg_number = struct.get_space_group_info()
        spacegroup_string = get_spacegroup(sg_symbol, sg_number)

        chemical_system = struct.chemical_system
        num_sites = struct.num_sites

        # Raw GitHub link (adjust as needed)
        cif_path_remote = (
            "https://raw.githubusercontent.com/"
            "paolodeangelis/Energy-GNoME/refs/heads/main/"
            + str(cif_path.relative_to(root_dir)).replace(os.sep, "/")
        )

        # Minimal metadata for each material; you can store as much or as little as you wish
        materials_data[m_id] = {
            "material_id": m_id,
            "chemical_formula_html": chemf_html,
            "cif_url": cif_path_remote,
            "cell": {
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma
            },
            "density_gcm3": rho,
            "space_group": spacegroup_string,
            "chemical_system": chemical_system,
            "num_sites": num_sites,
            "found_in_dbs": info["in_db"],  # e.g. ['cathodes/insertion/Li', 'perovskites/mixed_models']
            "predicted_properties": {}       # We’ll fill this next
        }

    # 3) Add properties from each DB entry
    ai_experts_m_perv = {}
    ai_experts_dev_perv = {}
    ai_experts_m_thermo = {}
    ai_experts_dev_thermo = {}
    for db_key, df in db_dict.items():
        for _, row in df.iterrows():
            m_id = row["Material Id"]

            ### TODO: REMOVE:
            try:
                materials_data[m_id]
            except KeyError:
                continue

            # Initialize predicted properties for this material
            if "predicted_properties" not in materials_data[m_id]:
                materials_data[m_id]["predicted_properties"] = {}

            # Add cathode properties if in cathode DB
            if "cathodes" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                wion = db_key.split("/")[-1]
                model = db_key.split("/")[-1].replace("_models", "")
                ai_experts_m = data["AI-experts confidence (-)"].values[0]
                ai_experts_dev = data["AI-experts confidence (deviation) (-)"].values[0]
                property_dict, note = add_cathode_properties(data, model)

                # If the material already has cathodes predictions, append new models
                if "cathode" in materials_data[m_id]["predicted_properties"]:
                    existing_props = materials_data[m_id]["predicted_properties"]["cathode"]["properties"]

                    for prop, values in property_dict.items():
                        if prop in existing_props:
                            existing_props[prop]["Value[^val]"].extend(values["Value[^val]"])
                            existing_props[prop]["Model"].extend(values["Model"])
                        else:
                            existing_props[prop] = values  # First entry

                else:
                    materials_data[m_id]["predicted_properties"]["cathode"] = {
                        "properties": property_dict,
                        "note": note
                    }

                # Ensure infos are stored properly under the category
                if "infos" not in materials_data[m_id]:
                    materials_data[m_id]["infos"] = []

                # Check if the category is already present
                existing_categories = {info["category"] for info in materials_data[m_id]["infos"]}

                cathode_info = make_cathode_info(wion, ai_experts_m, ai_experts_dev)
                if cathode_info["category"] not in existing_categories:
                    materials_data[m_id]["infos"].append(cathode_info)

                materials_data[m_id]["predicted_properties"]["cathode"]["properties"] = {
                    key: value for key, value in sorted(
                        materials_data[m_id]["predicted_properties"]["cathode"]["properties"].items())}


            # Add perovskite properties if in perovskites DB
            elif "perovskites" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                model = db_key.split("/")[-1].replace("_models", "")
                ai_experts_m_perv[model] = data["AI-experts confidence (-)"].values[0]
                ai_experts_dev_perv[model] = data["AI-experts confidence (deviation) (-)"].values[0]
                property_dict, note = add_perovskites_properties(data, model)

                # If the material already has perovskite predictions, append new models
                if "perovskite" in materials_data[m_id]["predicted_properties"]:
                    existing_props = materials_data[m_id]["predicted_properties"]["perovskite"]["properties"]

                    for prop, values in property_dict.items():
                        if prop in existing_props:
                            existing_props[prop]["Value[^val]"].extend(values["Value[^val]"])
                            existing_props[prop]["Model"].extend(values["Model"])
                        else:
                            existing_props[prop] = values  # First entry

                else:
                    materials_data[m_id]["predicted_properties"]["perovskite"] = {
                        "properties": property_dict,
                        "note": note
                    }

                # Ensure infos are stored properly under the category
                if "infos" not in materials_data[m_id]:
                    materials_data[m_id]["infos"] = []

                # Check if the category is already present
                existing_categories = {info["category"] for info in materials_data[m_id]["infos"]}

                perov_info = make_perovskite_info(ai_experts_m_perv, ai_experts_dev_perv)
                if perov_info["category"] not in existing_categories:
                    materials_data[m_id]["infos"].append(perov_info)

                materials_data[m_id]["predicted_properties"]["perovskite"]["properties"] = {
                    key: value for key, value in sorted(
                        materials_data[m_id]["predicted_properties"]["perovskite"]["properties"].items())}

            # Add thermoelectric properties if in thermoelectrics DB
            elif "thermoelectrics" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                wT = db_key.split("/")[-1].replace("K", "")
                ai_experts_m_thermo[wT] = data["AI-experts confidence (-)"].values[0]
                ai_experts_dev_thermo[wT] = data["AI-experts confidence (deviation) (-)"].values[0]
                property_dict, note = add_thermoelectric_properties(data, wT)

                # If the material already has thermoelectric predictions, append new models
                if "thermoelectric" in materials_data[m_id]["predicted_properties"]:
                    existing_props = materials_data[m_id]["predicted_properties"]["thermoelectric"]["properties"]

                    for prop, values in property_dict.items():
                        if prop in existing_props:
                            existing_props[prop]["Value[^val]"].extend(values["Value[^val]"])
                            existing_props[prop]["Model"].extend(values["Model"])
                        else:
                            existing_props[prop] = values  # First entry

                else:
                    materials_data[m_id]["predicted_properties"]["thermoelectric"] = {
                        "properties": property_dict,
                        "note": note
                    }

                # Ensure infos are stored properly under the category
                if "infos" not in materials_data[m_id]:
                    materials_data[m_id]["infos"] = []

                thermo_info = make_thermoelectric_info(ai_experts_m_thermo, ai_experts_dev_thermo)
                materials_data[m_id]["infos"] = [thermo_info]

                materials_data[m_id]["predicted_properties"]["thermoelectric"]["properties"] = {
                    key: value for key, value in sorted(
                        materials_data[m_id]["predicted_properties"]["thermoelectric"]["properties"].items())}


    # 4) Write out the single JSON file
    output_dir = root_dir / "docs" / "materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "materials.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(materials_data, f, indent=2)

    print(f"Saved all materials data to {output_json}")

if __name__ == "__main__":
    main()
