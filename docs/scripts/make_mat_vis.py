import os
from pathlib import Path
import warnings

from ase.formula import Formula
from ase.io import read
import pandas as pd
from pymatgen.core import Structure
from tqdm.auto import tqdm
import yaml

TEMPLATE = """---
title: {title}
material_viewer: true
hide:
  - path
  - navigation
  - toc
mid: {id}
cif_path: {cif_path}
---

# {chemf} — `{id}`

<!--- <div class="container"> -->
<div class="grid cards" style="margin: 0 auto;">
    <!-- Viewer Section -->
    <ul class="flex-wrapper">
    <li class="viewer-li">
    <p>
        <strong>Viewer</strong>
    </p>
    <hr>
    <div class="viewer-section">
        <div id="container" style="height: 450px; width: 100%; position: relative;"></div>
        <div class="controls">
            <!-- <button onclick="viewer.spin(true)">Spin</button> -->
            <!-- <button onclick="viewer.spin(false)">Stop</button> -->
            <button class="controls-button" onclick="toggleLabels()">Show/Hide Atom Names</button>
            <button class="md-button md-button-primary" onclick="downloadCifFile('{cif_path}','{id}.cif')">
            Download CIF
            </button>
        </div>
    </div>
    </li>

    <!-- Side-by-side Table Section -->
    <li class="table-li">
    <p>
        <strong>Crystal properties</strong>
    </p>
    <hr>
    <div class="table-section", style="text-align: center;" markdown="1">
        <table>
            <tr>
                <th>\\( \\mathbf{{a}} \\)</td>
                <th>\\( {a:.2f} \\; \\text{{Å}} \\)</td>
            </tr>
            <tr>
                <td>\\( \\mathbf{{b}} \\)</td>
                <td>\\( {b:.2f} \\; \\text{{Å}} \\)</td>
            </tr>
            <tr>
                <td>\\( \\mathbf{{c}} \\)</td>
                <td>\\( {c:.2f} \\; \\text{{Å}} \\)</td>
            </tr>
            <tr>
                <td>\\( \\mathbf{{\\alpha}} \\)</td>
                <td>\\( {alpha:.1f}  ^\\circ \\)</td>
            </tr>
            <tr>
                <td>\\(  \\mathbf{{\\beta}} \\)</td>
                <td>\\( {beta:.1f}  ^\\circ \\)</td>
            </tr>
            <tr>
                <td>\\( \\mathbf{{\\gamma}} \\)</td>
                <td>\\( {gamma:.1f}  ^\\circ \\)</td>
            </tr>
            <tr>
                <td><b>Density</b></td>
                <td>\\( {rho:.2f} \\; \\; \\mathrm{{g/cm^3}} \\)</td>
            </tr>
            <tr>
                <td><b>Space group</b></td>
                <td><a href="http://img.chem.ucl.ac.uk/sgp/large/{int_sg:03d}az1.htm" target="_blank">\\( \\; \\mathrm{{ {sg} }} \\) ({int_sg})</a></td>
            </tr>
            <tr>
                <td><b>Chemical system</b></td>
                <td>{chem_sys}</td>
            </tr>
            <tr>
                <td><b>Number of sites</b></td>
                <td>{num_sites:d}</td>
            </tr>
        </table>
    </div>
    </li>
    </ul>
</div>

<script>
    let viewer;
    let atoms;
    let labelsVisible = false; // Track whether labels are currently shown

    // Initialize viewer
    $(document).ready(function() {{
        viewer = $3Dmol.createViewer("container");
        viewer.setBackgroundColor(0x1f2429, 0.0);

        // Load CIF file
        const cifUrl = "{cif_path}";
        jQuery.ajax(cifUrl, {{
            dataType: "text",
            success: function(data) {{
                atoms = viewer.addModel(data, "cif");
                // Show a ball-and-stick style by default
                viewer.setStyle({{}}, {{
                    stick:{{
                        radius:0.15,
                        // Use the Jmol color scheme:
                        colorscheme: "Jmol"
                        }},
                        sphere:{{
                        scale:0.33,
                        // Use the Jmol color scheme:
                        colorscheme: "Jmol"
                        }}
                    }});
                // Add the unit cell box (if CIF has cell info)
                viewer.addUnitCell(atoms,{{
                    box:{{
                        color:0x64c4d3
                        }}
                    }});
                viewer.zoomTo();
                viewer.render();
            }}
        }});
    }});

    function toggleLabels() {{
        const theButton = document.querySelector('.controls-button');
        if (!labelsVisible) {{
            // Show atom labels
            viewer.removeAllLabels();
            let selected = viewer.selectedAtoms({{}});
            for (let i = 0; i < selected.length; i++) {{
                let a = selected[i];
                viewer.addLabel(a.elem, {{
                    position: {{ x: a.x, y: a.y, z: a.z }},
                    backgroundColor: "white",
                    backgroundOpacity: 0.5,
                    fontColor: 0x1f2429,
                    fontSize: 14
                }});
            }}
             // Toggle "active" style
            theButton.classList.add('active-button');
            labelsVisible = true;
        }} else {{
            // Hide atom labels
            viewer.removeAllLabels();
            labelsVisible = false;
            // Remove "active" style
            theButton.classList.remove('active-button');
            labelsVisible = false;
        }}
        viewer.render();
    }}

    function downloadCifFile(fileUrl, fileName) {{
    fetch(fileUrl)
        .then(response => {{
        if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
        }}
        return response.blob();
        }})
        .then(blob => {{
        // Create a temporary link <a> element
        const tempUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = tempUrl;
        // Use the second parameter, so the file will download as `fileName`
        a.download = fileName;

        // Attach the <a> to the DOM
        document.body.appendChild(a);
        // Programmatically click it to start download
        a.click();

        // Cleanup: remove the link and revoke object URL
        document.body.removeChild(a);
        URL.revokeObjectURL(tempUrl);
        }})
        .catch(err => {{
        console.error('File download failed:', err);
        alert('Could not download file.');
        }});
    }}

</script>

<style>
    /* Turn the <ul> into a flex container */
    .flex-wrapper {{
        display: flex !important;
        gap: 20px;       /* space between items */
        padding: 0;      /* remove default <ul> padding/margins */
        margin: 0;
    }}

    /* Make the first element wider than the second */
    .viewer-li {{
        flex: 2;         /* 2 parts out of total 3, so ~66% of space */
        width: 60% !important;
        min-width: 600px !important;/* enforce a minimum width if you like */
    }}
    .table-li {{
        flex: 1;         /* 1 part out of total 3, so ~33% of space */
        min-width: 200px;
    }}
    /* Flex container for side-by-side layout */
    .container {{
        display: flex;
        flex-wrap: wrap; /* so that on narrow screens, they stack */
        gap: 20px;
        margin: 0 auto;
    }}
    .viewer-section {{
        flex: 1 1 400px;
        min-width: 500px;
    }}
    table {{
        border-collapse: collapse;
        border-spacing: 0;
        border:none!important;
    }}
    th {{
    font-weight: 400!important; /* or 400 */
    }}


    .table-section {{
        flex: 1 1 200px;
        min-width: 200px;
        border-collapse: collapse;
        margin: 0 auto;
        /* No outer border on the table itself */
        border: none;
    }}
    .table-section td, .table-section th {{
        min-width: 120px;  /* each column has at least 100px */
    }}

    /* Adjust overall container style */
    .viewer-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }}
    .controls {{
        margin-top: 10px;
    }}
    .controls button {{
        margin: 5px;
        padding: .625em 2em;
    }}
    .controls-button {{
    border: .1rem solid;
    border-radius: .1rem;
    color: var(--md-primary-fg-color);
    cursor: pointer;
    display: inline-block;
    font-weight: 700;
    transition: color 125ms, background-color 125ms, border-color 125ms;
    background-color: transparent; /* Default: no background */
    }}

    /* Hover effect: when the mouse is over the button */
    .controls-button:hover {{
    background-color: var(--md-primary-fg-color);
    border-color: var(--md-primary-fg-color);
    color: #fff;
    }}

    /* Toggle/Active class: when clicked, we apply this class via JS */
    .active-button {{
    background-color: var(--md-primary-fg-color);
    color: #fff;
    }}
</style>

"""


root_dir = Path(__file__).parent.parent.parent
amu2g = 1.66053906e-24  # amu to grams
A2cm3 = 1e-24  # Å³ to cm³


def load_meta_social(filepath):
    """
    Loads the existing meta_social.yaml file, if it exists.
    """
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_meta_social(filepath, meta_social):
    """
    Saves the updated meta_social dictionary to the YAML file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(meta_social, f, default_flow_style=False)


def get_database_dict():
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
            db_dict[f"{db_t_}/{sub_db_}"] = pd.read_json(path_db)
    return db_dict


def get_cif_path(mid, db_name):
    if "thermoelectrics" in db_name:
        cif_path = root_dir / f"data/final/thermoelectrics/cif/{mid}.CIF"
    else:
        cif_path = root_dir / f"data/final/{db_name}/cif/{mid}.CIF"
    if not cif_path.exists():
        raise FileNotFoundError(f"{cif_path} NOT FOUND!")
    return cif_path


def get_cif_dict(db_dict):
    cif_files = {}
    for dp_name_, db_ in db_dict.items():
        for m_id in tqdm(db_["Material Id"].values, desc=f"scanning db:{dp_name_}"):
            if m_id in cif_files:
                # print(f"{m_id} found in multple db ({cif_files[m_id]['in_db']} + {dp_name_})")
                cif_files[m_id]["in_db"] += [dp_name_]
            else:
                cif_files[m_id] = {"in_db": [dp_name_], "cif": get_cif_path(m_id, dp_name_)}
    return cif_files


def make_viewer_gen_info(material_id, cif_path):
    atoms = read(cif_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        struct = Structure.from_file(cif_path)
    cif_loc_path = cif_path.relative_to(root_dir)
    formula = Formula(atoms.get_chemical_formula())
    chemf_html = formula.reduce()[0].format("html")
    chemf_red = formula.reduce()[0].format("reduce")
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    rho = (atoms.get_masses().sum() * amu2g) / (atoms.get_volume() * A2cm3)
    sg, int_sg = struct.get_space_group_info()
    chem_sys = struct.chemical_system
    num_sites = struct.num_sites
    cif_path_remote = (
        "https://raw.githubusercontent.com/" "paolodeangelis/Energy-GNoME/refs/heads/main/"
    ) + str(cif_loc_path)

    page_content = TEMPLATE.format(
        title=chemf_red,
        id=material_id,
        chemf=chemf_html,
        cif_path=cif_path_remote,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        rho=rho,
        sg=sg,
        int_sg=int_sg,
        chem_sys=chem_sys,
        num_sites=num_sites,
    )
    return page_content, chemf_red


# Info
def make_cathode_info(wion, ai_experts_m, ai_experts_dev):
    info = rf"""
!!! info "Possible {wion}-cathode"

    The material was identified by [AI experts](../about_db/index.md) as a potential cathode material for {wion}-ion batteries, with a probability of \( {ai_experts_m*100.0:.2f} \pm {ai_experts_dev*100.0:.2f} \;\% \).[^val]
"""
    return info


def make_perovskite_info(ai_experts_m_dict, ai_experts_dev_dict):
    if len(ai_experts_m_dict) == 1:
        model = list(ai_experts_m_dict.keys())[0]
        info = rf"""
!!! info "Possible perovskite"

    The material was identified by [AI experts](../about_db/index.md) as a potential perovskite material for *{model}* regression model, with a probability of \( {ai_experts_m_dict[model]*100.0:.2f} \pm {ai_experts_dev_dict[model]*100.0:.2f} \;\% \).[^val]
"""
    else:
        info = """
!!! info "Possible perovskite"

    The material was identified by [AI experts](../about_db/index.md) as a potential perovskite material for both regression model, with a probability of """
        info_ = []
        for model in ai_experts_m_dict.keys():
            info_.append(
                rf"\( {ai_experts_m_dict[model]*100.0:.2f} \pm {ai_experts_dev_dict[model]*100.0:.2f} \;\% \) for *{model}* model"
            )
        info += " and ".join(info_)
        info += ".[^val]\n\n"
    return info


def make_thermoelectric_info(ai_experts_m_dict, ai_experts_dev_dict):
    if len(ai_experts_m_dict) == 1:
        wT = list(ai_experts_m_dict.keys())[0]
        info = rf"""
!!! info "Possible thermoelectric"

    The material was identified by [AI experts](../about_db/index.md) as a potential thermoelectric material for the working temperature {wT} K, with a probability of \( {ai_experts_m_dict[wT]*100.0:.2f} \pm {ai_experts_dev_dict[wT]*100.0:.2f} \;\%\).[^val]
"""
    else:
        wT = list(ai_experts_m_dict.keys())
        temp_list = ", ".join([f"{t} K" for t in wT[:-1]])
        temp_list = temp_list + f" and {wT[-1]} K"
        info = f"""
!!! info "Possible thermoelectric"

    The material was identified by [AI experts](../about_db/index.md) as a potential thermoelectric material for {temp_list} working temperatures, with a probability of :"""
        info_ = []
        for wT_ in ai_experts_m_dict.keys():
            info_.append(
                rf"\( {ai_experts_m_dict[wT_]*100.0:.2f} \pm {ai_experts_dev_dict[wT_]*100.0:.2f} \;\% \) for {wT_} K"
            )
        info += ", ".join(info_[:-1])
        info = info + f" and {info_[-1]}"
        info += ".[^val]\n\n"
    return info


### Properties
def add_cathode_properties(data, wion):
    df_property = pd.DataFrame(columns=["**Value**[^val]", "**Model**"])
    note = data["Note"].values[0]
    if len(note) > 0:
        foot = f"[^c{wion}]"
        note = f"[^c{wion}]: " + note.replace("R² < 0.5", r"\( R^2 < 0.5 \)").replace(
            "AUC < 0.5", r"\( AUC < 0.5 \)"
        )
    else:
        foot = ""
    df_property.loc["**Formation energy**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{eV/atom}} \)".format(data["Formation Energy (eV/atom)"].values[0])
    )
    df_property.loc["**Formation energy**", "**Model**"] = "GNoME"
    df_property.loc["**Average voltage**", "**Value**[^val]"] = (
        r"\( {:.3f} \;\pm\;{:.3f}  \; \mathrm{{V}} \)".format(
            data["Average Voltage (V)"].values[0],
            data["AI-experts confidence (deviation) (-)"].values[0],
        )
    )
    df_property.loc["**Average voltage**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Max volume expansion**", "**Value**[^val]"] = (
        r"\( {:.3f} \;\pm\;{:.3f}  \; \mathrm{{\%}} \)".format(
            data["Max Volume expansion (-)"].values[0] * 100.0,
            data["Max Volume expansion (deviation) (-)"].values[0] * 100.0,
        )
    )
    df_property.loc["**Max volume expansion**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Stability charge**", "**Value**[^val]"] = (
        r"\( {:.3f} \;\pm\;{:.3f}  \; \mathrm{{eV/atom}} \)".format(
            data["Stability charge (eV/atom)"].values[0],
            data["Stability charge (deviation) (eV/atom)"].values[0],
        )
    )
    df_property.loc["**Stability charge**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Stability discharge**", "**Value**[^val]"] = (
        r"\( {:.3f} \;\pm\;{:.3f}  \; \mathrm{{eV/atom}} \)".format(
            data["Stability discharge (eV/atom)"].values[0],
            data["Stability discharge (deviation) (eV/atom)"].values[0],
        )
    )
    df_property.loc["**Stability discharge**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Volumetric capacity**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{mAh/L}} \)".format(data["Volumetric capacity (mAh/L)"].values[0])
    )
    df_property.loc["**Volumetric capacity**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Gravimetric capacity**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{mAh/g}} \)".format(data["Gravimetric capacity (mAh/g)"].values[0])
    )
    df_property.loc["**Gravimetric capacity**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Volumetric energy**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{Wh/L}} \)".format(data["Volumetric energy (Wh/L)"].values[0])
    )
    df_property.loc["**Volumetric energy**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    df_property.loc["**Gravimetric energy**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{Wh/kg}} \)".format(data["Gravimetric energy (Wh/kg)"].values[0])
    )
    df_property.loc["**Gravimetric energy**", "**Model**"] = f"E(3)NN ({wion}-cathode)" + foot
    return df_property, note


def add_perovskites_properties(data, model):
    df_property = pd.DataFrame(columns=["**Value**[^val]", "**Model**"])
    note = data["Note"].values[0]
    if len(note) > 0:
        foot = f"[^p{model}]"
        note = f"[^p{model}]: " + note.replace("R²", r"\( \matrhm{R^2} \)")
    else:
        foot = ""
    df_property.loc["**Formation energy**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{eV/atom}} \)".format(data["Formation Energy (eV/atom)"].values[0])
    )
    df_property.loc["**Formation energy**", "**Model**"] = "GNoME"
    df_property.loc["**Band Gap**", "**Value**[^val]"] = (
        r"\( {:.3f} \;\pm\;{:.3f}  \; \mathrm{{eV}} \)".format(
            data["Average Band Gap (eV)"].values[0],
            data["Average Band Gap (deviation) (eV)"].values[0],
        )
    )
    df_property.loc["**Band Gap**", "**Model**"] = f"E(3)NN ({model})" + foot

    return df_property, note


def add_thermoelectric_properties(data, wT):
    df_property = pd.DataFrame(columns=["**Value**[^val]", "**Model**"])
    note = data["Note"].values[0]
    if len(note) > 0:
        foot = f"[^t{wT}]"
        note = f"[^t{wT}]: " + note.replace("R²", r"\( \matrhm{R^2} \)")
    else:
        foot = ""
    df_property.loc["**Formation energy**", "**Value**[^val]"] = (
        r"\( {:.3f} \; \mathrm{{eV/atom}} \)".format(data["Formation Energy (eV/atom)"].values[0])
    )
    df_property.loc["**Formation energy**", "**Model**"] = "GNoME"
    df_property.loc["**zT**", "**Value**[^val]"] = r"\( {:.3f} \;\pm\;{:.3f}  \)".format(
        data["Average zT (-)"].values[0], data["Average zT (deviation) (-)"].values[0]
    )
    df_property.loc["**zT**", "**Model**"] = f"E(3)NN ({wT}K)" + foot

    return df_property, note


def make_boc_grind(df, box_name):
    table = df.sort_index().to_markdown()
    table = (
        f"- __{box_name}__\n\n\t---\n\n"
        + "\n".join(["\t" + t for t in table.split("\n")])
        + "\n\n"
    )
    return table


def main():
    output_folder = root_dir / "docs/materials"
    meta_social_path = root_dir / "docs/meta_social.yaml"
    os.makedirs(output_folder, exist_ok=True)
    db_dict = get_database_dict()
    cif_files = get_cif_dict(db_dict)
    social_meta = load_meta_social(meta_social_path)
    for m_id, m_dict in tqdm(cif_files.items(), desc="Making materials pages"):
        cif_path = m_dict["cif"]
        page_content, chem_formula = make_viewer_gen_info(m_id, cif_path)

        # get info
        infos = []
        ai_experts_m_perv = {}
        ai_experts_dev_perv = {}
        ai_experts_m_thermo = {}
        ai_experts_dev_thermo = {}
        for db_key in m_dict["in_db"]:
            if "cathodes" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                wion = db_key.split("/")[-1]
                ai_experts_m = data["AI-experts confidence (-)"].values[0]
                ai_experts_dev = data["AI-experts confidence (deviation) (-)"].values[0]
                infos.append(make_cathode_info(wion, ai_experts_m, ai_experts_dev))
            elif "perovskites" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                model = db_key.split("/")[-1].replace("_models", "")
                ai_experts_m_perv[model] = data["AI-experts confidence (-)"].values[0]
                ai_experts_dev_perv[model] = data["AI-experts confidence (deviation) (-)"].values[
                    0
                ]
            elif "thermoelectrics" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                wT = db_key.split("/")[-1].replace("K", "")
                ai_experts_m_thermo[wT] = data["AI-experts confidence (-)"].values[0]
                ai_experts_dev_thermo[wT] = data["AI-experts confidence (deviation) (-)"].values[0]

        if len(ai_experts_m_perv) > 0:
            infos.append(make_perovskite_info(ai_experts_m_perv, ai_experts_dev_perv))
        if len(ai_experts_m_thermo) > 0:
            infos.append(make_thermoelectric_info(ai_experts_m_thermo, ai_experts_dev_thermo))

        page_content += "\n".join(infos)

        # get properties
        notes = [
            r"[^val]: The value after the \(\pm\) symbol does not indicate the *uncertainty* of the model but rather the *deviation*, specifically the root mean square error (RMSE) among the committee of models used. The value before the symbol represents the mean prediction from the committee."
        ]

        property_dfs_cath = []
        property_dfs_perov = []
        property_dfs_thermo = []
        for db_key in m_dict["in_db"]:
            if "cathodes" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                wion = db_key.split("/")[-1]
                property_df_, note_ = add_cathode_properties(data, wion)
                notes.append(note_)
                property_dfs_cath.append(property_df_)
            elif "perovskites" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                model = db_key.split("/")[-1].replace("_models", "")
                property_df_, note_ = add_perovskites_properties(data, model)
                notes.append(note_)
                property_dfs_perov.append(property_df_)
            elif "thermoelectrics" in db_key:
                data = db_dict[db_key][db_dict[db_key]["Material Id"] == m_id]
                wT = db_key.split("/")[-1].replace("K", "")
                property_df_, note_ = add_thermoelectric_properties(data, wT)
                notes.append(note_)
                property_dfs_thermo.append(property_df_)

        grids = []
        if len(property_dfs_cath) > 0:
            property_dfs_cath = pd.concat(property_dfs_cath).drop_duplicates()
            grids.append(make_boc_grind(property_dfs_cath, "Predicted properties (cathode)"))
        if len(property_dfs_perov) > 0:
            property_dfs_perov = pd.concat(property_dfs_perov).drop_duplicates()
            grids.append(make_boc_grind(property_dfs_perov, "Predicted properties (perovskites)"))
        if len(property_dfs_thermo) > 0:
            property_dfs_thermo = pd.concat(property_dfs_thermo).drop_duplicates()
            grids.append(
                make_boc_grind(property_dfs_thermo, "Predicted properties (thermoelectric)")
            )

        grid = '<div class="grid cards" style="margin: 0 auto;" markdown>\n\n'
        grid += "\n".join(grids)
        grid += "</div>\n"

        page_content += grid + "\n"
        page_content += "\n".join(notes)
        # update social card meta
        out_html_path = f"materials/{m_id}/index.html"
        social_meta[out_html_path] = {
            "description": "A Living Database of Selected Materials for Energy Applications",
            "img": "assets/images/social/materials/material_viewer.png",
            "title": "Material Detail",
        }
        # save
        out_path = os.path.join(output_folder, m_id + ".md")
        with open(out_path, "w") as f:
            f.write(page_content)
        # break
    save_meta_social(meta_social_path, social_meta)


if __name__ == "__main__":
    main()
