---
title: material
material_viewer: true
hide:
  - path
  - navigation
  - toc
  - feedback
---


<!-- docs/material.html -->
<!-- <!DOCTYPE html>
<html lang="en">
<!-- <head> -->
  <!-- <meta charset="UTF-8">
  <title>Material Viewer</title> --> 
  <!-- Include jQuery, 3Dmol.js in your MkDocs build (e.g., via extra_javascript in mkdocs.yml) -->
  <!-- Or you can load them here if hosting the scripts locally. -->
  <!--
    <script src="path/to/jquery.min.js"></script>
    <script src="path/to/3Dmol-min.js"></script>
  -->
  
  <script>
    let viewer;
    let labelsVisible = false;

    async function loadMaterial() {
      // 1) Get the ID from the URL, e.g. ?id=092187976b
      const urlParams = new URLSearchParams(window.location.search);
      const materialId = urlParams.get("id");
      if (!materialId) {
        document.body.innerHTML = "<h1>Error: No material ID provided.</h1>";
        return;
      }

      // 2) Fetch the big JSON file of all materials
      const response = await fetch("materials.json");
      const allMaterials = await response.json();

      // 3) Look up the requested material
      const matData = allMaterials[materialId];
      if (!matData) {
        document.body.innerHTML = "<h1>Material not found in materials.json</h1>";
        return;
      }

      // 4) Inject some basic info (Formula, ID, etc.)
      document.getElementById("mat-title").innerHTML =
        matData.chemical_formula_html + ` — <code>${materialId}</code>`;

      // **Update the page title by removing HTML tags**
      var formulaText = matData.chemical_formula_html.replace(/<[^>]+>/g, ''); // Strips HTML tags
      document.title = formulaText + " - Energy-GNoME";

      // Fill the cell parameters
      const cell = matData.cell;
      document.getElementById("cell-a").textContent = cell.a.toFixed(2);
      document.getElementById("cell-b").textContent = cell.b.toFixed(2);
      document.getElementById("cell-c").textContent = cell.c.toFixed(2);
      document.getElementById("cell-alpha").textContent = cell.alpha.toFixed(1);
      document.getElementById("cell-beta").textContent = cell.beta.toFixed(1);
      document.getElementById("cell-gamma").textContent = cell.gamma.toFixed(1);

      document.getElementById("density").textContent =
        matData.density_gcm3.toFixed(2);
      document.getElementById("spacegroup").innerHTML =
        matData.space_group;
      document.getElementById("chem-system").textContent =
        matData.chemical_system;
      document.getElementById("num-sites").textContent = matData.num_sites;

      // 5) Setup 3Dmol viewer
      viewer = $3Dmol.createViewer("container");
      viewer.setBackgroundColor(0x1f2429, 0.0);

      // 6) Load CIF
      jQuery.ajax(matData.cif_url, {
            dataType: "text",
            success: function(data) {
                atoms = viewer.addModel(data, "cif");
                // Show a ball-and-stick style by default
                viewer.setStyle({}, {
                    stick:{
                        radius:0.15,
                        // Use the Jmol color scheme:
                        colorscheme: "Jmol"
                        },
                        sphere:{
                        scale:0.33,
                        // Use the Jmol color scheme:
                        colorscheme: "Jmol"
                        }
                    });
                // Add the unit cell box (if CIF has cell info)
                viewer.addUnitCell(atoms,{
                    box:{
                        color:0x64c4d3
                        }
                    });
                viewer.zoomTo();
                viewer.render();
            }
        });

      // 7) Render any predicted properties
const propsSection = document.getElementById("predicted-properties");

if (matData.predicted_properties) {
    let tableHtml = `<ul>`;
    let footnoteIndex = 2; // Starting from 2 since 1 is used for [^val]
    const footnotes = [];
    const modelFootnotes = new Map(); // To track models that already have footnotes

    const abbreviationDict = {
        "GNoME": "Graph Networks for Materials Exploration",
        "E(3)NN": "Euclidean Neural Networks",
        "GBDT": "Gradient Boosted Decision Trees",
    };

    function addAbbreviationToModel(model) {
        Object.keys(abbreviationDict).forEach(abbr => {
            const regex = new RegExp(`\\b${abbr.replace(/[.*+?^=!:${}()|[\]/\\]/g, "\\$&")}\\b`, 'g');
            model = model.replace(regex, match => {
                return `<abbr title="${abbreviationDict[abbr]}">${match}</abbr>`;
            });
        });
        return model;
    }

    Object.keys(matData.predicted_properties).forEach(category => {
        const categoryData = matData.predicted_properties[category];

        // Extract footnotes per model (e.g., [^cMixed]: text)
        const modelNotes = {};
        const noteLines = Array.isArray(categoryData.note)
            ? categoryData.note
            : [categoryData.note];

        noteLines.forEach(noteLine => {
            const matches = noteLine?.matchAll(/\[\^c(.+?)\]:\s*(.+)/gi);
            if (matches) {
                for (const match of matches) {
                    const tag = match[1].toLowerCase();
                    const noteText = match[2];
                    modelNotes[tag] = noteText;
                }
            }
        });

        if (categoryData && categoryData.properties) {
            tableHtml += `
                <li>
                    <p><strong>Predicted properties (${category})</strong></p>
                    <hr>
                    <table>
                        <tr>
                        <th></th>
                        <th><b>Value</b><sup id="fnref:val"><a class="footnote-ref" href="#fn:val">1</a></sup></th>
                        <th><b>Model</b></th>
                        </tr>`;

            Object.keys(categoryData.properties).forEach(property => {
                const propertyData = categoryData.properties[property];
                const seenModels = new Set(); // Track models already processed for this property

                if (Array.isArray(propertyData["Value[^val]"]) && Array.isArray(propertyData["Model"])) {
                    propertyData["Value[^val]"].forEach((value, index) => {
                        let model = propertyData["Model"][index];

                        // Skip if model has been processed already
                        if (seenModels.has(model)) return;

                        let modelWithAbbr = addAbbreviationToModel(model);

                        // Check for matching footnote
                        let noteTag = Object.keys(modelNotes).find(tag =>
                            model.toLowerCase().includes(tag)
                        );

                        if (noteTag && !modelFootnotes.has(model)) {
                            const footnoteKey = `c${noteTag}`.toLowerCase();  // model-specific key
                            const footnoteNumber = footnoteIndex;

                            modelWithAbbr += `<sup id="fnref:${footnoteKey}" data-note-key="${model}"><a class="footnote-ref" href="#fn:${footnoteKey}">${footnoteNumber}</a></sup>`;

                            footnotes.push({
                                index: footnoteNumber,
                                text: modelNotes[noteTag]
                            });

                            modelFootnotes.set(model, { key: footnoteKey, number: footnoteNumber }); // store both
                            footnoteIndex++;

                        } else if (noteTag) {
                            const { key: footnoteKey, number: footnoteNumber } = modelFootnotes.get(model);
                            modelWithAbbr += `<sup id="fnref:${footnoteKey}" data-note-key="${model}"><a class="footnote-ref" href="#fn:${footnoteKey}">${footnoteNumber}</a></sup>`;
                        }

                        tableHtml += `<tr>
                            <td><b>${property}</b></td>
                            <td>${value}</td>
                            <td>${modelWithAbbr}</td>
                        </tr>`;

                        seenModels.add(model); // Mark this model as processed
                    });
                } else {
                    let model = propertyData["Model"];
                    let value = propertyData["Value[^val]"];

                    let modelWithAbbr = addAbbreviationToModel(model);

                    let noteTag = Object.keys(modelNotes).find(tag =>
                        model.toLowerCase().includes(tag)
                    );

                    // If footnote is found and model hasn't been assigned a footnote yet
                    if (noteTag && !modelFootnotes.has(model)) {
                        const footnoteKey = `c${noteTag}`;  // Use model-specific key, e.g., 'cCs-cathode'
                        modelWithAbbr += `<sup id="fnref:${footnoteKey}"><a class="footnote-ref" href="#fn:${footnoteKey}">${footnoteIndex}</a></sup>`;

                        footnotes.push({
                            index: footnoteIndex,
                            text: modelNotes[noteTag]
                        });

                        // Mark the model as having a footnote with the unique footnote index
                        modelFootnotes.set(model, footnoteKey);  // Store model-specific key
                        footnoteIndex++; // Increment the footnote index
                    } else if (noteTag) {
                        // Reuse the footnote index if the model already has a footnote
                        const footnoteKey = modelFootnotes.get(model);  // Get the model-specific footnote key
                        modelWithAbbr += `<sup id="fnref:${footnoteKey}"><a class="footnote-ref" href="#fn:${footnoteKey}">${footnoteIndex - 1}</a></sup>`;
                    }

                    tableHtml += `<tr>
                        <td><b>${property}</b></td>
                        <td>${value ?? '–'}</td>
                        <td>${modelWithAbbr ?? '–'}</td>
                    </tr>`;
                }
            });

            tableHtml += `</table></li>`;
        }
    });

    tableHtml += `</ul>`;

    propsSection.innerHTML = tableHtml;
}

      // 8) Function to process and format infos dynamically
      function generateInfosHtml(infos) {
          if (!infos || infos.length === 0) {
              return ""; // Return empty if there are no infos
          }

          let infosHtml = "";

          infos.forEach(info => {
              let htmlText = info.text
              infosHtml += htmlText;
          });

          return infosHtml;
      }

      // 9) Rendering the infos dynamically
      const infosSection = document.getElementById("infos-section");

      if (matData.infos) {
          infosSection.innerHTML = generateInfosHtml(matData.infos);
      }

      // 10) Append category notes as numbered footnotes outside the table
        const footnoteList = document.querySelector(".footnote ol");

        // Create a map to track which noteTag got which number
        const footnoteNumberMap = {};
        let footnoteCounter = footnoteList.querySelectorAll("li").length + 1;

        const usedModels = new Set();

        if (matData.predicted_properties) {
            Object.entries(matData.predicted_properties).forEach(([category, categoryData]) => {
                const rawNotes = categoryData.note;
                const notesArray = Array.isArray(rawNotes) ? rawNotes : [rawNotes];

                const modelNotes = {};

                // Extract all [^cXYZ]: ... into a map
                notesArray.forEach(noteString => {
                    const match = noteString.match(/\[\^c(.+?)\]:\s*(.+)/i);
                    if (!match) return;
                    const tag = match[1].toLowerCase();
                    const noteText = match[2];
                    modelNotes[tag] = noteText;

                    // Only assign footnote number once per tag
                    if (!(tag in footnoteNumberMap)) {
                        footnoteNumberMap[tag] = footnoteCounter++;
                    }
                });

                // Loop over each property and model
                Object.entries(categoryData.properties).forEach(([property, propertyData]) => {
                    const models = propertyData["Model"];
                    if (!Array.isArray(models)) return;

                    models.forEach((model, index) => {
                        if (usedModels.has(model)) return;
                        usedModels.add(model);

                        // Find matching tag for this model
                        const noteTag = Object.keys(modelNotes).find(tag =>
                            model.toLowerCase().includes(tag)
                        );

                        if (noteTag) {
                            const footnoteKey = `c${noteTag}`; // Model-specific key
                            const noteText = modelNotes[noteTag];

                            // Add footnote only once per model
                            if (!document.getElementById(footnoteKey)) {
                                const li = document.createElement("li");
                                li.id = footnoteKey;
                                li.innerHTML = `
                                    <p>
                                        ${noteText}&nbsp;
                                        <a class="footnote-backref" href="#fnref:${footnoteKey}" title="Jump back to footnote ${footnoteNumberMap[noteTag]} in the text">↩</a>
                                    </p>
                                `;
                                footnoteList.appendChild(li);
                            }

                            // Update all superscripts for this model
                            const noteSuperscripts = document.querySelectorAll(`sup[data-note-key="${model}"]`);
                            noteSuperscripts.forEach(sup => {
                                sup.innerHTML = `<a class="footnote-ref" href="#${footnoteKey}">${footnoteNumberMap[noteTag]}</a>`;
                            });
                        }
                    });
                });
            });
        }

    }

    function toggleLabels() {
        const theButton = document.querySelector('.controls-button');
        if (!labelsVisible) {
            // Show atom labels
            viewer.removeAllLabels();
            let selected = viewer.selectedAtoms({});
            for (let i = 0; i < selected.length; i++) {
                let a = selected[i];
                viewer.addLabel(a.elem, {
                    position: { x: a.x, y: a.y, z: a.z },
                    backgroundColor: "white",
                    backgroundOpacity: 0.5,
                    fontColor: 0x1f2429,
                    fontSize: 14
                });
            }
             // Toggle "active" style
            theButton.classList.add('active-button');
            labelsVisible = true;
        } else {
            // Hide atom labels
            viewer.removeAllLabels();
            labelsVisible = false;
            // Remove "active" style
            theButton.classList.remove('active-button');
            labelsVisible = false;
        }
        viewer.render();
    }

    function downloadCifFile() {
      // Use the same materialId => matData => matData.cif_url
      // But we must re-fetch the file as a blob
      const urlParams = new URLSearchParams(window.location.search);
      const materialId = urlParams.get("id");
      fetch(`materials.json`)
        .then(r => r.json())
        .then(allMats => {
          if(!allMats[materialId]) return;
          const cifUrl = allMats[materialId].cif_url;
          // Now fetch the actual CIF
          fetch(cifUrl)
            .then(res => {
              if(!res.ok){
                throw new Error(`HTTP error! status: ${res.status}`);
              }
              return res.blob();
            })
            .then(blob => {
              const tempUrl = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.style.display = "none";
              a.href = tempUrl;
              a.download = `${materialId}.cif`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(tempUrl);
            })
            .catch(err => {
              alert("Could not download CIF: " + err.message);
            });
        });
    }

    window.onload = loadMaterial;
  </script>
 
 <style>
/* Turn the <ul> into a flex container */
  .flex-wrapper {
      display: flex !important;
      gap: 20px;       /* space between items */
      padding: 0;      /* remove default <ul> padding/margins */
      margin: 0 !important;
      border: 0 !important;
  }

  /* Make the first element wider than the second */
  .viewer-li {
      /* flex: 2;         2 parts out of total 3, so ~66% of space */
      display: flex !important;
      width: 60% !important;
      min-width: 600px !important;/* enforce a minimum width if you like */
  }
  .table-li {
      flex: 1;         /* 1 part out of total 3, so ~33% of space */
      min-width: 200px;
      /* list-style: none; */
  }
  /* Flex container for side-by-side layout */
  .container {
      display: flex;
      flex-wrap: wrap; /* so that on narrow screens, they stack */
      gap: 20px;
      margin: 0 auto;
  }
  .viewer-section {
      flex: 1 1 400px;
      min-width: 500px;
  }
  table {
      border-collapse: collapse;
      border-spacing: 0;
      border:none!important;
      font-size: .75rem!important;
  }
  th {
  font-weight: 400!important; /* or 400 */
  }
  .admonition {
      font-size: .75rem!important;
  }

  .table-section {
      flex: 1 1 200px;
      min-width: 200px;
      border-collapse: collapse;
      margin: 0 auto;
      /* No outer border on the table itself */
      border: none;
  }
  .table-section td, .table-section th {
      min-width: 120px;  /* each column has at least 100px */
  }

  /* Adjust overall container style */
  .viewer-container {
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
  }
  .controls {
      margin-top: 10px;
  }
  .controls button {
      margin: 5px;
      padding: .625em 2em;
  }
  .controls-button {
  border: .1rem solid;
  border-radius: .1rem;
  color: var(--md-primary-fg-color);
  cursor: pointer;
  display: inline-block;
  font-weight: 700;
  transition: color 125ms, background-color 125ms, border-color 125ms;
  background-color: transparent; /* Default: no background */
  }

  /* Hover effect: when the mouse is over the button */
  .controls-button:hover {
  background-color: var(--md-primary-fg-color);
  border-color: var(--md-primary-fg-color);
  color: #fff;
  }

  /* Toggle/Active class: when clicked, we apply this class via JS */
  .active-button {
  background-color: var(--md-primary-fg-color);
  color: #fff;
  }
</style>



<h1 id="mat-title">Loading...</h1>

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
        <div id="container" style="height: 480px; width: 100%; position: relative;"></div>
        <div class="controls">
            <!-- <button onclick="viewer.spin(true)">Spin</button> -->
            <!-- <button onclick="viewer.spin(false)">Stop</button> -->
            <button class="controls-button" onclick="toggleLabels()">Show/Hide Atom Names</button>
            <button class="md-button md-button-primary" onclick="downloadCifFile()">Download CIF</button>
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
          <tr><td><b>a</b></td><td><span id="cell-a"></span> Å</td></tr>
          <tr><td><b>b</b></td><td><span id="cell-b"></span> Å</td></tr>
          <tr><td><b>c</b></td><td><span id="cell-c"></span> Å</td></tr>
          <tr><td><b>α</b></td><td><span id="cell-alpha"></span> °</td></tr>
          <tr><td><b>β</b></td><td><span id="cell-beta"></span> °</td></tr>
          <tr><td><b>γ</b></td><td><span id="cell-gamma"></span> °</td></tr>
          <tr><td><b>Density</b></td><td><span id="density"></span> g/cm³</td></tr>
          <tr><td><b>Space group</b></td><td><span id="spacegroup"></span></td></tr>
          <tr><td><b>Chemical system</b></td><td><span id="chem-system"></span></td></tr>
          <tr><td><b>Number of sites</b></td><td><span id="num-sites"></span></td></tr>
        </table>
    </div>
    </li>
    </ul>
  </div>

!!! quote ""

    Data contained in the Graph Networks for Materials Exploration (GNoME) Database is available for use under
    the terms of the Creative Commons Attribution Noncommercial 4.0 International Licence ([CC BY NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)).
    If you are using this resource please cite the following publications:

    - Merchant, A., Batzner, S., Schoenholz, S.S. *et al.* "Scaling deep learning for materials discovery". *Nature* 624, 80-85, **2023**. doi: [10.1038/s41586-023-06735-9](https://doi.org/10.1038/s41586-023-06735-9).
    - De Angelis P., Trezza G., Barletta G., Asinari P., Chiavazzo E. "Energy-GNoME: A Living Database of Selected Materials for Energy Applications". *arXiv* November 15, **2024**. doi: [10.48550/arXiv.2411.10125](https://doi.org/10.48550/arXiv.2411.10125).

  <div id="infos-section"></div>

  <!-- Predicted Properties Section -->
  <!-- <li class="table-li", list-style= None> -->
  <div id="predicted-properties" class="grid cards" style="margin: 0 auto;">
      <!-- This will be dynamically populated -->
  </div> 

  <div class="footnote">
    <hr>
    <ol>
      <li id="fn:val">
        <p>
        The value after the '±' symbol does not indicate the <em>uncertainty</em> of the model but rather the <em>deviation</em>, specifically the root mean square error (RMSE) among the committee of models used. The value before the symbol represents the mean prediction from the committee.&nbsp;
        <a class="footnote-backref" href="#fnref:val" title="Jump back to footnote 1 in the text">↩</a>
        <a class="footnote-backref" href="#fnref2:val" title="Jump back to footnote 1 in the text">↩</a>
        </p>
      </li>
    </ol>
  </div>