---
draft: false
date:
  created: 2025-01-11
categories:
  - Features
  - News
tags:
  - energy-gnome
authors:
  - team
  - paolodeangelis
---

# **Introducing Material-Specific Web Pages in Energy-GNoME**

We introduced a new feature that enhances the Energy-GNoME database's usability: **dedicated web pages for individual materials**! This update brings a significant step forward in how researchers and enthusiasts can explore materials, leveraging advanced visualization tools and property insights and enabling people who are not computational material experts to fully explore the database, alongside the dashboard.

<!-- more -->

![Material Page](../../assets/img/blog/material_page/material_page_overview.gif#only-light)
![Material Page](../../assets/img/blog/material_page/material_page_overview_dark.gif#only-dark)

## What's New?

Each material now has its own dedicated webpage designed to provide an **intuitive overview** of its characteristics and predicted properties. Here's what you can expect:

### 1. Interactive 3D Visualization with 3DMol.js
<div style="text-align: center;" markdown>
![3DMol viewer](../../assets/img/blog/material_page/material_viewer.png#only-light){ style="max-width: 500px;" }
![3DMol viewer](../../assets/img/blog/material_page/material_viewer_dark.png#only-dark){ style="max-width: 500px;" }
</div>

- [**3DMol.js**](https://3dmol.csb.pitt.edu/) powers an immersive 3D viewer for visualizing the material's structure.
- Rotate, zoom, and explore atomic arrangements in a highly interactive manner, making it easier to understand the material's structural characteristics.
- At the bottom, two buttons are available: one to toggle atomic labels in the 3D viewer to visualize the element of each atom, and another for downloading the `.CIF` file of the crystal.

### 2. Crystal Properties
<div style="text-align: center;" markdown>
![Crystal Properties](../../assets/img/blog/material_page/crystal_properties.png#only-light){ style="max-width: 300px;" }
![Crystal Properties](../../assets/img/blog/material_page/crystal_properties_dark.png#only-dark){ style="max-width: 300px;" }
</div>

- Detailed information about the **crystal structure**, including lattice parameters, density and symmetry groups.

### 3. Application Predictions
<div style="text-align: center;" markdown>
![Application Predictions](../../assets/img/blog/material_page/candidate_box.png#only-light)
![Application Predictions](../../assets/img/blog/material_page/candidate_box_dark.png#only-dark)
</div>

- For each material, highlight the AI-driven predictions for its potential energy applications.
- Note that some materials, such as cathodes for multivalent batteries, may demonstrate versatility and suitability across multiple energy domains.

### 4. Predicted Properties
<div style="text-align: center;" markdown>
![Predicted Properties](../../assets/img/blog/material_page/predicted_properties.png#only-light){ style="max-width: 400px;" }
![Predicted Properties](../../assets/img/blog/material_page/predicted_properties_dark.png#only-dark){ style="max-width: 400px;" }
</div>

- The predicted properties for each material are listed and categorized according to application and the specific models used for their prediction.

## How to Access These Pages?

Currently, these material-specific pages can be accessed via the [**Material Dashboard App**](../../apps/index.md){.wiki-hover} links associated with each of our screened applications.
By clicking on a material link within these dashboards, you will be directed to its corresponding webpage.

## Why This Matters

We believe these material-specific pages will:

- **Enhance Research Accessibility**: By offering an intuitive interface, researchers can easily access key material data without requiring coding or advanced data analysis skills.
- **Streamline Material Discovery**: The ability to visualize and analyze predicted properties helps researchers prioritize materials with the most promising characteristics.

!!! example

    - Possible Na cathode material [Fe~5~Na~4~O~22~P~6~](../../materials/35ca0bcad2.md)
    - Possible multivalent (Li-Na) cathode material [LiNaS~12~V~6~](../../materials/092187976b.md)
    - Possible perovskite material [Ba~3~Bi~2~LaO~9~](../../materials/4ed36f8003.md)
    - Possible thermoelectric material [Bi~3~Sb~13~Te~3~](../../materials/ac2022f2d0.md)
    - Possible Mg cathode and perovskite material [CoMgNd~2~O~6~](../../materials/bd64810b56.md)

<div style="text-align: right; margin-top: 20px;" markdown>
*Discover. Predict. Energize.*<br>
The Energy-GNoME Team
</div>
