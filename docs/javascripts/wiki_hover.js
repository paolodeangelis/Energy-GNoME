/**
 * @file Wiki Hover
 * @description Hover over a link to see the content of the link
 * Doesn't work on mobile and header
 * @see tippy https://atomiks.github.io/tippyjs/
 */

const blogURL = document.querySelector('meta[name="site_url"]')
  ? document.querySelector('meta[name="site_url"]').content
  : location.origin;
const position = ["top", "right", "bottom", "left"];

/**
 * @description Replace broken image with encoded image in first para
 * @param {Element} firstPara
 * @returns {Element} firstPara
 */
function brokenImage(firstPara) {
  const brokenImage = firstPara?.querySelectorAll("img");
  if (brokenImage) {
    for (let i = 0; i < brokenImage.length; i++) {
      const encodedImage = brokenImage[i];
      encodedImage.src = decodeURI(decodeURI(encodedImage.src));
      //replace broken image with encoded image in first para
      encodedImage.src = encodedImage.src.replace(
        location.origin,
        blogURL
      );
    }
  }
  return firstPara
}



/**
 * Strip text of first para of unwanted characters
 * @param {Element} firstPara
 * @returns {Element} firstPara
 */
function cleanText(firstPara) {
  firstPara.innerText = firstPara.innerText
    .replaceAll("↩", "")
    .replaceAll("¶", "");
  return firstPara
}

function calculateHeight(firstPara) {
  const paragraph = firstPara ? firstPara.innerText ? firstPara.innerText : firstPara : "";
  const height = Math.floor(
    paragraph.split(" ").length / 100
  );
  if (height < 2) {
    return `auto`;
  } else if (height >= 5) {
    return `25rem`;
  }
  return `${height}rem`;
}

try {
//   tippy(`.md-content a[href^="${blogURL}"], a.footnote-ref, a[href^="./"]`, {
    tippy(`.wiki-hover`, {
    content: "",
    allowHTML: true,
    animation: "scale-subtle",
    theme: "translucent",
    followCursor: true,
    arrow: false,
    touch: "hold",
    inlinePositioning: true,
    followCursor: false, // Disable following the mouse
    placement: "auto", // Automatically place based on available space
    // placement: position[Math.floor(Math.random() * position.length - 1)],
    onShow(instance) {
      fetch(instance.reference.href)
        .then((response) => response.text())
        .then((html) => {
          const parser = new DOMParser();
          return parser.parseFromString(html, "text/html");
        })
        .then((doc) => {
          //create section for each content after header
          const headers = doc.querySelectorAll("h1, h2, h3, h4, h5, h6");
          headers.forEach(function (header) {
            const headerName = header.id || header.innerText.split("\n")[0].toLowerCase().replaceAll(" ", "-");
            if (headerName.length > 0) {
              const div = doc.createElement("div");
              div.classList.add(headerName);
              let nextElement = header.nextElementSibling;
              while (nextElement && !nextElement.matches("h1, h2, h3, h4, h5, h6")) {
                div.appendChild(nextElement);
                nextElement = nextElement.nextElementSibling;
              }
              header.parentNode.insertBefore(div, header.nextSibling);
            }
          });
            // Inject a base tag into the fetched document
            const base = doc.createElement("base");
            base.href = instance.reference.href; // Set base to the URL of the fetched content
            doc.head.appendChild(base);
          return doc;
        })
        //inject head into doc
        .then((doc) => {
          if (location.href.replace(location.hash, "") === instance.reference.href) {
            instance.hide();
            instance.destroy();
            return;
          }
           // Remove specific elements or classes
          const elementsToRemove = doc.querySelectorAll('.md-source-file, .md-content__button');
          elementsToRemove.forEach((element) => element.remove());
          let firstPara = doc.querySelector("article");
          const firstHeader = doc.querySelector("h1");
          if (firstHeader && firstHeader.innerText === "Index") {
            const realFileName = decodeURI(
              doc.querySelector('link[rel="canonical"]').href
            )
              .split("/")
              .filter((e) => e)
              .pop();
            firstHeader.innerText = realFileName;
          }
          //broken link in first para
          firstPara = brokenImage(firstPara);
          const element1 = document.querySelector(`[id^="tippy"]`);
          if (element1) {
            element1.classList.add("tippy");
          }
          const partOfText = instance.reference.href.replace(/.*#/, "#");
          let toDisplay = firstPara;
          let displayType;
          if (partOfText.startsWith("#")) {
            firstPara = doc.querySelector(
              `[id="${partOfText.replace("#", "")}"]`
            );
            if (firstPara.tagName.includes("H")) {
              const articleDOM = doc.createElement("article");
              articleDOM.classList.add("md-content__inner", "md-typeset");
              articleDOM.appendChild(doc.querySelector(`div.${partOfText.replace("#", "")}`));
              toDisplay = articleDOM;
              firstPara = toDisplay;
            } else if (firstPara.innerText.replace(partOfText).length === 0) {
              firstPara = doc.querySelector("div.citation");
              toDisplay = firstPara;
            } else {
              toDisplay = cleanText(firstPara).innerText;
            }
            instance.popper.style.height = "auto";
          } else {
            instance.popper.style.height = calculateHeight(firstPara);
          }

          instance.popper.placement =
            position[Math.floor(Math.random() * position.length)];
          if (firstPara.innerText.length > 0) {
            if (!displayType) {
              instance.setContent(toDisplay)
              instance.popper.style.height = calculateHeight(toDisplay);
            }
          } else {
            firstPara = doc.querySelector("article");
            instance.reference.href.replace(/.*#/, "#");
            instance.popper.style.height = calculateHeight(firstPara);
          }
        })
        .catch((error) => {
          console.log(error);
          instance.hide();
          instance.destroy();
        });
    },
  });
} catch {
  console.log("tippy error, ignore it");
}
