import os
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import yaml

# Configuration
BUILD_DIR = "site"
DOCS_DIR = "docs"
SUBDIR_TO_SKIP = ["overrides", "partial"]
META_SOCIAL_FILE = os.path.join(DOCS_DIR, "meta_social.yaml")
ASSETS_DIR = os.path.join(BUILD_DIR, "assets/images/social")
SITE_PREFIX = "https://paolodeangelis.github.io/Energy-GNoME/"
DEFAULT_IMAGE_SIZE = (1200, 630)  # Standard social card size
ICON = "docs/assets/img/logo.png"

# Font paths
ROBOTO_PATH = "docs/assets/fonts/Roboto"  # Update this to the folder containing Roboto fonts
FONT_BOLD = f"{ROBOTO_PATH}/Roboto-Bold.ttf"
FONT_REGULAR = f"{ROBOTO_PATH}/Roboto-Regular.ttf"

# Font colors
TITLE_COLOR = "#ffffff"
DESCRIPTION_COLOR = "#ffffff"
SITE_NAME_COLOR = "#ffffff"


def clean_markdown(text):
    """
    Removes markdown special characters from the text.
    """
    if not text:
        return text
    # Remove markdown formatting characters: *, **, `
    return re.sub(r"[*`]+", "", text)


def load_meta_social():
    """
    Loads the existing meta_social.yaml file, if it exists.
    """
    if os.path.exists(META_SOCIAL_FILE):
        with open(META_SOCIAL_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_meta_social(meta_social):
    """
    Saves the updated meta_social dictionary to the YAML file.
    """
    with open(META_SOCIAL_FILE, "w", encoding="utf-8") as f:
        yaml.dump(meta_social, f, default_flow_style=False)


def extract_metadata_from_html(html_file):
    """
    Extracts metadata and image path from an HTML file.
    """
    try:
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Extract metadata
        title = soup.find("meta", {"property": "og:title"})
        description = soup.find("meta", {"property": "og:description"})
        image_meta = soup.find("meta", {"property": "og:image"})

        # Extract image path and remove the site prefix if present
        image_url = image_meta["content"] if image_meta else None
        if image_url and image_url.startswith(SITE_PREFIX):
            image_path = image_url.replace(SITE_PREFIX, "")
        else:
            image_path = None

        return {
            "title": clean_markdown(title["content"] if title else None),
            "description": description["content"] if description else None,
            "image_path": image_path,
        }
    except Exception as e:
        print(f"Warning: Failed to extract metadata from {html_file} - {e}")
        return {"title": None, "description": None, "image_path": None}


def update_html_meta(html_file, meta_data):
    """
    Updates the HTML file with Open Graph and Twitter meta tags.
    """
    try:
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Define meta tags to update
        meta_tags = {
            "og:title": meta_data["title"],
            "og:description": meta_data["description"],
            "og:image": meta_data["image_url"],
            "og:url": f"{SITE_PREFIX}{os.path.relpath(html_file, BUILD_DIR)}",
            "twitter:title": meta_data["title"],
            "twitter:description": meta_data["description"],
            "twitter:image": meta_data["image_url"],
            "twitter:card": "summary_large_image",
        }

        # Update or insert each meta tag
        for prop, content in meta_tags.items():
            if prop.startswith("og:"):
                tag = soup.find("meta", {"property": prop}) or soup.new_tag("meta")
                tag["property"] = prop
            elif prop.startswith("twitter:"):
                tag = soup.find("meta", {"name": prop}) or soup.new_tag("meta")
                tag["name"] = prop
            tag["content"] = content
            if not tag.parent:
                soup.head.append(tag)

        # Save updated HTML
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(str(soup))
        # print(f"Updated metadata for {html_file}")

    except Exception as e:
        print(f"Warning: Failed to update HTML meta for {html_file} - {e}")


def create_test_image(image_path, title, description, site_name="Energy GNoME", icon_path=ICON):
    """
    Creates a social card with a background, icon, title, and description.
    Always overwrites the image if it already exists.
    """
    background_path = "docs/assets/img/social/social_bg.png"

    # Dimensions of the social card
    card_width = 1200
    card_height = 630
    max_title_width = 1030

    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))

    # Load the background image
    try:
        background = Image.open(background_path)
        background = ImageOps.fit(
            background, (card_width, card_height), method=Image.Resampling.LANCZOS
        )
    except FileNotFoundError:
        print(
            f"Warning: Background image not found at {background_path}. Using a plain background."
        )
        background = Image.new(
            "RGB", (card_width, card_height), color=(255, 255, 255)
        )  # Plain white background

    # Load the icon
    try:
        icon = Image.open(icon_path).convert("RGBA")
        icon_width, icon_height = icon.size
        scale_factor = 150 / icon_height  # Scale icon height to 100px
        icon = icon.resize((int(icon_width * scale_factor), 150), Image.Resampling.LANCZOS)
    except FileNotFoundError:
        print(f"Warning: Icon image not found at {icon_path}. Skipping the icon.")
        icon = None

    # Create a drawing context
    draw = ImageDraw.Draw(background)

    # Define fonts
    try:
        title_font = ImageFont.truetype(FONT_BOLD, 100)  # Large bold font for title
        site_font = ImageFont.truetype(FONT_BOLD, 45)  # Medium bold font for site name
        description_font = ImageFont.truetype(FONT_REGULAR, 30)  # Regular font for description
    except OSError:
        print("Warning: Could not load custom fonts. Using default fonts.")
        title_font = ImageFont.load_default()
        site_font = ImageFont.load_default()
        description_font = ImageFont.load_default()

    # Function to calculate text dimensions
    def get_text_size(text, font):
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height

    # Function to wrap text
    def wrap_text(text, font, max_width):
        wrapped_lines = []
        for line in text.split("\n"):
            words = line.split()
            wrapped_line = ""
            for word in words:
                test_line = f"{wrapped_line} {word}".strip()
                if get_text_size(test_line, font)[0] <= max_width:
                    wrapped_line = test_line
                else:
                    wrapped_lines.append(wrapped_line)
                    wrapped_line = word
            wrapped_lines.append(wrapped_line)
        return wrapped_lines

    # Add the icon on the top-right
    if icon:
        icon_position = (card_width - icon.size[0] - 90, 70)
        background.paste(icon, icon_position, icon)

    # Add the site name on the top-left
    site_text_position = (90, 70)
    draw.text(site_text_position, site_name, font=site_font, fill=SITE_NAME_COLOR)

    # Add the title in the center, wrapped if necessary
    title = clean_markdown(title or "Untitled")
    wrapped_title = wrap_text(title, title_font, max_title_width)
    current_y = 190
    N_line = len(wrapped_title)
    if N_line > 3:
        title_font = ImageFont.truetype(FONT_BOLD, 70)
        wrapped_title = wrap_text(title, title_font, max_title_width)
        if len(wrapped_title) > 3:
            wrapped_title = wrapped_title[:3]
            wrapped_title[-1] += "..."
    for line in wrapped_title:
        line_width, line_height = get_text_size(line, title_font)
        line_position = (90, current_y)
        draw.text(line_position, line, font=title_font, fill=TITLE_COLOR)
        current_y += line_height + 25

    # Add the description at the bottom
    description = description or "No description provided."
    desc_width, desc_height = get_text_size(description, description_font)
    desc_position = (90, card_height - 100)
    draw.text(desc_position, description, font=description_font, fill=DESCRIPTION_COLOR)

    # Save the final image
    background.save(image_path)
    # print(f"Generated social card: {image_path}")


def process_files():
    """
    Processes all HTML files and updates social metadata.
    """
    # Load existing meta_social.yaml
    meta_social = load_meta_social()

    # Scan HTML files in the site directory
    for root, _, files in tqdm(list(os.walk(BUILD_DIR)), desc="making social"):
        for file in files:
            if any(folder in root.split(os.sep) for folder in SUBDIR_TO_SKIP):
                print(
                    f"Skipping operation for {file} in {root} as it is in one of the excluded subfolders."
                )
                continue
            if file.endswith(".html"):
                html_path = os.path.join(root, file)

                # Extract metadata from the HTML file
                metadata = extract_metadata_from_html(html_path)

                # Get relative HTML path for meta_social lookup
                relative_html_path = os.path.relpath(html_path, BUILD_DIR)

                # Override metadata with meta_social.yaml if available
                if relative_html_path in meta_social:
                    metadata.update(
                        {
                            "title": clean_markdown(
                                meta_social[relative_html_path].get("title", metadata.get("title"))
                            ),
                            "description": meta_social[relative_html_path].get(
                                "description", metadata.get("description")
                            ),
                            "image_path": meta_social[relative_html_path].get(
                                "img", metadata.get("image_path")
                            ),
                        }
                    )

                # Ensure image path is set
                img_path = (
                    os.path.join(BUILD_DIR, metadata["image_path"])
                    if metadata["image_path"]
                    else None
                )
                metadata["image_url"] = (
                    f"{SITE_PREFIX}{metadata['image_path']}" if metadata["image_path"] else None
                )

                # Update the HTML metadata
                update_html_meta(html_path, metadata)

                # Create the social card image
                if img_path:
                    # create_test_image(img_path, metadata["title"], metadata["description"])
                    try:
                        create_test_image(img_path, metadata["title"], metadata["description"])
                    except:  # noqa: E722
                        print(f"something wrong happen creating image {img_path}")

                # Update meta_social.yaml with final metadata
                meta_social[relative_html_path] = {
                    "img": metadata["image_path"],
                    "title": metadata["title"],
                    "description": metadata["description"],
                }

    # Save updated meta_social.yaml
    save_meta_social(meta_social)


if __name__ == "__main__":
    process_files()
