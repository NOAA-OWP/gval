#!/usr/bin/env python3

import os
import re
import shutil
from glob import glob
from tempfile import mkdtemp

from pypandoc import convert_text
from pypandoc.pandoc_download import download_pandoc

from sphinx import conf


def compile_readme() -> None:
    """
    Compiles the README.MD, SPHINX_README.MD, and PYPI_README.MD files from the markdown files in the markdown directory (`docs/markdown`). Replace the path to the images in the local images directory (`docs/images`) with the urls to the images in the main branch of the repository.

    This also copies the images, notebooks, and CONTRIBUTING.MD file to the sphinx directory (`docs/sphinx`).
    """

    # Make pandoc path
    os.makedirs(conf.pandoc_dir_path, exist_ok=True)

    # pandoc executable path and add to environment variables
    os.environ.setdefault("PYPANDOC_PANDOC", conf.pandoc_executable_path)

    # Set pandoc download path
    pandoc_download_path = mkdtemp("pandoc_tmp")

    # Download pandoc
    download_pandoc(
        targetfolder=conf.pandoc_dir_path, download_folder=pandoc_download_path
    )

    # Define input and output paths
    input_path = os.path.join(conf.docs_dir_path, "markdown", "*.MD")
    output_path = os.path.join(conf.docs_dir_path, "..", "README.MD")

    # Get list of markdown files
    md_files = glob(input_path)

    # Sort markdown files
    md_files = sorted(md_files)

    # Read and concatenate markdown files
    concatenated_md = ""
    for md_file in md_files:
        with open(md_file, "r") as file:
            concatenated_md += file.read().strip() + "\n\n"

    # Strip trailing whitespace from concatenated markdown
    concatenated_md = concatenated_md.strip()

    # Convert concatenated markdown to README
    convert_text(concatenated_md, "gfm", format="gfm", outputfile=output_path)

    contents = None
    with open(f"{conf.docs_dir_path}/../README.MD", "r") as file:
        contents = file.read()
        contents = contents.replace(
            "../images", "https://github.com/NOAA-OWP/gval/raw/main/docs/images"
        )
        contents = contents.replace("\\_", "_")
        contents = contents.replace("\\*", "*")
        contents = contents.replace("  -", "-")

        matches = re.findall("<code>[^>]*>[^~]*?", contents)

        for match in matches:
            contents = contents.replace(match, match.replace(" ", "&nbsp;"))

    with open(f"{conf.docs_dir_path}/../README.MD", "w") as file:
        file.write(contents)

    # For Sphinx documentation
    sphinx_contents = contents.replace(
        "![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/"
        "gval_dark_mode.png#gh-dark-mode-only)",
        "",
    )

    with open(f"{conf.docs_dir_path}/sphinx/PYPI_README.MD", "w") as file:
        file.write(sphinx_contents)

    sphinx_contents = sphinx_contents.replace(
        "![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/"
        "gval_light_mode.png#gh-light-mode-only)",
        '<img src="https://github.com/NOAA-OWP/gval/raw/main/docs/images/'
        'gval_light_mode.png" '
        'width="350" height="130" />',
    )

    sphinx_contents = sphinx_contents.replace(
        "![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/"
        "agreement_map.png)",
        '<img src="https://github.com/NOAA-OWP/gval/raw/main/docs/images/'
        'agreement_map.png" '
        'width="500" height="180" />',
    )

    sphinx_contents = sphinx_contents.replace(
        "![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/"
        "cross_table.png)",
        '<img src="https://github.com/NOAA-OWP/gval/raw/main/docs/images/'
        'cross_table.png" '
        'width="450" height="120" />',
    )

    sphinx_contents = sphinx_contents.replace(
        "![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/"
        "metric_table.png)",
        '<img src="https://github.com/NOAA-OWP/gval/raw/main/docs/images/'
        'metric_table.png" '
        'width="700" height="180" />',
    )

    sphinx_contents = sphinx_contents.replace(
        "\nSee the full documentation [here](noaa-owp.github.io/gval/).\n", ""
    )

    with open(f"{conf.docs_dir_path}/sphinx/SPHINX_README.MD", "w") as file:
        file.write(sphinx_contents)

    shutil.copy(
        f"{conf.docs_dir_path}/../notebooks/Tutorial.ipynb",
        f"{conf.docs_dir_path}/sphinx/SphinxTutorial.ipynb",
    )

    shutil.copy(
        f"{conf.docs_dir_path}/../notebooks/Continuous Comparison Tutorial.ipynb",
        f"{conf.docs_dir_path}/sphinx/SphinxContinuousTutorial.ipynb",
    )

    shutil.copy(
        f"{conf.docs_dir_path}/../notebooks/Multi-Class Categorical Statistics.ipynb",
        f"{conf.docs_dir_path}/sphinx/SphinxMulticatTutorial.ipynb",
    )

    shutil.copy(
        f"{conf.docs_dir_path}/../notebooks/Subsampling Tutorial.ipynb",
        f"{conf.docs_dir_path}/sphinx/SphinxSubsamplingTutorial.ipynb",
    )

    shutil.copy(
        f"{conf.docs_dir_path}/../notebooks/Catalog Tutorial.ipynb",
        f"{conf.docs_dir_path}/sphinx/SphinxCatalogTutorial.ipynb",
    )

    shutil.copy(
        f"{conf.docs_dir_path}/../CONTRIBUTING.MD",
        f"{conf.docs_dir_path}/sphinx/SPHINX_CONTRIBUTING.MD",
    )


if __name__ == "__main__":
    compile_readme()
