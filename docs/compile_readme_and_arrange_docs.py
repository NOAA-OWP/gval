import subprocess
import os
import re
import shutil


def compile_readme():
    abs_path = os.path.dirname(os.path.abspath(__file__))

    if os.name == "nt":
        ret = subprocess.call(
            "where /q pandoc || ECHO Could not find app. && EXIT /B", shell=True
        )
        ret_code = 0 if ret != "Could not find app." else 1
    else:
        ret_code = subprocess.call("command -v pandoc", shell=True)

    if ret_code != 0:
        raise "No Pandoc installed"

    subprocess.call(
        f"pandoc -f gfm -t gfm {abs_path}/markdown/*.MD >" + f"{abs_path}/../README.MD",
        shell=True,
    )

    contents = None
    with open(f"{abs_path}/../README.MD", "r") as file:
        contents = file.read()
        contents = contents.replace(
            "../images", "https://github.com/NOAA-OWP/gval/raw/main/docs/images"
        )
        contents = contents.replace("\\_", "_")
        contents = contents.replace("\\*", "*")
        contents = contents.replace("  -", "-")

        matches = re.findall("<code>[^>]*>[^~]*?", contents)
        print(matches)

        for match in matches:
            contents = contents.replace(match, match.replace(" ", "&nbsp;"))

    with open(f"{abs_path}/../README.MD", "w") as file:
        file.write(contents)

    # For Sphinx documentation
    sphinx_contents = contents.replace(
        "![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/"
        "gval_dark_mode.png#gh-dark-mode-only)",
        "",
    )

    with open(f"{abs_path}/sphinx/PYPI_README.MD", "w") as file:
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

    with open(f"{abs_path}/sphinx/SPHINX_README.MD", "w") as file:
        file.write(sphinx_contents)

    shutil.copy(
        f"{abs_path}/../notebooks/Tutorial.ipynb",
        f"{abs_path}/sphinx/SphinxTutorial.ipynb",
    )

    shutil.copy(
        f"{abs_path}/../notebooks/Continuous Comparison Tutorial.ipynb",
        f"{abs_path}/sphinx/SphinxContinuousTutorial.ipynb",
    )

    shutil.copy(
        f"{abs_path}/../notebooks/Multi-Class Categorical Statistics.ipynb",
        f"{abs_path}/sphinx/SphinxMulticatTutorial.ipynb",
    )

    shutil.copy(
        f"{abs_path}/../notebooks/Subsampling Tutorial.ipynb",
        f"{abs_path}/sphinx/SphinxSubsamplingTutorial.ipynb",
    )

    shutil.copy(
        f"{abs_path}/../notebooks/Catalog Tutorial.ipynb",
        f"{abs_path}/sphinx/SphinxCatalogTutorial.ipynb",
    )

    shutil.copy(
        f"{abs_path}/../CONTRIBUTING.MD",
        f"{abs_path}/sphinx/SPHINX_CONTRIBUTING.MD",
    )


if __name__ == "__main__":
    compile_readme()
