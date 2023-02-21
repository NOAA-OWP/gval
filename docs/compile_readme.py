import subprocess
import os
import re


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
        contents = contents.replace("../../images", "./images")
        contents = contents.replace("\\_", "_")
        contents = contents.replace("\\*", "*")

        matches = re.findall("<code>[^>]*>[^~]*?", contents)
        print(matches)

        for match in matches:
            contents = contents.replace(match, match.replace(" ", "&nbsp;"))

    with open(f"{abs_path}/../README.MD", "w") as file:
        file.write(contents)


if __name__ == "__main__":
    compile_readme()
