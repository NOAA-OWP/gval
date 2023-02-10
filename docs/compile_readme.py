import subprocess
import os

abs_path = os.path.dirname(os.path.abspath(__file__))

subprocess.call(f"pandoc -f gfm -t gfm {abs_path}/markdown/*.MD > {abs_path}/../README.MD", shell=True)

contents = None
with open(f"{abs_path}/../README.MD", "r") as file:

    contents = file.read()
    contents = contents.replace('../../images', './images')
    contents = contents.replace('\\_', '_')
    contents = contents.replace('\\*', '*')

with open(f"{abs_path}/../README.MD", "w") as file:

    file.write(contents)


