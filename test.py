import importlib.util

package_name = "plotly"
if importlib.util.find_spec(package_name) is not None:
    print(f"'{package_name}' is installed.")
else:
    print(f"'{package_name}' is NOT installed. Run 'pip install {package_name}' to install it.")