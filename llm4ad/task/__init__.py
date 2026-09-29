import os
import inspect
import importlib


def import_all_evaluation_classes(root_directory):
    """
    Recursively imports all classes from 'evaluation_cvrp.py' files found in the directory structure.

    Args:
        root_directory (str): The root directory to start the search for 'evaluation_cvrp.py' files.
    """
    for dirpath, _, filenames in os.walk(root_directory):
        # Check if 'evaluation_cvrp.py' exists in the current directory
        if 'evaluation_cvrp.py' in filenames:
            # Build the module path relative to the root directory
            module_name = 'llm4ad.task.' + '.'.join(os.path.relpath(dirpath, root_directory).split(os.sep)) + '.evaluation'

            # Dynamically import the 'evaluation_cvrp.py' module
            module = importlib.import_module(module_name)

            # Iterate over the module's attributes and check if they are classes
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type):  # Only import class objects
                    # Use inspect to check if the class is defined in the current module
                    if inspect.getmodule(attribute).__file__ == module.__file__:
                        globals()[attribute_name] = attribute  # Add the class to the global namespace
                        # print(f"Imported class {attribute_name} from {module_name}")
