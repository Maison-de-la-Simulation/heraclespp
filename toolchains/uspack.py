#!/usr/bin/env python3

"""
uspack
"""

import argparse
import json
import pathlib
import subprocess
import tempfile

def build_and_install(env, project_dir, install_dir, cmake_options):
    """Build and install a CMake project."""
    with tempfile.TemporaryDirectory() as build_dir:
        print(f"Using temporary build directory: {build_dir}")
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as script_file:
            script_file.write(f"{generate_environment(env)}\n"
                              f"cmake {' '.join(f'-D {cmake_option}' for cmake_option in cmake_options)} -B {build_dir} -S {project_dir}\n"
                              f"cmake --build {build_dir}\n"
                              f"cmake --install {build_dir} --prefix {str(install_dir)}\n")
            script_file.flush()
            subprocess.run(["bash", script_file.name], check=True)

def install_all_projects(env, projects, install_dir):
    """
    Install all specified CMake projects and generate a script for Package_ROOT variables.

    :param projects: List of tuples (project_path, package_name, cmake_options, commands)
    :param install_dir: Directory where all projects will be installed
    """
    install_path = pathlib.Path(install_dir).resolve()
    print(f"Installing all projects to: {install_path}")

    env["variables"] = env.get("variables", {})
    env["commands"] = env.get("commands", [])

    for project, spec in projects.items():
        build_system = spec["build_system"]
        provider = spec["provider"]
        with tempfile.TemporaryDirectory() as project_path:
            print(f"Installing project '{project}' at: {project_path}")
            if provider["type"] == "git":
                clone_cmd = ["git", "-c", "advice.detachedHead=false", "clone", "--depth", "1", provider["url"], project_path]
                if "version" in provider.keys():
                    clone_cmd.extend(["--branch", provider["version"]])
                subprocess.run(clone_cmd, check=True)
            else:
                raise RuntimeError(f"Unknown provider: {provider['type']}")

            package_install_dir = install_path / project

            if build_system["type"] == "cmake":
                cmake_options = build_system.get("options", [])
                cmake_package = build_system["package"]
                build_and_install(env, project_path, package_install_dir, cmake_options)
                env["variables"][f"{cmake_package}_ROOT"] = package_install_dir
            else:
                raise RuntimeError(f"Unknown build system: {build_system['type']}")

            if "commands" in spec:
                env["commands"] += spec["commands"]

    # Generate the script for Package_ROOT variables
    script_path = install_path / "activate.sh"
    print(f"Generating Package_ROOT script at: {script_path}")
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write(generate_environment(env))
    print("All projects installed successfully.")

def generate_environment(environment):
    """
    Generate a bash script to set up a base environment with modules, environment variables,
    and custom commands.

    This function takes a dictionary `environment` containing environment configuration details
    and generates a bash script as a string. The generated script includes commands to load 
    specified modules, export environment variables, and execute additional custom commands.

    Args:
        environment (dict): A dictionary with the following optional keys:
            - "modules" (list of str): A list of module names to load using `module load`.
            - "variables" (dict): A dictionary of environment variables to export, where the keys 
              are variable names and the values are their corresponding values.
            - "commands" (list of str): A list of additional bash commands to include in the script.

    Returns:
        str: A string representing the generated bash script.

    Example:
        >>> environment = {
        ...     "modules": ["gcc", "python"],
        ...     "variables": {"PATH": "/usr/local/bin", "DEBUG": "1"},
        ...     "commands": ["echo Environment configured!", "source setup.sh"]
        ... }
        >>> script = generate_environment(environment)
        >>> print(script)
        #!/bin/bash
        #
        # This file has been automatically generated script
        #
        module load gcc
        module load python
        export PATH=/usr/local/bin
        export DEBUG=1
        echo Environment configured!
        source setup.sh
    """

    script_name = "activate"
    guard_variable = f"{script_name.upper()}_LOADED"

    bash_script = "#!/bin/bash\n\n"
    bash_script += "# This file has been automatically generated script\n\n"

    # Add a guard to prevent sourcing the script multiple times
    bash_script += "# Safeguard to ensure this script is sourced only once\n"
    bash_script += f'if [ -n "${guard_variable}" ]; then\n'
    bash_script += f"    echo \"Script '{script_name}' already sourced. Skipping.\"\n"
    bash_script += "    return 0\n"
    bash_script += "fi\n"
    bash_script += f"export {guard_variable}=1\n\n"

    env_modules = environment.get("modules")
    if env_modules is not None:
        for module in env_modules:
            bash_script += f"module load {module}\n"
        bash_script += "\n"

    env_variables = environment.get("variables")
    if env_variables is not None:
        for key, value in env_variables.items():
            bash_script += f"export {key}={value}\n"
        bash_script += "\n"

    env_commands = environment.get("commands")
    if env_commands is not None:
        for command in env_commands:
            bash_script += f"{command}\n"

    return bash_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build suite")
    parser.add_argument("packages",
                        type=pathlib.Path,
                        help="Dependency file")
    parser.add_argument("--prefix",
                        default=pathlib.Path.cwd(),
                        type=pathlib.Path,
                        help="Install prefix directory")
    args = parser.parse_args()

    with open(args.packages, encoding="utf-8") as json_file:
        packages = json.load(json_file)

    prefix = pathlib.Path(args.prefix).resolve()

    try:
        install_all_projects(packages["environment"], packages["packages"], prefix)
    except RuntimeError as e:
        print(f"Error: {e}")
