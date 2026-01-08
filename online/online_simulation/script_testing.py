import subprocess

# Full path to the Python interpreter in your Conda environment
python_executable = "/Users/awaisali/.conda/envs/deep_generator/bin/python"  # Update as necessary

# List of commands to execute
commands = [
#     "./dg_predictiction.py -c ConsultaDataMining201618 -b \"ConsultaDataMining201618.h5\" -v \"am\" -r 1 -e \"ConsultaDataMining201618.csv\"",
#
#     "./dg_predictiction.py -c ConsultaDataMining201618 -b \"ConsultaDataMining201618.h5\" -v \"rc\" -r 1 -e \"ConsultaDataMining201618.csv\"",
#
#     "./dg_predictiction.py -c ConsultaDataMining201618 -b \"ConsultaDataMining201618.h5\" -v \"da\" -r 1 -e \"ConsultaDataMining201618.csv\"",
#
# "./dg_predictiction.py -c cvs_pharmacy -b \"cvs_pharmacy.h5\" -v \"da\" -r 1 -e \"cvs_pharmacy.csv\"",
# "./dg_predictiction.py -c cvs_pharmacy -b \"cvs_pharmacy.h5\" -v \"rc\" -r 1 -e \"cvs_pharmacy.csv\"",
# "./dg_predictiction.py -c cvs_pharmacy -b \"cvs_pharmacy.h5\" -v \"am\" -r 1 -e \"cvs_pharmacy.csv\"",
#
# "./dg_predictiction.py -c PurchasingExample -b \"PurchasingExample.h5\" -v \"da\" -r 1 -e \"PurchasingExample.csv\"",
# "./dg_predictiction.py -c PurchasingExample -b \"PurchasingExample.h5\" -v \"rc\" -r 1 -e \"PurchasingExample.csv\"",
# "./dg_predictiction.py -c PurchasingExample -b \"PurchasingExample.h5\" -v \"am\" -r 1 -e \"PurchasingExample.csv\"",

#     "./dg_predictiction.py -c confidential_1000 -b \"confidential_1000.h5\" -v \"am\" -r 1 -e \"confidential_1000.csv\"",
#
#     "./dg_predictiction.py -c confidential_1000 -b \"confidential_1000.h5\" -v \"rc\" -r 1 -e \"confidential_1000.csv\"",
#
#     "./dg_predictiction.py -c confidential_1000 -b \"confidential_1000.h5\" -v \"da\" -r 1 -e \"confidential_1000.csv\"",
#
# "./dg_predictiction.py -c confidential_2000 -b \"confidential_2000.h5\" -v \"am\" -r 1 -e \"confidential_2000.csv\"",
#
#     "./dg_predictiction.py -c confidential_2000 -b \"confidential_2000.h5\" -v \"rc\" -r 1 -e \"confidential_2000.csv\"",
#
#     "./dg_predictiction.py -c confidential_2000 -b \"confidential_2000.h5\" -v \"da\" -r 1 -e \"confidential_2000.csv\"",

"./dg_predictiction.py -c insurance -b \"insurance.h5\" -v \"da\" -r 1 -e \"insurance.csv\"",
"./dg_predictiction.py -c insurance -b \"insurance.h5\" -v \"rc\" -r 1 -e \"insurance.csv\"",
"./dg_predictiction.py -c insurance -b \"insurance.h5\" -v \"am\" -r 1 -e \"insurance.csv\"",


]

# Execute each command
for command in commands:
    full_command = f"{python_executable} {command}"  # Use the Conda environment's Python interpreter
    print(f"Executing: {full_command}")
    try:
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")

print("All commands executed.")
