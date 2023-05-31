# """
# BATTERY OF TESTS TO VERIFY THE INTEGRITY AND FUNCTIONALITY OF PYQCM
# """

import unittest
import os

TEST_FILES = []
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) # gets current path of script

if not os.path.exists(f"{SCRIPT_DIR}/test_outputs"): # checks if the output dir exists and if not creates it
    os.mkdir(f"{SCRIPT_DIR}/test_outputs")

os.chdir(f"{SCRIPT_DIR}/test_outputs") # changes working dir to the test output dir

for file in os.listdir(f"{SCRIPT_DIR}/test_files"): # finds all test files in the testing directory following the pattern test***********.py
    if file[:4]=="test" and file[-3:]==".py":
        TEST_FILES.append(file)

def run_file(test_name):
    """Runs a file of a given name inside the testing directory. Mostly aesthetic.
    """
    return os.system(f"python3 {SCRIPT_DIR}/test_files/{test_name}")


class TestAll(unittest.TestCase):

    # generates the appropriate methods given the detected test files
    for file in TEST_FILES:
        exec(f"""def {file[:-3]}(self): \n\tself.assertFalse(run_file("{file}"))""")

if __name__ == "__main__":
    unittest.main()