import subprocess

def getRepositoryPath():

    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).split()[0]