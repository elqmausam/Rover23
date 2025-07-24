import multiprocessing
import p
import cone3
n = int(input())

def run(runfile):
    with open(runfile, "r") as rnf:
        exec(rnf.read())

if n == 1:
    run("cone3.py")

else:
    print("p.py")
  