import subprocess
import shutil

def run_squeue(user=None):
    """
    Run `squeue` and return its stdout as a string.
    - user: set to a username (default: current user); pass None for all users.
    """
    if shutil.which("squeue") is None:
        raise RuntimeError("squeue not found on PATH. Is Slurm installed on this machine?")

    # Format: JOBID PARTITION NAME USER STATE TIME LIMIT NODES CPUS REASON NODELIST
    fmt = "%9i %10P %30j %10u %8T %10M %10l %4D %4C %20R %N"
    sort = "-t,+M"  # running first, then by elapsed time

    cmd = ["squeue", "-o", fmt, "--sort", sort]
    if user:  # limit to the given user
        cmd += ["-u", user]
    else:
        # If you want current user only by default, replace this branch with:
        # cmd += ["-u", os.getlogin()]
        cmd += ["--me"]
        pass

    # You can add "-h" to hide headers when parsing:
    # cmd.append("-h")

    try:
        res = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            timeout=10
        )
        return res.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"squeue failed (exit {e.returncode}): {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired:
        raise TimeoutError("squeue call timed out")
