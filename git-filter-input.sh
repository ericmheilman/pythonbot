import os
import subprocess

# Define the relative path to the sensitive file within the Git repository
SENSITIVE_FILE_RELATIVE_PATH = "bots/iterate-algo.py"  # Relative path
GIT_REPO_PATH = "/Users/ericheilman/python-bot"

def get_sensitive_value():
    """Prompt the user to enter the sensitive value (e.g., API key)"""
    return input("Enter the sensitive value (e.g., API key) to remove from the file: ")

def remove_sensitive_data(sensitive_value):
    """Remove sensitive information from the file."""
    print(f"Removing sensitive information from {SENSITIVE_FILE_RELATIVE_PATH}...")

    # Full path to the sensitive file
    sensitive_file_full_path = os.path.join(GIT_REPO_PATH, SENSITIVE_FILE_RELATIVE_PATH)

    # Open the file and read its contents
    with open(sensitive_file_full_path, "r") as file:
        lines = file.readlines()

    # Rewrite the file without the sensitive line
    with open(sensitive_file_full_path, "w") as file:
        for line in lines:
            if sensitive_value not in line:
                file.write(line)

    print(f"Sensitive information removed from {sensitive_file_full_path}.")

def clean_git_history():
    """Clean Git history to remove traces of the sensitive data."""
    print("Cleaning Git history...")

    # Run git filter-branch to remove the sensitive file from Git history
    os.chdir(GIT_REPO_PATH)  # Change to the repository directory

    # Use git filter-branch to clean up the history with the relative path
    subprocess.run([
        "git", "filter-branch", "--force", "--index-filter",
        f"git rm --cached --ignore-unmatch {SENSITIVE_FILE_RELATIVE_PATH}",
        "--prune-empty", "--tag-name-filter", "cat", "--", "--all"
    ], check=True)

    # Cleanup references and reflog to prevent access to old history
    subprocess.run(["rm", "-rf", ".git/refs/original/"], check=True)
    subprocess.run(["git", "reflog", "expire", "--expire=now", "--all"], check=True)
    subprocess.run(["git", "gc", "--prune=now", "--aggressive"], check=True)

    print("Git history cleaned.")

def force_push_clean_repo():
    """Force-push the cleaned repository back to GitHub."""
    print("Force-pushing the cleaned repository to GitHub...")
    
    # Force-push the cleaned repository
    subprocess.run(["git", "push", "--force", "origin", "main"], check=True)

    print("Repository force-pushed successfully.")

def main():
    """Main function to execute the script."""
    # Pull the sensitive value from stdin
    sensitive_value = get_sensitive_value()

    # Remove sensitive data from the file
    remove_sensitive_data(sensitive_value)
    
    # Clean the Git history
    clean_git_history()
    
    # Force-push the cleaned repository to GitHub
    force_push_clean_repo()

if __name__ == "__main__":
    main()

