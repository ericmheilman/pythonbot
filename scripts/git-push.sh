#!/bin/bash
cat /Users/ericheilman/t.txt

# Prompt for the GitHub username
echo -n "Enter your GitHub username: "
read username

# Prompt for the GitHub Personal Access Token (hidden input)
echo -n "Enter your GitHub Personal Access Token: "
read -s token

# Prompt for a commit message
echo -n "Enter commit message: "
read commit_message

# Add all changes to the staging area
git add .

# Commit the changes with the provided message
git commit -m "$commit_message"

# Set the remote URL with the provided username and token (token is not stored in plain text)
git remote set-url origin https://$username:$token@github.com/$username/pythonbot.git

# Push to origin/main branch
git push -u origin main
