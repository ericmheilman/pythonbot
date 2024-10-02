git filter-branch --force --index-filter \
"git rm --cached --ignore-unmatch scripts/API-KEY.sh" \
--prune-empty --tag-name-filter cat -- --all

rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force --all

