#!/bin/sh
git filter-branch --force --index-filter \
"git rm --cached --ignore-unmatch scripts/API-KEY.sh:2" \
--prune-empty --tag-name-filter cat -- --all
