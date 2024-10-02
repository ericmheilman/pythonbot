#!/bin/sh
git filter-branch --force --index-filter \
"git rm --cached --ignore-unmatch bots/iterate-algo.py" \
--prune-empty --tag-name-filter cat -- --all
