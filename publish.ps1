# Publish: snapshot everything as a single commit that replaces all history, then force-push the
# gh-pages branch of tum-pbs/stocbench; the push triggers the GitHub Pages deploy.
param($m = 'Update site')
Set-Location $PSScriptRoot
git add -A
git reset -q --soft (git commit-tree (git write-tree) -m $m)
git push -f origin gh-pages
