# How to issue an xDEM release

## The easy way

1. Change the version number in setup.py. It can be easily done from GitHub directly without a PR. The version number is important for PyPI as it will determine the file name of the wheel. A name can [never be reused](https://pypi.org/help/#file-name-reuse), even if a file or project have been deleted.

2. Follow the steps to [create a new release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) on GitHub.
Use the same release number and tag as in setup.py.

An automatic GitHub action will start to push and publish the new release to PyPI.

**Note**: A tag and a release can easily be deleted if doing a mistake, but if the release is pushed to PyPI with a new version number, it will not be possible to re-use the same version number anymore.

**In short, if you mess up a release by forgetting to change the version number**:

- PyPI will block the upload, so the GitHub action failed. All is fine.
- You can now edit the version number on the main branch.
- Before releasing, you need to delete **both** the tag and the release of the previous release. If you release with the same tag without deletion, it will ignore your commit changing the version number, and PyPI will block the upload again. You're stuck in a circle. 

## The hard way

 1. Go to your local main repository (not the fork) and ensure your master branch is synced:
       git checkout master
       git pull
 2. Look over whats-new.rst and the docs. Make sure "What's New" is complete
    (check the date!) and add a brief summary note describing the release at the
    top.
 3. If you have any doubts, run the full test suite one final time!
      pytest --run-slow --mpl .
 4. On the master branch, commit the release in git:
      git commit -a -m 'Release v1.X.Y'
 5. Tag the release:
      git tag -a v1.X.Y -m 'v1.X.Y'
 6. Build source and binary wheels for pypi:
      git clean -xdf  # this deletes all uncommited changes!
      python setup.py bdist_wheel sdist
 7. Use twine to register and upload the release on pypi. Be careful, you can't
    take this back!
      twine upload dist/xdem-1.X.Y*
    You will need to be listed as a package owner at
    https://pypi.python.org/pypi/xdem for this to work.
 8. Push your changes to master:
      git push origin master
      git push origin --tags
 9. Update the stable branch (used by ReadTheDocs) and switch back to master:
      git checkout stable
      git rebase master
      git push origin stable
      git checkout master
    It's OK to force push to 'stable' if necessary.
    We also update the stable branch with `git cherrypick` for documentation
    only fixes that apply the current released version.
10. Add a section for the next release (v.X.(Y+1)) to doc/whats-new.rst.
11. Commit your changes and push to master again:
      git commit -a -m 'Revert to dev version'
      git push origin master
    You're done pushing to master!
12. Issue the release on GitHub. Click on "Draft a new release" at
    https://github.com/xdem/releases. Type in the version number, but
    don't bother to describe it -- we maintain that on the docs instead.
13. Update the docs. Login to https://readthedocs.org/projects/xdem/versions/
    and switch your new release tag (at the bottom) from "Inactive" to "Active".
    It should now build automatically.
14. Issue the release announcement!
