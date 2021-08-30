$PROJECT = 'rever'
$ACTIVITIES = [
              'changelog',  # Uses files in the news folder to create a changelog for release
              'tag',  # Creates a tag for the new version number
              'push_tag',  # Pushes the tag up to the $TAG_REMOTE
              'ghrelease'  # Creates a Github release entry for the new tag
              'pypi',  # Sends the package to pypi
               ]
$CHANGELOG_FILENAME = 'CHANGELOG.rst'  # Filename for the changelog
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'  # Filename for the news template
$PUSH_TAG_REMOTE = 'git@github.com:st3107/tomology.git'  # Repo to push tags to

$GITHUB_ORG = 'st3107'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'tomology'  # Github repo for Github releases  and conda-forge
