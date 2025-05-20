import requests


# NB: This was used to get old releases of veropt on zenodo. Future releases should be uploaded automatically
#   - If you need to use this script 1) fill in the token and 2) choose a release from the retrieved list


repo = "aster-stoustrup/veropt"
token = ""  # Fill in token from github settings (it's under webhooks)

headers = {"Accept": "application/vnd.github.v3+json"}

repo_response = requests.get(
    url=f"https://api.github.com/repos/{repo}",
    headers=headers
)
release_response = requests.get(
    url=f"https://api.github.com/repos/{repo}/releases",
    headers=headers
)

# Choose which release to trigger from this list (0 gets the latest)
chosen_release = release_response.json()[0]

payload = {"action": "published", "release": chosen_release, "repository": repo_response.json()}

submit_response = requests.post(
    url=f"https://zenodo.org/api/hooks/receivers/github/events/?access_token={token}",
    json=payload
)

print(submit_response)
