import json
import requests

headers = {'Accept': 'application/vnd.github.v3+json'}
org = 'PaddlePaddle'
repo = 'PaddleHub'
url = f'https://api.github.com/repos/{org}/{repo}/contributors'

contributors = requests.get(url, headers=headers).text
contributors = json.loads(contributors)

_all_contributorsrc = {
    'projectName': 'PaddleHub',
    'projectOwner': 'PaddlePaddle',
    'files': ['README.md'],
    'imageSize': 100,
    'contributors': []
}
print('<p align="center">')
for _c in contributors:
    avatar = _c['avatar_url']
    homepage = _c['html_url']
    username = _c['login']
    _all_contributorsrc['contributors'].append({
        'login': username,
        'avatar_url': avatar,
        'profile': homepage,
        'name': username,
        'contributions': []
    })
    print(f'    <a href="{homepage}"><img src="{avatar}" width=75 height=75></a>')

print('</p>')
