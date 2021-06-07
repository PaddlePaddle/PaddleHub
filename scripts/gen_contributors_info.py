import json
import requests

headers = {'Accept': 'application/vnd.github.v3+json'}
org = 'PaddlePaddle'
repo = 'PaddleHub'
url = f'https://api.github.com/repos/{org}/{repo}/contributors'

page_id = 1
contributors = []
while True:
    params = {'per_page': 100, 'page': page_id}
    response = requests.get(url, headers=headers, params=params).text

    _result = json.loads(response)
    if not _result:
        break

    contributors += _result
    page_id += 1

print('<p align="center">')
for _c in contributors:
    avatar = _c['avatar_url']
    homepage = _c['html_url']
    username = _c['login']
    print(f'    <a href="{homepage}"><img src="{avatar}" width=75 height=75></a>')

print('</p>')
