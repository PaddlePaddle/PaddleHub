import requests
import json
import argparse

try:
    from termcolor import colored, cprint
except:
    raise ImportError(
        "The module requires additional dependencies: termcolor. Please run 'pip install termcolor' to install it."
    )

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-a", type=str, default="127.0.0.1", help="the plato2_en_base serving address")
parser.add_argument("-p", type=int, default=8866, help="the plato2_en_base serving port")
args = parser.parse_args()
# yapf: enable.

headers = {"Content-type": "application/json"}
url = "http://%s:%s/predict/plato2_en_base" % (args.addr, args.port)

context = ""
start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
cprint(start_info, "yellow", attrs=["bold"])
while True:
    user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()
    if user_utt == "[EXIT]":
        break
    elif user_utt == "[NEXT]":
        context = ""
        cprint(start_info, "yellow", attrs=["bold"])
    else:
        context += user_utt
        data = {'texts': [context]}
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        bot_response = r.json()["results"]
        print(
            colored("[Bot]:", "blue", attrs=["bold"]),
            colored(bot_response, attrs=["bold"]))
        context += "\t%s" % bot_response
