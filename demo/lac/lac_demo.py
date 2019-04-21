import os
import paddlehub as hub

if __name__ == "__main__":
    # Load LAC Module
    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]

    # Set input dict
    inputs = {"text": test_text}

    # execute predict and print the result
    results = lac.lexical_analysis(data=inputs)
    for result in results:
        print(result['word'])
        print(result['tag'])
