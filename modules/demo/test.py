import paddlehub as hub

senta_test = hub.Module(directory="senta_test")
print(senta_test.sentiment_classify(["这部电影太糟糕了", "这部电影太棒了"]))
