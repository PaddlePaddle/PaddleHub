import paddlehub as hub

module = hub.Module(name="ernie_gen_leave")

test_texts = ["理由"]
results = module.generate(texts=test_texts, use_gpu=False, beam_width=2)
for result in results:
    print(result)