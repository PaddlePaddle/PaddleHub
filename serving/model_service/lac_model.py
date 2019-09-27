from serving.model_service.text_model_service import TextModelService
import paddlehub as hub


class LacModelService(TextModelService):
    def __init__(self):
        self.model = None
        self._initialize()

    def _initialize(self):
        self.model = hub.Module("lac")

    def _pre_processing(self, data):
        # print("在这里进行处理吧，唉～")
        # print(data)
        # print(type(data["text"][0]))
        # print(data["text"])
        # print(type(data["text"]))
        data["text"] = data["text"].encode("utf-8")
        data["text"] = [data["text"]]
        # for index, value in enumerate(data["text"]):
        #     if isinstance(value, unicode):
        #         print(data["text"][index])
        #         print(type(data["text"][index]))
        #         data["text"][index] = data["text"][index].encode("utf-8")
        return data

    def _inference(self, data):
        data = self._pre_processing(data)
        print(data)
        results = self.model.lexical_analysis(data=data)
        results = self._post_processing(results)
        return results

    def _post_processing(self, results):
        for i in range(len(results)):
            for key, value in results[i].items():
                for j in range(len(results[i][key])):
                    results[i][key][j] = results[i][key][j].encode("utf-8")
        return results

    def inference(self, data):
        return self._inference(data)


lac_module_service = LacModelService.instance()
