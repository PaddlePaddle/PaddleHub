import paddlehub as hub


class ImageModelService(object):
    @classmethod
    def instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_module(cls, name):
        module = hub.Module(name=name)
        return module

    def _initialize(self):
        pass

    def _pre_processing(self):
        pass

    def _inference(self):
        pass

    def _post_processing(self):
        pass