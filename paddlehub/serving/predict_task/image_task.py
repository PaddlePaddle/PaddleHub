from paddlehub.serving.predict_task.basic_task import BasicTask


class ImageTask(BasicTask):
    def __init__(self, id, data):
        super(ImageTask, self).__init__(id, data)
        self.id = id
        self.data = data
