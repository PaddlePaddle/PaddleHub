from paddlehub.serving.predict_task.basic_task import BasicTask


class TextTask(BasicTask):
    def __init__(self, id, data):
        super(TextTask, self).__init__(id, data)
        self.id = id
        self.data = data
