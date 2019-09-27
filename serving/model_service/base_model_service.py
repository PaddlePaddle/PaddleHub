import six
import abc


class BaseModelService(object):

    def _initialize(self):
        pass

    @abc.abstractmethod
    def _pre_processing(self, data):
        pass

    @abc.abstractmethod
    def _inference(self, data):
        pass

    @abc.abstractmethod
    def _post_processing(self, data):
        pass

