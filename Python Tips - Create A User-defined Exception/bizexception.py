# Define a businenss logic exception

class BizException(Exception):
    __code = 'E0001'
    __message = 'A business logic exception'

    @property
    def code(self):
        return self.__code

    @property
    def message(self):
        return self.__message

    def getcode(self):
        return self.__code

    def getmessage(self):
        return self.__message

    def __str__(self):
        return self.__code + ': ' + self.__message

    @code.setter
    def code(self, code):
        self.__code = code

    @message.setter
    def message(self, message):
        self.__message = message

    @message.getter
    def message(self):
        return self.__message

    """
    __code = 'class variable: E0001'
    __message = 'class variable: A business logic exception'
 
    def __init__(self):
        self.__code = 'E0001'
        self.__message = 'A business logic exception'
           
    def __init__(self, message):
        assert isinstance(message, str)
        self._message = message
    """

class SubBizException(BizException):

    def __init__(self):
        # self.message = 'message defined in the super class'
        self._submessage = 'a sub class inherits from MyException'

    @property
    def submessage(self):
        return self._submessage

    @submessage.setter
    def submessage(self, submessage):
        self._submessage = submessage

    def submethod(self):

        print('submethod: do something in sub class', self.getmessage())