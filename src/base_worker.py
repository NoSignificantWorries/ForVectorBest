class BaseWorker:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("Not implemented 'call' method")
    
    def verify(self):
        raise NotImplementedError("Not implemented 'verify' method")
    
    def save_call(self):
        raise NotImplementedError("Not implemented 'save_call' method")
