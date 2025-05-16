class BaseWorker:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("Not implemented 'call' method")
    
    def visualize(self):
        raise NotImplementedError("Not implemented 'visualize' method")
