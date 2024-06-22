class BaseLayer: #will be inherited by every layer
    def __init__(self):
        self.trainable = False
        #Optionally, you can add other members like a default weights parameter, which might
        #come in handy.