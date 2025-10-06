class PositionPublisher:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def publish(self, event):
        for cb in self.subscribers:
            cb(event)