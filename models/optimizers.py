class DummyOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass
    
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass
