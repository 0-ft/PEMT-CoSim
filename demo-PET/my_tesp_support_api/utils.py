class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = lambda self, key: DotDict(self.get(key)) if type(self.get(key)) is dict else self.get(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
