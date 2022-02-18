from time import time
from functools import wraps

def timing(f):
    """
     Create a decorator to time functions
    :param f: functions:
    :return:
    """
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print(f'Func:{f.__name__} took: {end-start:.2f} seconds')
        return result
    return wrap
