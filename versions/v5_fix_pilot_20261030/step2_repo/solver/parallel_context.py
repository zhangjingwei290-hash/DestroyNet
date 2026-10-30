"""Thread-local repair dispatch; no effect on legacy/serial environments."""
import threading
from contextlib import contextmanager
_LOCAL=threading.local()
def current_client():return getattr(_LOCAL,'client',None)
@contextmanager
def use_client(client):
    old=current_client();_LOCAL.client=client
    try:yield
    finally:_LOCAL.client=old
