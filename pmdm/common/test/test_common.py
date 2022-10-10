from concurrent.futures import Executor, Future


class FakeExecutor(Executor):
    def submit(self, f, *args, **kwargs):
        future = Future()
        future.set_result(f(*args, **kwargs))
        return future

    def shutdown(self, wait=True):
        pass
