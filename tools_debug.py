import inspect
from datetime import datetime, timezone


def jprint(*args):
    frame = inspect.currentframe().f_back
    name_file = frame.f_code.co_filename
    name_func = frame.f_code.co_name
    num_line = frame.f_lineno

    str_target = ' '.join([str(arg) for arg in args])

    print(f'[YUKAI][DEBUG][{name_func}:{num_line}] {str_target} [FILE] {name_file}')


class Timer:
    def __init__(self):
        self.time_previous = None

    def get_current_time(self):
        return datetime.now(timezone.utc).replace(microsecond=0)

    def click(self):
        if self.time_previous is None:
            self.time_previous = self.get_current_time()
            return self

        time_current = self.get_current_time()
        duration = time_current - self.time_previous
        self.time_previous = time_current
        return duration
