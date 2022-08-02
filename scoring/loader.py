import os
import sys
import traceback


class LoaderError(Exception):
    """
    Desc:
        Customize the Exception
    """

    def __init__(self, err_message):
        Exception.__init__(self, err_message)


class Loader(object):
    """
    Desc:
        Dynamically Load a Module
    """

    def __init__(self):
        """ """

    @staticmethod
    def load(path):
        """
        Args:
            path to the script
        """
        try:
            items = os.path.split(path)
            sys.path.append(os.path.join(*items[:-1]))
            ip_module = __import__(items[-1][:-3])
            return ip_module
        except Exception as error:
            traceback.print_exc()
            raise LoaderError("IMPORT ERROR: {}, load module [path: {}]!".format(error, path))
