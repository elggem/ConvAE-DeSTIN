import os
import time
import theano.misc.pkl_utils as tPickle

class ResultSaver(object):
    def __init__(self):
        self.result_dir = "../results/"+str(time.time())+"/"
        if not os.path.exists(os.path.dirname(self.result_dir)):
            try:
                os.makedirs(os.path.dirname(self.result_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def dump(self, obj, filename):
        tPickle.dump(obj, open(self.result_dir+filename, "w"))