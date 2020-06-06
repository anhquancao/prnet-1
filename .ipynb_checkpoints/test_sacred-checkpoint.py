import os
import torch 

from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

SETTINGS.CAPTURE_MODE = 'sys'  # for tqdm

ex = Experiment('hello_config')
observer = MongoObserver.create(url='10.3.54.105:27017', # don't change this
                                db_name='qcao_test_sacred') # change this
ex.observers.append(observer)
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm

config = dict(message='Hello world!',
              nb_epochs=20)
ex.add_config(config)

@ex.automain
def my_main(message, nb_epochs, _run):
    print(message)
    
    # Get the job id of the cluster in case
    # we want to redo the experiment.
    _run.meta_info['container_name'] = os.environ.get('CONTAINER_NAME', 'container name')

    for epoch in range(nb_epochs):
        for metric in ('loss', 'acc'):
            _run.log_scalar(metric, epoch * 1.18, epoch)