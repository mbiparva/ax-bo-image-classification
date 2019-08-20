import _init_lib_path

from datetime import datetime
import datetime as dt
import time

from utils.config import cfg
from epoch_loop import EpochLoop

from pprint import PrettyPrinter

from ax.service.managed_loop import optimize

pp = PrettyPrinter(indent=4)
cfg.TRAINING = True
cfg.VALIDATING = True


def set_hyperparameters(params):
    cfg.TRAIN.LR = params.get('lr', cfg.TRAIN.LR)
    cfg.TRAIN.WEIGHT_DECAY = params.get('weight_decay', cfg.TRAIN.WEIGHT_DECAY)
    cfg.TRAIN.MOMENTUM = params.get('momentum', cfg.TRAIN.MOMENTUM)


def train_evaluate(parameterization):
    set_hyperparameters(parameterization)
    epoch_loop = EpochLoop()
    epoch_loop.main()

    eval_metric = cfg.METERS[1]  # label_accuracy
    return epoch_loop.validator.meters[eval_metric].avg


def ax_optimization():
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {'name': 'lr', 'type': 'range', 'bounds': [1e-4, 1.0], 'log_scale': True},
            {'name': 'weight_decay', 'type': 'range', 'bounds': [1e-6, 1e-4]},
            {'name': 'momentum', 'type': 'range', 'bounds': [0.0, 1.0]},
        ],
        evaluation_function=train_evaluate,
        objective_name='accuracy',
    )

    pp.pprint(best_parameters)
    pp.pprint(values)
    pp.pprint(experiment)
    pp.pprint(model)


if __name__ == '__main__':

    print('configuration file cfg is loaded for training ...')
    pp.pprint(cfg)

    started_time = time.time()
    print('*** started @', datetime.now())
    ax_optimization()
    length = time.time() - started_time
    print('*** ended @', datetime.now())
    print('took', dt.timedelta(seconds=int(length)))
