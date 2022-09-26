import logging
import config


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def print_log_print(print_words,logger1):
    print(print_words)
    logger1.info(print_words)

def print_hyperpar(logger2):
    print_log_print('Batch_size:  '+str(config.batch_size),logger2)
    print_log_print('Loss_function:  '+config.lossname,logger2)
    print_log_print('Learning_rate:  '+str(config.learning_rate),logger2)
    print_log_print('Model_name:  '+config.modelname,logger2)
    print_log_print('Distillation alpha:  '+str(config.distillation_alpha),logger2)
    print_log_print('Label_smoothing alpha:  '+str(1-config.label_smooothing_alpha),logger2)
    print_log_print('Vit_head:  '+str(config.vit_head),logger2)
    print_log_print('Vit_depth:  '+str(config.depth),logger2)
