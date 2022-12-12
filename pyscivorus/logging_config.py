from datetime import datetime
import logging

currentDateTime = datetime.now().strftime('%m-%d-%Y--%H:%M:%S')
log_path = f'logs/log-{currentDateTime}.log'

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'loggers': {
        'root': {
            'handlers': ['fileHandler', 'consoleHandler'],
            'level': 'DEBUG',
        }
    },
    'handlers': {
        'consoleHandler': {
            'level': 'ERROR',
            'formatter': 'standardFormatter',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        'fileHandler': {
            'level': 'DEBUG',
            'formatter': 'standardFormatter',
            'class': 'logging.FileHandler',
            'filename': log_path,
            'mode': 'a',
        }
    },
    'formatters': {
        'standardFormatter': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s :: %(funcName)s - %(message)s'
        }
    },
}

logging.config.dictConfig(LOGGING_CONFIG)