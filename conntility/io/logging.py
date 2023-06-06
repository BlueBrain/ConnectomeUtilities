# SPDX-License-Identifier: Apache-2.0
"""
Contributed by Vishal Sood
Last changed: 2021/11/29
"""
import os
from pathlib import Path
import logging, logging.config

logging.basicConfig(level=logging.NOTSET)
ERROR_FORMAT = ("%(levelname)s at %(asctime)s in %(funcName)s in %(pathname) "
                "at line %(lineno)d: %(message)s")
INFO_FORMAT = " %(asctime)s: %(message)s"
DEBUG_FORMAT = "%(lineno)d in %(pathname)s at %(asctime)s: %(message)s"
LOG_CONFIG = {'version':1,
                'formatters':{'error':{'format': ERROR_FORMAT},
                              "info": {"format": INFO_FORMAT},
                              'debug':{'format': DEBUG_FORMAT}},
                'handlers':{'console':{'class': 'logging.StreamHandler',
                                        'formatter': 'info',
                                        'level': logging.INFO},
                            'file':{'class': 'logging.FileHandler',
                                    'filename': Path.cwd()/"topology_analysis.log",
                                    'formatter': 'error',
                                    'level': logging.ERROR}},
                'root':{'handlers':('console', 'file')}}

logging.config.dictConfig(LOG_CONFIG)


def get_logger(for_step, at_level=None):
    """Centralized logging for ConnSense Apps."""
    logger = logging.getLogger(f"ConnSense {for_step.upper()}")
    logger.setLevel(at_level if at_level
                    else os.environ.get("LOGLEVEL", logging.INFO))
    return logger
