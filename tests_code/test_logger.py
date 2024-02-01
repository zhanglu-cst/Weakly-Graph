import mmcv

from preprocess_data.tests.test_logger2 import func

logger = mmcv.get_logger(name = 'preprocess_data', log_file = 'test_log.log')
# logger = mmcv.

logger.info('hhhhhhh')

func()
