import mmcv


class Global_Var():
    GLOBAL_VARS_DICT = {}

    @staticmethod
    def set(key, value):
        Global_Var.GLOBAL_VARS_DICT[key] = value

    @staticmethod
    def get(key):
        return Global_Var.GLOBAL_VARS_DICT[key]

    @staticmethod
    def set_logger(logger):
        Global_Var.GLOBAL_VARS_DICT['logger'] = logger

    @staticmethod
    def logger():
        if ('logger' in Global_Var.GLOBAL_VARS_DICT):
            return Global_Var.GLOBAL_VARS_DICT['logger']
        else:
            return mmcv.get_logger(name = 'default')
