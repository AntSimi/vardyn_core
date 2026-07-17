"""
Loads and merges experiment configuration files.
"""

from copy import deepcopy
import os
import sys


class Config(dict):

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __str__(self):
        res = []
        for key, val in self.items():
            res.append(f"{key}: {val}")
        res.append("")
        return "\n".join(res)

    def copy(self):
        other = {}
        for key, val in self.items():
            if type(val) == dict:
                other[key] = deepcopy(val)
            else:
                other[key] = val
        return Config(other)


def _merge_configs(config_exp, config_def, NAME):

    dict_exp = config_exp.get(NAME)
    dict_def = config_def.get(dict_exp["super"])

    _config = {}
    _config["super"] = dict_exp["super"]
    for key in dict_def:
        if key in dict_exp:
            # Check if we have encapsulated super class
            if type(dict_exp[key]) == dict and "super" in dict_exp[key]:
                # If yes, we merge with default config
                _dict_def = getattr(config_def, dict_exp[key]["super"])
                for _key in _dict_def:
                    if _key not in dict_exp[key]:
                        dict_exp[key][_key] = _dict_def[_key]
            _config[key] = dict_exp[key]
        else:
            _config[key] = dict_def[key]

    return Config(_config)


def merge_configs(config_exp, config_def, name):
    NAME = config_exp.get(name)
    if type(NAME) == list:
        config = {}
        for _NAME in NAME:
            config[_NAME] = _merge_configs(config_exp, config_def, _NAME)
        config = Config(config)
    else:
        config = _merge_configs(config_exp, config_def, NAME)
    return config


def Exp(config_exp):
    """
    NAME
        exp

    DESCRIPTION
        Read experimental parameters
        Args:
            path_config (str): path (dir+name) of the configuration file.`
        Returns:
            configuration module
    """
    # Merge with default config file
    from .config_default import default as config_def

    config = Config({})

    # EXP
    config.EXP = {}
    for key, val in config_exp["EXP"].items():
        config.EXP[key] = val
    for key, val in config_def["EXP"].items():
        if key not in config.EXP:
            config.EXP[key] = val
    config.EXP = Config(config.EXP)
    config.GRID = merge_configs(config_exp, config_def, "NAME_GRID")
    config.OBS = merge_configs(config_exp, config_def, "NAME_OBS")
    config.MOD = merge_configs(config_exp, config_def, "NAME_MOD")
    config.OBSOP = merge_configs(config_exp, config_def, "NAME_OBSOP")
    config.BASIS = merge_configs(config_exp, config_def, "NAME_BASIS")
    config.INV = merge_configs(config_exp, config_def, "NAME_INV")
    if "NAME_DIAG" in config_exp:
        config.DIAG = merge_configs(config_exp, config_def, "NAME_DIAG")

    del config_exp

    return config
