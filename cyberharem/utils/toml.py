import re

import toml
from hbutils.design import SingletonMark
from hbutils.string import singular_form

NOT_EXIST = SingletonMark('NOT_EXIST')
IGNORE = SingletonMark('IGNORE')


def _raw_dict_merge(obj1, obj2):
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        retval = {}
        keys = sorted(set(obj1.keys()) | set(obj2.keys()))
        for key in keys:
            if key not in obj1:
                retval[key] = obj2[key]
            elif key not in obj2:
                retval[key] = obj1[key]
            else:
                retval[key] = _raw_dict_merge(obj1[key], obj2[key])
        return retval

    else:
        if obj2 is IGNORE:
            return obj1
        else:
            return obj2


def _remove_not_exist(obj1):
    if isinstance(obj1, dict):
        retval = {}
        for key, value in obj1.items():
            if value is not NOT_EXIST:
                retval[key] = _remove_not_exist(value)
        return retval

    else:
        return obj1


def dict_merge(*objs: dict):
    retval = {}
    for obj in objs:
        retval = _raw_dict_merge(retval, obj)
    return _remove_not_exist(retval)


def create_safe_toml(toml_file: str, dst_toml_file: str):
    data = toml.load(toml_file)

    def _recursion(d):
        if isinstance(d, dict):
            retval = {}
            for key, value in d.items():
                words = list(map(singular_form, filter(bool, re.split(r'[\W_]', key.lower()))))
                if any(word in {'path', 'file', 'dir', 'directory', 'weight'} for word in words) \
                        and isinstance(value, str):
                    retval[key] = '******'
                else:
                    retval[key] = _recursion(value)
            return retval

        else:
            return d

    data = _recursion(data)
    with open(dst_toml_file, 'w') as f:
        toml.dump(data, f)
