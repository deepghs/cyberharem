from hbutils.design import SingletonMark

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
