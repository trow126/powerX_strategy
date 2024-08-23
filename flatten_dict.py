def flatten_dict(nested_dict, prefix=''):
    flattened_dict = {}
    for key, value in nested_dict.items():
        new_key = prefix + key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key + '.'))
        else:
            flattened_dict[new_key] = value
    return flattened_dict
