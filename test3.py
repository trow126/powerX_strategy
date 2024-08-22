def compare_dict_keys(dict1, dict2):
    return sorted(dict1.keys()) == sorted(dict2.keys())

def compare_dict_values(dict1, dict2):
    if not compare_dict_keys(dict1, dict2):
        return False
    
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not compare_dict_values(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    
    return True

def compare_dicts(dict1, dict2):
    return compare_dict_keys(dict1, dict2) and compare_dict_values(dict1, dict2)
