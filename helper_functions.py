def check_type(input, type_array):
    if len(type_array)>0:
        truth_array = [isinstance(input, types) for types in type_array]

    else:
        truth_array = [isinstance(input, type_array)]

    return True in truth_array
