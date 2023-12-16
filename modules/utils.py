import re
import numpy as np
from apps.neural_network import settings


def seperate_string_number(string: str):
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i

        previous_character = i

        if x == len(string) - 2:
            groups.append(newword)
            newword = ''
    return groups


def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def snake_to_camel(snake_dict):
    camel_dict = {}
    for key in snake_dict.keys():
        camel_key = ''.join(word.capitalize() for word in key.split('_'))
        camel_dict[camel_key] = snake_dict[key]
    return camel_dict


def snake_to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def snake_dict_to_camel(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = snake_dict_to_camel(v)
            elif isinstance(v, list):
                v = [snake_dict_to_camel(item) if isinstance(
                    item, dict) else item for item in v]
            new_dict[snake_to_camel_case(k)] = v
        return new_dict
    else:
        return d


def partition_dataset(data):
    x, y = [], []
    input_sequence_length = settings.INPUT_SEQUENCE
    output_sequence_length = 20  # Single value prediction

    for i in range(input_sequence_length, len(data) - output_sequence_length):
        # Extract the input sequence (x)
        x.append(data[i - input_sequence_length:i])

        # Extract the output sequence (y)
        # Assuming price is in the first column
        y.append(data[i:i + output_sequence_length, 0])

    # Convert x and y to NumPy arrays
    x = np.array(x)
    y = np.array(y)

    return x, y


def get_ticker_for_symbol(symbol) -> any:
    for currency in settings.SUPPORTED_CURRENCIES:
        if currency.symbol == symbol or currency.debug_ticker == symbol:
            return currency
