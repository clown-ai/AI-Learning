my_dict = {
    'a': [1, 2, 3],
    'b': [4, 5],
    'c': [6, 7, 8, 9],
    'd': [10],
    'e': [2, 2, 2, 2]
}

result = sorted(my_dict.items(), key=lambda item: (len(item[1]), item[0]))
print(result)
