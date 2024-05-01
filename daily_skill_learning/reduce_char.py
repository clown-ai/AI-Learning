from functools import reduce
str_line = "sdfghjkldsdfghjjhgfdsdfnmngfdfhgfdssdfghjkljhgfd"

result = reduce(lambda x, y: {**x, y: x.get(y, 0) + 1}, str_line, {})
dict_sorted_res = dict(sorted(result.items()))
freq_sorted_res = dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))
print(result)
print(dict_sorted_res)
print(freq_sorted_res)
