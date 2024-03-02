import math

def Softmax(input_):
    denominator = 0
    res = []
    for numerator in input_:
        denominator += math.exp(numerator)
    for numerator in input_:
        res.append(math.exp(numerator) / denominator)
    return res

input_ = [-4.4774e-01, -8.2948e-01, -1.9950e-01, -1.2991e-01, -7.1054e-02, 4.9257e-01,  1.2125e+00, -5.4788e-01]
res = Softmax(input_)
print(res)
