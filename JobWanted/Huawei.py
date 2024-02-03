# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:15:37 2022

@author: 86153
"""

while True:
    try:
        N, M = map(int, input().split(" "))
        start_score = list(map(int, input().split(" ")))
        obj_list = []
        max_list = []
    except:
        break
    else:
        for ele in range(M):
            obj = list(map(str, input().split(" ")))
            obj_list.append(obj)
        for ele in obj_list:
            for data in ele:
                A = int(ele[1])
                B = int(ele[2])
                if data == 'Q':
                    max_list = start_score[min(A, B)-1:max(A, B)]
                    print(max(max_list))
                if data == 'U':
                    start_score[A-1] = B


file_dict = dict()
err_list = []
while True:
    try:
        err = [ele for ele in input().split()]
        filename = err[0].split("\\")[-1]
        lineno = err[1]
    
        if len(filename) > 16:
            filename = filename[-16:]
        key = filename + " " + lineno
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
            err_list.append(key)
    except (EOFError, KeyboardInterrupt):
        break
for key in err_list[-8:]:
    print(key, file_dict.get(key))

    
poke_all = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'joker', 'JOKER']
poke = input().split("-")
poke_left = poke[0].split()
poke_right = poke[1].split()
if "joker JOKER" in poke:
    print("joker JOKER")
else:
    if (len(poke_left)!=4) and (len(poke_right)!=4):
        if len(poke_left) != len(poke_right):
            print("ERROR")
        else:
            if poke_all.index(poke_left[0]) > poke_all.index(poke_right[0]):
                print(poke[0])
            else:
                print(poke[1])
    elif len(poke_left)!=4 and len(poke_right)==4:
        print(poke[1])
    elif len(poke_left)==4 and len(poke_right)!=4:
        print(poke[0])
    else:
        if poke_all.index(poke_left[0]) > poke_all.index(poke_right[0]):
            print(poke[0])
        else:
            print(poke[1])
            
# HJ1:
input_str = input()
word_list = input_str.strip().split()
last_word = word_list[-1]
print(len(last_word))

# HJ2:
str1 = input().lower()
str2 = input().lower()
print(str1.count(str2))

# HJ3:
while True:
    try:
        N = input()
        input_data = []
        for i in range(int(N)):
            input_data.append(int(input()))
        unique = set(input_data)
        for j in sorted(unique):
            print(j)
    except (EOFError, KeyboardInterrupt):
        break

# HJ4：
while True:
    try:
        string = input()
        while(len(string) > 0):
            print(string[:8].ljust(8, "0"))
            string = string[8:]
    except:
        break

# HJ5:
while True:
    try:
        number = input()
        print(int(number, 16))
    except:
        break

# HJ6:
import math
N = int(input())
for i in range(2, int(math.sqrt(N)+1)):
    while N % i == 0:
        print(i, end=' ')
        N //= i
if  N > 2:
    print(N)

# HJ7:
while True:
    try:
        number = float(input())
        if number - int(number) >= 0.5:
            print(int(number+1))
        else:
            print(int(number))
    except:
        break

# HJ8:
n = int(input())
dict_list = {}
for i in range(n):
    data = input().split()
    key = int(data[0])
    value = int(data[1])
    dict_list[key] = dict_list.get(key, 0) + value
for ele in sorted(dict_list):
    print(ele, dict_list[ele])

# HJ9:
number = list(reversed(input()))
res = []
for ele in number:
    if ele in res:
        continue
    else:
        res.append(ele)
print("".join(res))

# HJ10:
def count_character(string):
    string = "".join(set(string))
    count = 0
    for ele in string:
        if 0<=ord(ele)<=127:
            count += 1
    return count
string = input()
print(count_character(string))

# HJ11:
number = input()
print("".join(reversed(number)))

# HJ12:
string = input()
print("".join(reversed(string)))

# HJ13:
sentence = input().split()[::-1]
for ele in sentence:
    print(ele, end=' ')

# HJ14:
n = int(input())
word_list = []
for i in range(n):
    word_list.append(input())
word_list.sort()
for ele in word_list:
    print(ele)
    
# HJ15:
number = int(input())
count = 0
while number > 0:
    if number % 2:
        count += 1
    number //= 2
print(count)

#number = int(input())
#print(bin(number).count("1"))

# HJ16:
N, m = map(int, input().split())
primary, annex = {}, {}
for i in range(1, m+1):
    v, p, q  = map(int, input().split())
    if q == 0: #主件
        primary[i] = [v, p]
    else:
        if q in annex: #第二附件
            annex[q].append([v, p])
        else: #第一附件
            annex[q] = [[v, p]]
m = len(primary) #主件个数转换为物品数
dp = [[0]*(N+1) for _ in range(m+1)]
w, V = [[]], [[]]
for key in primary:
    w_temp, V_temp = [], [] #对主件进行处理
    w_temp.append(primary[key][0])
    V_temp.append(primary[key][0] * primary[key][1])
    if key in annex: #存在附件
        w_temp.append(w_temp[0] + annex[key][0][0]) #附件1
        V_temp.append(V_temp[0] + annex[key][0][0] * annex[key][0][1])
        if len(annex[key]) > 1: #存在两个附件
            w_temp.append(w_temp[0] + annex[key][1][0]) #附件2
            V_temp.append(V_temp[0] + annex[key][1][0] * annex[key][1][1])
            w_temp.append(w_temp[0] + annex[key][0][0] + annex[key][1][0]) #附件1与附件2
            V_temp.append(V_temp[0] + annex[key][0][0] * annex[key][0][1] + annex[key][1][0] * annex[key][1][1])
    w.append(w_temp)
    V.append(V_temp)
for i in range(1, m+1):
    for j in range(10, N+1, 10):
        max_i = dp[i-1][j]
        for k in range(len(w[i])):
            if j - w[i][k] >= 0:
                max_i = max(max_i, dp[i-1][j-w[i][k]] + V[i][k])
        dp[i][j] = max_i
print(dp[m][N])

# HJ17
import sys
import re
x, y = 0, 0
cmd_list = sys.stdin.readline().strip().split(';')
func = {
        'A': lambda a,b,p:(a-p, b),
        'D': lambda a,b,p:(a+p, b),
        'W': lambda a,b,p:(a, b+p),
        'S': lambda a,b,p:(a, b-p)
}
for cmd in cmd_list:
    if re.search(r'^[A|D|W|S]\d\d?$', cmd) and len(cmd)<3:
        x, y = func[cmd[0]](x, y, int(cmd[1:]))
print(f'{x},{y}')

data = input().split(";")
e_cmd = []
for cmd in data:
    length = len(cmd)
    if length == 3 or length == 2: #两位数以内
        if cmd[0] in ['A', 'D', 'W', 'S'] and cmd[1:length].isdigit():
            e_cmd.append(cmd)
x, y = 0, 0
for ele in e_cmd:
    if ele[0] == 'A':
        x -= int(ele[1:len(ele)])
    if ele[0] == 'D':
        x += int(ele[1:len(ele)])
    if ele[0] == 'W':
        y += int(ele[1:len(ele)])
    if ele[0] == 'S':
        y -= int(ele[1:len(ele)])
print("%d,%d"%(x,y))

# HJ18:
import sys
res = [0,0,0,0,0,0,0]

def public(ip):
    if 1 <= ip[0] <= 126:
        res[0] += 1
    if 128 <= ip[0] <= 191:
        res[1] += 1
    if 192 <= ip[0] <= 223:
        res[2] += 1
    if 224 <= ip[0] <= 239:
        res[3] += 1
    if 240 <= ip[0] <= 255:
        res[4] += 1
    return 

def private(ip):
    if (ip[0] == 10) or (ip[0] == 172 and 16<=ip[1]<=32) or(ip[0]==192 and ip[1]==168):
        res[6] += 1
    return

def Mask(mask):
    val = (mask[0]<<24) + (mask[1]<<16) + (mask[2]<<8) + mask[3]
    if val == 0:
        return False
    if (val+1) == (1<<32):
        return False
    flag = 0
    while(val):
        digit = val & 1
        if digit == 1:
            flag = 1
        if flag == 1 and digit == 0:
            return False
        val >>= 1
    return True

def judge(line):
    ip, mask = line.strip().split("~")
    ips = [int(x) for x in filter(None, ip.split("."))]
    masks = [int(x) for x in filter(None, mask.split("."))]
    if ips[0] == 0 or ips[0] == 127:
        return
    if len(ips)<4 or len(masks)<4:
        res[5] += 1
    if Mask(masks) == True:
        public(ips)
        private(ips)
    else:
        res[5] += 1
    return
for line in sys.stdin:
    judge(line)
res = [str(x) for x in res]
print(" ".join(res))

# HJ19:
file_dict = dict()
err_list = []
while True:
    try:
        err = [ele for ele in input().split()]
        filename = err[0].split("\\")[-1]
        lineno = err[1]
    
        if len(filename) > 16:
            filename = filename[-16:]
        key = filename + " " + lineno
        if key in file_dict:
            file_dict[key] += 1
        else:
            file_dict[key] = 1
            err_list.append(key)
    except (EOFError, KeyboardInterrupt):
        break
for key in err_list[-8:]:
    print(key, file_dict.get(key))

# HJ20:
def check(s):
    if len(s) < 8:
        return 0
    a, b, c, d = 0, 0, 0, 0
    for item in s:
        if item.isdigit():
            a = 1
        elif item.islower():
            b = 1
        elif item.isupper():
            c = 1
        else:
            d = 1
    if a+b+c+d < 3:
        return 0
    for i in range(len(s)-2):
        x = s[i:i+3]
        if x in s[i+3:]:
            return 0
    return 1
while True:
    try:
        print("OK" if check(input()) else "NG")
    except:
        break

# HJ21
while True:
    try:
        string = input()
        key = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        result = ""
        for item in string:
            if item.isalpha():
                if item.isupper():
                    if item == 'Z':
                        result += 'a'
                    else:
                        result += chr(ord(item.lower())+1)
                else:
                    for ele in key:
                        if item in ele:
                            result += str(key.index(ele)+2)
                            break
            else:
                result += item
        print(result)
    except:
        break

# HJ22
while True:
    n = int(input())
    if n == 0: break
    print(n // 2)

def f(n):
    if n == 0: return 0
    if n == 1: return 0
    if n >= 2: return f(n-2) + 1
while True:
    try:
        n = int(input())
        if n!=0:
            print(f(n))
    except(EOFError, KeyboardInterrupt):
        break

# HJ23:
while True:
    try:
        string = input()
        dic, result = {}, ""
        for item in string:
            if item not in dic:
                dic[item] = 1
            else:
                dic[item] += 1
        Min = min(dic.values())
        for ele in string:
            if dic[ele] != Min:
                result += ele
        print(result)
    except:
        break

# HJ24:
import bisect
def Seq(seq):
    dp = [1] * len(seq)
    arr = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] > arr[-1]:
            arr.append(seq[i])
            dp[i] = len(arr)
        else:
            pos = bisect.bisect_left(arr, seq[i])
            arr[pos] = seq[i]
            dp[i] = pos+1
    return dp
while True:
    try:
        N = int(input())
        seq = list(map(int, input().split()))
        dp1, dp2 = Seq(seq), Seq(seq[::-1])[::-1]
        res = []
        for i in range(len(seq)):
            res.append(dp1[i] + dp2[i] - 1)
        print(N - max(res))
    except:
        break

# HJ25
while True:
    try:
        I = input().split()
        R = input().split()
        I_len = int(I[0])
        I.remove(I[0])
        R_len = int(R[0])
        R.remove(R[0])
        R = list(set(R))
        for i in range(len(R)):
            R[i] = int(R[i])
        R.sort()
        for i in range(len(R)):
            R[i] = str(R[i])
        res = []
        
        for i in range(len(R)):
            num = 0
            for j in range(len(I)):
                if (R[i] in I[j]):
                    num += 1
            if num == 0:
                continue
            res.append(R[i])
            res.append(str(num))
            for j in range(len(I)):
                if (R[i] in I[j]):
                    res.append(str(j))
                    res.append(I[j])
        res.insert(0, str(len(res)))
        print(' '.join(res))
    except:
        break
    
# HJ26
while True:
    try:
        string = input()
        temp = ""
        for item in string:
            if item.isalpha():
                temp += item
        temp = sorted(temp, key=str.upper)
        index = 0
        res = ""
        for i in range(len(string)):
            if string[i].isalpha():  #是字母就去排好序的temp中取
                res += temp[index]
                index += 1
            else:                   #不是数字则从string中直接加入
                res += string[i]
        print(res)
    except:
        break
    
# HJ27
while True:
    try:
        string = input().split()
        total = string[0]
        key = string[1:-2]
        input_key = string[-2]
        input_count = string[-1]
        count = 0
        temp = []
    
        for item in key:
            if item == input_key:
                continue
            elif sorted(item) == sorted(input_key):
                temp.append(item)
                count += 1
        print(count)
        res = sorted(temp)
        print(res[int(input_count)-1])
    except:
        break

# HJ28
import math
def is_prime(num):
    if num == 1:
        return False
    for i in range(2, int(math.sqrt(num)+1)):
        if num%i == 0:
            return False
    return True
def match(odd, evens, visit, choose):
    for v, even in enumerate(evens):
        if is_prime(odd+even) and visit[v] == 0:
            visit[v] = 1
            if choose[v]==0 or match(choose[v], evens, visit, choose):
                choose[v] = odd
                return True
    return False
while True:
    try:
        n = int(input())
        input_num = list(map(int, input().split()))
        count = 0
        odds, evens = [], []
        for ele in input_num:
            if ele%2 == 0:
                evens.append(ele)
            else:
                odds.append(ele)
        choose = [0] * len(evens)
        for odd in odds:
            visit = [0] * len(evens)
            if match(odd, evens, visit, choose):
                count += 1
        print(count)
    except:
        break
    
# HJ29
def encrypt(s1):
    a = list(s1)
    for i in range(len(a)):
        if a[i].isupper():
            if a[i] == 'Z':
                a[i] = 'a'
            else:
                a[i] = a[i].lower()
                c = ord(a[i]) + 1
                a[i] = chr(c)
        elif a[i].islower():
            if a[i] == 'z':
                a[i] = 'A'
            else:
                a[i] = a[i].upper()
                c = ord(a[i]) + 1
                a[i] = chr(c)
        elif a[i].isdigit():
            if a[i] == '9':
                a[i] = '0'
            else:
                a[i] = int(a[i]) + 1
                a[i] = str(a[i])
        else:
            a[i] = a[i]
    return a
def decrypt(s2):
    b = list(s2)
    for i in range(len(b)):
        if b[i].isupper():
            if b[i] == 'A':
                b[i] = 'z'
            else:
                b[i] = b[i].lower()
                c = ord(b[i]) - 1
                b[i] = chr(c)
        elif b[i].islower():
            if b[i] == 'a':
                b[i] = 'Z'
            else:
                b[i] = b[i].upper()
                c = ord(b[i]) - 1
                b[i] = chr(c)
        elif b[i].isdigit():
            if b[i] == '0':
                b[i] = '9'
            else:
                b[i] = int(b[i]) - 1
                b[i] = str(b[i])
        else:
            b[i] = b[i]
    return b
while True:
    try:
        s1 = input()
        s2 = input()
        res1 = encrypt(s1)
        res2 = decrypt(s2)
        print("".join(res1))
        print("".join(res2))
    except:
        break

# HJ30
import re
def Encrypt(x):
    if re.search(r'[0-9A-Fa-f]', x):
        return hex(int(bin(int(x, 16))[2:].rjust(4, '0')[::-1], 2))[2:].upper()
    else:
        return x
while True:
    try:
        s = list(input().replace(" ", ""))
        res = ""
        s[::2] = sorted(s[::2])
        s[1::2] = sorted(s[1::2])
        for ele in s:
            res += Encrypt(ele)
        print(res)
    except:
        break

while True:
    try:
        s = list(input().replace(" ", ""))
        s[0::2] = sorted(s[0::2])
        s[1::2] = sorted(s[1::2])
        for ele in s:
            if '0'<=ele<='9' or 'a'<=ele<='f' or 'A'<=ele<='F':
                ele = bin(int(ele, 16))[2:].rjust(4, '0') #左端补0
                ele = ele[::-1]
                ele = hex(int(ele, 2))[2:].upper()
                print(ele, end='')
            else:
                print(ele, end='')
    except (EOFError, KeyboardInterrupt):
        break
    
# HJ31
while True:
    try:
        string = input()
        for ele in string:
            if not ele.isalpha():
                string = string.replace(ele, " ")
        temp = string.split()[::-1]
        for ele in temp:
            print(ele, end=' ')
    except:
        break

# HJ32
out_len = 0
while True:
    try:
        s = input()
        for i in range(len(s)):
            j = i - 1
            k = i + 1
            len_ABA = 1
            while j>=0 and k<len(s):
                if s[j] == s[k]:
                    j -= 1
                    k += 1
                    len_ABA += 2
                else:
                    break
            
            m = i
            n = i+1
            len_ABBA = 0
            while m>=0 and n<len(s):
                if s[m] == s[n]:
                    m -= 1
                    n += 1
                    len_ABBA += 2
                else:
                    break
                
            cur_len = max(len_ABA, len_ABBA)
            if out_len < cur_len:
                out_len = cur_len
        print(out_len)
    except:
        break

# HJ33
def ip2int(s):
    s = s.split(".")
    res = ""
    for ele in s:
        res += bin(int(ele))[2:].rjust(8, '0')
    return int('0b'+res, 2)
def int2ip(num):
    res = ""
    bin_str = bin(int(num))[2:].rjust(32, '0')
    for i in range(4):
        res += str(int('0b'+bin_str[i*8:i*8+8],2)) + '.'
    return res[:-1]
while True:
    try:
        print(ip2int(input()))
        print(int2ip(input()))
    except:
        break

# HJ34
while True:
    try:
        s = list(input())
        for i in range(len(s)):
            s[i] = ord(s[i])
        s.sort()
        for i in range(len(s)):
            s[i] = chr(s[i])
        print("".join(s))
    except:
        break
    
# HJ35
while True:
    try:
        N = int(input())
        for i in range(1, N+1):
            for j in range(i, N+1):
                print(int((j+j*j)/2-i+1), end=" ")
            print()
    except(EOFError, KeyboardInterrupt):
        break
    
# HJ36
while True:
    try:
        key = list(input())
        input_str = list(input())
        new_key = []
        for i in key:
            if i not in new_key:
                new_key.append(i)
        for i in range(ord('a'), ord('z')+1):
            if chr(i) not in new_key:
                new_key.append(chr(i))
        dic = {}
        j = 0
        for i in range(ord('a'), ord('z')+1):
            dic[chr(i)] = new_key[j]
            j += 1
        for i in range(len(input_str)):
            input_str[i] = dic[input_str[i]]
        print("".join(input_str))
    except:
        break

# HJ37
while True:
    try:
        month = int(input())
        n = month - 1
        def fun(n):
            if n<2:
                return 1
            else:
                return fun(n-1) + fun(n-2)
        print(fun(n))
    except:
        break
# HJ38
while True:
    try:
        x = int(input())
        total = 0-x
        for i in range(5):
            total += x*2
            x /= 2
        print(float(total))
        print(float(x))
    except:
        break

# HJ39
while True:
    try:
        mask = input().split(".")
        ip1 = input().split(".")
        ip2 = input().split(".")
        res1, res2 = [], []
        
        for i in range(4):
            mask[i] = int(mask[i])
            ip1[i] = int(ip1[i])
            ip2[i] = int(ip2[i])
        
        if mask[0]!=255 or mask[3]!=0 or max(mask+ip1+ip2)>255 or min(mask+ip1+ip2)<0:
            print("1")
            break
        else:
            for i in range(4):
                temp1 = int(bin(int(mask[i]))[2:]) & int(bin(int(ip1[i]))[2:])
                res1.append(temp1)
                temp2 = int(bin(int(mask[i]))[2:]) & int(bin(int(ip2[i]))[2:])
                res2.append(temp2)
            if res1 == res2:
                print("0")
                break
            else:
                print("2")
                break
    except:
        break

# HJ40
while True:
    try:
        s = input()
        res = [0, 0, 0, 0]
        for ele in s:
            if ele.isalpha():
                res[0] += 1
            if ele.isspace():
                res[1] += 1
            if ele.isdigit():
                res[2] += 1
        res[3] = len(s) - (res[0]+res[1]+res[2])
        for i in range(4):
            print(res[i])
    except:
        break

# HJ41 称砝码
while True:
    try:
        n = int(input())
        m = list(map(int, input().split()))
        x = list(map(int, input().split()))
        weights = {0,}
        amount = []
        for i in range(n):
            for j in range(x[i]):
                amount.append(m[i])
        for i in amount:
            for j in list(weights):
                weights.add(i+j)
        print(len(weights))
    except:
        break

# HJ42 数字翻译成英文
unit1 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
         'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
unit2 = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
def lessHundred(n):
    if n>0:
        if n<20:
            res.append(unit1[n])
        else:
            res.append(unit2[n//10])
            if n%10 != 0:
                res.append(unit1[n%10])
def lessThousand(n):
    if n>=100:
        res.append(unit1[n//100])
        res.append('hundred')
        if n%100 != 0:
            res.append('and')
    lessHundred(n%100)
while True:
    try:
        num = int(input())
        res = []
        a = num % 1000
        b = (num//1000) % 1000
        c = (num//1000000) % 1000
        d = num //1000000000
        
        if d>0:
            lessThousand(d)
            res.append('billion')
        if c>0:
            lessThousand(c)
            res.append('million')
        if b>0:
            lessThousand(b)
            res.append('thousand')
        if a>0:
            lessThousand(a)
        print(" ".join(res))
    except:
        break

# HJ43 以迷宫，当前坐标，路径与终点为参数进行 DFS 求解
def dfs(matrix, x, y, path, dst):
    if (x,y) == dst:
        path.append((x,y))
        for i in path:
            print("({},{})".format(i[0],i[1]))
        return
     
     
    if not 0<=x<len(matrix) or not 0<=y<len(matrix[0]) or matrix[x][y] == 1 or (x,y) in path:
        return
     
    path.append((x,y))
    dfs(matrix, x+1, y, path, dst)
    dfs(matrix, x, y+1, path, dst)
    dfs(matrix, x-1, y, path, dst)
    dfs(matrix, x, y-1, path, dst)
    path.pop()
while True:
    try:
        x, y = list(map(int, input().split()))
        dst = (x-1, y-1)
        matrix = []
        for i in range(x):
            matrix.append(list(map(int, input().split())))
        dfs(matrix, 0, 0, [], dst)
         
    except:
        break

# HJ45  统计字母的出现次数，然后以权重降低赋予
while True:
    try:
        N = int(input())
        for i in range(N):
            word = input()
            dic = {}
            for ele in word:
                if ele not in dic:
                    dic[ele] = 1
                else:
                    dic[ele] += 1
            temp = sorted(dic.values(), reverse=True)
            ans = 0
            m = 0
            for ele in temp:
                ans += (26-m)*ele
                m += 1
            print(ans)
    except:
        break

# HJ46
while True:
    try:
        string = list(input())
        N = int(input())
        print("".join(string[:N]))
    except:
        break

# HJ48 从单向链表中删除指定值的节点
while True:
    try:
        param = list(map(int, input().split()))
        N = param[0]        # 结点个数
        head = param[1]     # 头结点
        delete = param[-1]  # 要删除的值
        data = param[2:-1]  # 待插入结点
        res = [head]
        
        for i in range(0, len(data), 2):
            b = data[i+1]
            a = data[i]
            res.insert(res.index(b)+1, a)
        res.remove(delete)
        for i in range(len(res)):
            res[i] = str(res[i])
        print(' '.join(res))
    except:
        break
# HJ50
def group(s):
    num, res = '', []
    for i, c in enumerate(s):
        if c.isdigit():
            num += c # 数字可能有很多位数
        else:
            if num:
                res.append(num)
                num = ''
            if c == '-': # 负数的判断
                if (i == 0) or (s[i-1] in '+-*/([{'):
                    num += c
                    continue
            res.append(c)
    if num:
        res.append(num)
    return res
 
while True:
    try:
        s = input()
        List = group(s)
        stack_num, stack_op = [], []
        '''
            遍历数字和符号列表lst：
            1.如果遇到数字，添加到数字栈stack_n中；
            2.如果遇到*/([{这些符号，直接添加到符号栈stack_op中；
            3.如果遇到+-号:
                (1).如果符号栈stack_op为空或栈顶元素是左括号([{的话，直接入栈；
                (2).如果符号栈stack_op不为空，则不断从符号栈stack_op中弹出一个符号，
                    同时从数字栈stack_n中弹出两个数字进行运算，并将运算结果保存到数字栈stack_n中。
                    期间若遇到左括号([{，则跳出循环，最后再将加号+或者减号-添加到符号栈中。
            4.如果遇到右括号)]}，在栈顶元素不是左括号([{之前，不断地取出数字和符号进行运算，
              同时将结果保存到数字栈stack_n中，最后删除左括号。
        '''
        for ele in List:
            if ele not in '+-*/()[]{}': # 数字
                stack_num.append(ele)
            elif ele in '*/([{':
                stack_op.append(ele)
            elif ele in '+-':
                if len(stack_op) == 0 or stack_op[-1] in '([{':
                    stack_op.append(ele)
                else:
                    while stack_op:
                        if stack_op[-1] in '([{':
                            break
                        op = stack_op.pop()
                        n2, n1 = stack_num.pop(), stack_num.pop()
                        stack_num.append(str(eval(n1 + op + n2)))
                    stack_op.append(ele)
            elif ele in ')]}':
                while stack_op[-1] not in '([{':
                    op = stack_op.pop()
                    n2, n1 = stack_num.pop(), stack_num.pop()
                    stack_num.append(str(int(eval(n1 + op + n2))))
                stack_op.pop()
        # 对数字栈和符号栈中剩余元素进行运算
        while stack_op:
            op = stack_op.pop()
            n2, n1 = stack_num.pop(), stack_num.pop()
            stack_num.append(str(int(eval(n1 + op + n2))))
        # 弹出并打印数字栈中最后一个数字，即运算结果
        print(stack_num.pop())
    except:
        break
    
while True:
    try:
        exp = input()
        exp = exp.replace('{', '(').replace('}', ')').replace('[', '(').replace(']', ')')
        print(int(eval(exp)))
    except:
        break

# HJ51 输出单向链表倒数第 K 个节点
class Node(object):
    def __init__(self, val=0):
        self.val = val
        self.next = None
while True:
    try:
        head = Node()
        count, num_list, k = int(input()), list(map(int, input().split())), int(input())
        while k:
            head.next = Node(num_list.pop())
            head = head.next
            k -= 1
        print(head.val)
    except:
        break

# HJ52 计算字符串的编辑距离
while True:
    try:
        str1 = input()
        str2 = input()
        m = len(str1)
        n = len(str2)
        dp = [[0 for i in range(m+1)] for j in range(n+1)]
        for i in range(m+1):
            dp[0][i] = i
        for j in range(n+1):
            dp[j][0] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                if str1[j-1] == str2[i-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1
        print(dp[n][m])
    except:
        break

# HJ53
#while True:
#   try:
#       n = int(input())
#        arr = [[] for i in range(n)]
#        arr[0] = 1
#        arr[1] = [1, 1, 1]
#        m = 2
#        while m < n:
#            column = m*2+1
#            for i in range(column):
#                arr[m].append(sum(arr[m-1][i-2 if i-2>=0 else 0:i+1])
#            m += 1
while True:
    try:
        n = int(input())
        if n <= 2:
            print("-1")
        elif (n-1)%2 == 0:
            print("2")
        elif n%4 == 0:
            print("3")
        elif (n-2)%4 == 0:
            print("4")
    except:
        break

# HJ54
def group(s):
    num, res = "", []
    for i, c in enumerate(s):
        if c.isdigit():
            num += c
        else:
            if num:
                res.append(num)
                num = ""
            if c == '-':
                if i==0 or s[i-1] in "+-*/(":
                    num += c
                    continue
            res.append(c)
    if num:
        res.append(num)
    return res
while True:
    try:
        s = input()
        List = group(s)
        stack_num, stack_op = [], []
        for ele in List:
            if ele not in "+-*/()":
                stack_num.append(ele)
            elif ele in "*/(":
                stack_op.append(ele)
            elif ele in "+-":
                if len(stack_op)==0 or stack_op[-1] in "(":
                    stack_op.append(ele)
                else:
                    while stack_op:
                        if stack_op[-1] in "(":
                            break
                        op = stack_op.pop()
                        num2, num1 = stack_num.pop(), stack_num.pop()
                        stack_num.append(str(eval(num1+op+num2)))
                    stack_op.append(ele)
            elif ele in ")":
                while stack_op[-1] not in "(":
                    op  = stack_op.pop()
                    num2, num1 = stack_num.pop(), stack_num.pop()
                    stack_num.append(str(int(eval(num1+op+num2))))
                stack_op.pop()
        while stack_op:
            op = stack_op.pop()
            num2, num1 = stack_num.pop(), stack_num.pop()
            stack_num.append(str(int(eval(num1+op+num2))))
        print(stack_num.pop())
    except:
        break
        
# HJ55
while True:
    try:
        n = int(input())
        count = 0
        for i in range(1,n+1):
            if i%7 == 0:
                count += 1
            elif str(i).count("7")>0:
                count += 1
        print(count)
    except:
        break

# HJ56
while True:
    try:
        n = int(input())
        count = 0
        for i in range(1, n):
            res = []
            for j in range(1, i):
                if i%j == 0:
                    res.append(j)
            if sum(res) == i:
                    count += 1
        print(count)
    except:
        break
        
# HJ57
while True:
    try:
        str1 = list(map(int, input()))[::-1]
        str2 = list(map(int, input()))[::-1]
        res = ""
        i = 0
        carry = 0
        Sum = 0
        while i < max(len(str1), len(str2)):
            a = 0 if i >= len(str1) else str1[i]
            b = 0 if i >= len(str2) else str2[i]
            Sum = (carry + a+b) % 10
            carry = (carry + a+b) // 10
            res = str(Sum) + res
            i += 1
        if carry > 0:
            res = "1" + res
        print(res)
    except:
        break

# HJ58
import heapq
while True:
    try:
        n, k = list(map(int, input().split()))
        num = list(map(int, input().split()))
        min_heap = []
        for ele in num:
            if len(min_heap) < k:
                heapq.heappush(min_heap, -ele)
            else:
                if min_heap and min_heap[0] < -ele:
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap, -ele)
            res = [-i for i in min_heap]
        temp = sorted(res)
        print(" ".join(str(i) for i in temp))
    except:
        break

# HJ59
while True:
    try:
        string = input()
        res = []
        for ele in string:
            if string.count(ele) == 1:
                print(ele)
                break
        else:
            print("-1")
    except:
        break

# HJ60
def isPrime(num):
    if num == 1:
        return False
    for i in range(2, int(math.sqrt(num)+1)):
        if num%i == 0:
            return False
    return True
    
while True:
    try:
        n = int(input())
        for i in range(int(n/2), n):
            if isPrime(i) and isPrime(n-i):
                print(n-i)
                print(i)
                break
    except:
        break

# HJ61
while True:
    try:
        m, n = map(int, input().split())
        dp = [[0 for i in range(n+1)] for j in range(m+1)]
        for i in range(m+1):
            dp[i][1] = 1
        for j in range(1, n+1):
            dp[0][j] = 1
            dp[1][j] = 1
        for i in range(2, m+1):
            for j in range(2, n+1):
                if i<j:
                    dp[i][j] = dp[i][i]
                else:
                    dp[i][j] = dp[i-j][j] + dp[i][j-1]
        print(dp[m][n])
    except:
        break

# HJ62
while True:
    try:
        n = int(input())
        binary = bin(n)[2:]
        temp = str(binary)
        count = temp.count("1")
        print(count)
    except:
        break

# HJ63
while True:
    try:
        DNA = input()
        N = int(input())
        temp = []
        res = []
        for i in range(len(DNA)):
            s = DNA[i:i+N].count("G") + DNA[i:i+N].count("C")
            temp.append(DNA[i:i+N])
            res.append(s/N)
        x = max(res)
        print(temp[res.index(x)])
    except:
        break

# HJ64
while True:
    try:
        N = int(input())
        order = input()
        if N <= 4:
            cur = 1
            for ele in order:
                if ele == 'U':
                    if cur == 1:
                        cur = N
                    else:
                        cur -= 1
                if ele == 'D':
                    if cur == 4:
                        cur = 1
                    else:
                        cur += 1
            print(" ".join([str(i) for i in range(1, N+1)]))
            print(cur)
        else:
            cur = 1
            screen_ptr = 1 #当前屏幕的第一首歌曲
            for ele in order:
                if ele == 'U':
                    if cur == 1:
                        cur = N
                        screen_ptr = N-3
                    elif cur == screen_ptr:
                        cur -= 1
                        screen_ptr -= 1
                    else:
                        cur -= 1
                if ele == 'D':
                    if cur == N:
                        cur = 1
                        screen_ptr = 1
                    elif cur == screen_ptr+3:
                        cur += 1
                        screen_ptr += 1
                    else:
                        cur += 1
            print(" ".join([str(i) for i in range(screen_ptr, screen_ptr+4)]))
            print(cur)
    except:
        break
    
# HJ65
while True:
    try:
        short, long = input(), input()
        if len(short) > len(long):
            short, long = long, short
        res = ""
        for i in range(len(short)):
            for j in range(i+1, len(short)):
                if short[i:j+1] in long and j+1-i>len(res):
                    res = short[i:j+1]
        print(res)
    except:
        break
while True:
    try:
        short, long = input(), input()
        if len(short) > len(long):
            short, long = long, short
        m, n = len(short), len(long)
        dp = [[0 for i in range(n+1)] for j in range(m+1)]
        max_sub_len = 0
        ptr = 0
        for i in range(m):
            for j in range(n):
                if short[i] == long[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                    if dp[i+1][j+1] > max_sub_len:
                        max_sub_len = dp[i+1][j+1]
                        ptr = i
        print(short[ptr+1-max_sub_len:ptr+1])
        print(max_sub_len)
    except:
        break

# HJ66
while True:
    try:
        command = input().split()
        key = ['reset', 'reset board', 'board add', 'board delete', 'reboot backplane', 'backplane abort']
        value = ['reset what', 'board fault', 'where to add', 'no board at all', 'impossible', 'install first']
        if len(command)<1 or len(command)>2:
            print("unknown command")
        elif len(command)==1:
            if command[0] == key[0][:len(command[0])]:
                print(value[0])
            else:
                print("unknown command")
        else:
            index = []
            for i in range(1, len(key)):
                temp = key[i].split()
                if command[0] == temp[0][:len(command[0])] and command[1] == temp[1][:len(command[1])]:
                    index.append(i)
            if len(index) != 1:
                print("unknown command")
            else:
                print(value[index[0]])
    except:
        break

# HJ67
def util(numArray, item):
    if item<1:
        return False
    if len(numArray) == 1:
        return numArray[0] == item
    else:
        for i in range(len(numArray)):
            m = numArray[0:i] + numArray[i+1:]
            n = numArray[i]
            if util(m, item+n) or util(m, item-n) or util(m, item*n) or util(m, item/n):
                return True
        return False
while True:
    try:
        if util(list(map(int, input().split())), 24):
            print("true")
        else:
            print("false")
    except:
        break
    
# HJ68
while True:
    try:
        n = int(input())
        flag = int(input())
        res = []
        for _ in range(n):
            name, score = input().split()
            res.append((name, int(score)))
        if flag:
            for i in sorted(res, key=lambda x:x[1]):
                print(i[0], i[1])
        else:
            for i in sorted(res, key=lambda x:x[1], reverse=True):
                print(i[0], i[1])
    except:
        break

# HJ69
while True:
    try:
        x = int(input())
        y = int(input())
        z = int(input())
        A = []
        B = []
        
        for i in range(x):
            A.append(list(map(int, input().split())))
        for j in range(y):
            B.append(list(map(int, input().split())))
        R = [[0 for k in range(z)] for i in range(x)]
        for i in range(x):
            for k in range(z):
                for j in range(y):
                    R[i][k] += A[i][j] * B[j][k]
        for i in range(x):
            for k in range(z):
                print(R[i][k], end=' ')
            print('')
    except:
        break
        
# HJ70
while True:
    try:
        n = int(input())
        arr = []
        order = []
        res = 0
        for i in range(n):
            arr.append(list(map(int, input().split())))
        f = input()
        for ele in f:
            if ele.isalpha():
                order.append(arr[ord(ele)-65])
            elif ele == ')' or len(order)>=2:
                    b = order.pop()
                    a = order.pop()
                    res += a[1]*b[1]*a[0]
                    order.append([a[0], b[1]])
        print(res)
    except:
        break

# HJ71
while True:
    try:
        normal = input()
        string = input()
        m = len(normal)
        n = len(string)
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, m+1):
            if normal[i-1] == '*':
                dp[i][0] = True
            else:
                break
        for i in range(1, m+1):
            for j in range(1, n+1):
                if normal[i-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif normal[i-1] == '?' and string[j-1].isalnum():
                    dp[i][j] = dp[i-1][j-1]
                elif normal[i-1].lower() == string[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]
        if dp[m][n]:
            print("true")
        else:
            print("false")
    except:
        break

# HJ72
while True:
    try:
        start = input()
        for i in range(4):
            a = 4*i
            b = 25 - 7*i
            c = 100 - a - b
            print(a,b,c)
    except:
        break

# HJ73
while True:
    try:
        y, m, d = map(int, input().split())
        month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if y%400==0 or (y%4==0 and y%100!=0):
            month[1] = 29
        print(sum(month[:m-1])+d)
    except:
        break

# HJ74
while True:
    try:
        string = input().replace(' ', '\n') #避免空格的混淆，实施替换
        res = ""
        flag = False
        for ele in string:
            if ele == '"':
                flag = not flag
            elif flag==True and ele=='\n':
                res += ' '
            else:
                res += ele
        temp = res.count('\n')+1
        print(temp)
        print(res)
    except:
        break

# HJ75
while True:
    try:
        str1 = input()
        str2 = input()
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        Max = 0
        for i in range(len(str1)):
            for j in range(i, len(str1)):
                if str1[i:j+1] in str2 and j+1-i>Max:
                    Max = j+1-i
        print(Max)
    except:
        break

while True:
    try:
        str1 = input()
        str2 = input()
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        m, n = len(str1), len(str2)
        max_sub_len = 0
        dp = [[0 for i in range(n+1)] for j in range(m+1)]
        for i in range(m):
            for j in range(n):
                if str1[i] == str2[j]:
                    dp[i+1][j+1] = dp[i][j]+1
                    if dp[i+1][j+1] > max_sub_len:
                        max_sub_len = dp[i+1][j+1]
        print(max_sub_len)
    except:
        break

# HJ76
while True:
    try:
        n = int(input())
        res = [str(i) for i in range(n*(n-1), n*(n+1)) if i%2==1]
        print("+".join(res))
    except:
        break

# HJ77
res = []
def dfs_train(wait, stack, out):
    if not wait and not stack:
        res.append(' '.join(map(str, out)))
    if wait:
        dfs_train(wait[1:], stack+[wait[0]], out)
    if stack:
        dfs_train(wait, stack[:-1], out+[stack[-1]])
while True:
    try:
        N, nums = input(), list(map(int, input().split()))
        dfs_train(nums, [], [])
        for ele in sorted(res):
            print(ele)
    except:
        break

# HJ78
while True:
    try:
        N, list1 = input(), list(map(int, input().split()))
        M, list2 = input(), list(map(int, input().split()))
        List = list1 + list2
        List = sorted((set(List)))
        List.sort()
        res = ''.join(list(map(str, List)))
        print(res)
    except:
        break

# HJ81
while True:
    try:
        short, long = input(), input()
        for ele in short:
            if ele not in long:
                print("false")
                break
        else:
            print("true")
    except:
        break
    
# HJ82
while True:
    try:
        a, b = map(int, input().split("/"))
        a *= 10
        b *= 10
        res = []
        while a:
            for i in range(a, 0, -1):
                if b%i==0:
                    res.append(f'1/{int(b/i)}')
                    a -= i
                    break
        print('+'.join(res))
    except:
        break

# HJ83
while True:
    try:
        m, n = map(int, input().split())
        x1, y1, x2, y2 = map(int, input().split())
        insert_x = int(input())
        insert_y = int(input())
        x, y = map(int, input().split())
        if 0<=m<=9 and 0<=n<=9:
            print("0")
        else:
            print("-1")
        if 0<=x1<m and 0<=y1<n and 0<=x2<=m and 0<=y2<=n:
            print("0")
        else:
            print("-1")
        if 0<=insert_x<m and m<9:
            print("0")
        else:
            print("-1")
        if 0<=insert_y<n and n<9:
            print("0")
        else:
            print("-1")
        if 0<=x<m and 0<=y<n:
            print("0")
        else:
            print("-1")
    except:
        break

# HJ84
while True:
    try:
        string = input()
        count = 0
        for ele in string:
            if ele.isalpha() and ele.isupper():
                count += 1
        print(count)
    except:
        break
        
# HJ85 最长回文子串
while True: # 左右指针进行扫描
    try:
        string = input()
        max_len = 0
        for i in range(len(string)):
            for j in range(len(string)-1, i-1, -1):
                if string[i:j+1] == string[i:j+1][::-1]:
                    if j+1-i > max_len:
                        max_len = j+1-i
                        res = string[i:j+1]
        print(max_len)
        print(res)
    except:
        break

while True: # 放置动态规划矩阵
    try:
        string = input()
        max_len = 1
        N = len(string)
        if N < 2:
            print(string)
        dp = [[False for _ in range(N)] for _ in range(N)]
        for i in range(N):
            dp[i][i] = True
        begin = 0
        for L in range(2, N+1): # 子串长度的搜索
            for i in range(N): # j+1-i = L
                j = i+L-1 #右边界
                if j>=N:
                    break
                if string[i] != string[j]:
                    dp[i][j] = False
                else:
                    if j-i<3: # 此时i与j相等
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1] # i往左搜索，j往右搜索
                if dp[i][j] and j+1-i>max_len:
                    max_len = j+1-i
                    begin = i
        print(string[begin:begin+max_len])
        print(max_len)
    except:
        break

# HJ86
while True:
    try:
        num = str(bin(int(input()))[2:]).split('0')
        res = []
        for ele in num:
            res.append(len(ele))
        print(max(res))
    except:
        break
    
# HJ87
while True:
    try:
        string = input()
        sign = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
        score = 0
        if len(string)<=4:
            score += 5
        elif 5<=len(string)<=7:
            score += 10
        else:
            score += 25
            
        lower = 0
        upper = 0
        digit = 0
        signer = 0
        for ele in string:
            if ele.islower():
                lower += 1
            elif ele.isupper():
                upper += 1
            elif ele.isdigit():
                digit += 1
            else:
                signer += 1
        if lower == 0 and upper == 0:
            score += 0
        elif lower != 0 and upper != 0:
            score += 20
        else:
            score += 10
            
        if digit == 0:
            score += 0
        elif digit == 1:
            score += 10
        else:
            score += 20

        if signer == 0:
            score += 0
        elif signer == 1:
            score += 10
        else:
            score += 25
        
        if lower>0 and upper>0 and digit>0 and signer>0:
            score += 5
        elif (lower>0 and digit>0) or (upper>0 and digit>0):
            score += 2
        else:
            score += 3
        
        if score >= 90:
            print("VERY_SECURE")
        if 80 <= score < 90:
            print('SECURE')
        if 70 <= score < 80:
            print('VERY_STRONG')
        if 60 <= score < 70:
            print('STRONG')
        if 50 <= score < 60:
            print('AVERAGE')
        if 25 <= score < 50:
            print('WEAK')
        if 0 <= score < 25:
            print('VERY_WEAK')
    except:
        break

# HJ88
while True:
    try:
        poke_all = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'joker', 'JOKER']
        string = input().split("-")
        left = string[0].split()
        right = string[1].split()
        if "joker JOKER" in string:
            print("joker JOKER")
        else:
            if len(left)!=4 and len(right)!=4:
                if len(left) != len(right):
                    print("ERROR")
                else:
                    if poke_all.index(left[0]) > poke_all.index(right[0]):
                        print(string[0])
                    else:
                        print(string[1])
            elif len(left)==4 and len(right)!=4:
                print(string[0])
            elif len(left)!=4 and len(right)==4:
                print(string[1])
            else:
                if poke_all.index(left[0]) > poke_all.index(right[0]):
                    print(string[0])
                else:
                    print(string[1])
    except:
        break
                
# HJ89
def helper(numArray, item, out):
    if len(numArray) == 1:
        if numArray[0] in dic:
            c = dic[numArray[0]]
        else:
            c = int(numArray[0])
        if item == c:
            res.append(numArray[0] + out)
    else:
        for i in range(len(numArray)):
            w = numArray[0:i] + numArray[i+1:]
            x = numArray[i]
            if x in dic:
                c = dic[numArray[i]]
            else:
                c = int(numArray[i])
            helper(w, item-c, "+"+x+out)
            helper(w, item+c, "-"+x+out)
            helper(w, item*c, "/"+x+out)
            helper(w, item/c, "*"+x+out)
            
while True:
    try:
        string = input().split()
        dic = {'J':11, 'Q':12, 'K':13, 'A':1}
        res = []
        if "joker" in string or "JOKER" in string:
            print("ERROR")
        else:
            helper(string, 24, "")
        if not res:
            print("NONE")
        else:
            print(res[0])
    except:
        break

# HJ90
while True:
    try:
        ip = input().split(".")
        count = 0
        if len(ip) != 4:
            print("NO")
            continue
        for ele in ip:
            if not ele.isdigit():
                print("NO")
                continue
            elif int(ele)>255 or ele.startswith("0") and len(ele)>1:
                print("NO")
                continue
            else:
                count += 1
        if count == 4:
            print("YES")
    except:
        break

# HJ91
def func(x, y):
    if x<0 or y<0:
        return 0
    elif x==0 or y==0:
        return 1
    else:
        return func(x-1, y) + func(x, y-1)
while True:
    try:
        n, m = map(int, input().split())
        res = func(n, m)
        print(res)
    except:
        break

while True:
    try:
        n, m = map(int, input().split())
        n, m = n+1, m+1
        dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(1, m+1):
                if i==1 and j==1:
                    dp[i][j] = 1
                    continue
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        print(dp[n][m])
    except:
        break

# HJ92
while True:
    try:
        string = input()
        for ele in string:
            if not ele.isdigit():
                string = string.replace(ele, ' ')
        string = string.split()
        max_len = 0
        res = ""
        for ele in string:
            if len(ele)>max_len:
                max_len = len(ele)
        for ele in string:
            if len(ele) == max_len:
                res += ele
        print(str(res) + "," + str(max_len))
    except:
        break
    
while True:
    try:
        string = input()
        sub_str = []
        max_len = 0
        for i in range(len(string)):
            for j in range(i+1, len(string)):
                if string[i:j+1].isdecimal() and j+1-i >= max_len:
                    sub_str.append(string[i:j+1])
                    max_len = j+1-i
        res = ""
        for ele in sub_str:
            if len(ele) == max_len:
                res += ele
        print(str(res) + "," + str(max_len))
    except:
        break

# HJ93
def dfs_digit(three, five, other):
    if not other:
        if sum(three) == sum(five):
            return True
        else:
            return False
    if dfs_digit(three+other[:1], five, other[1:]):
        return True
    if dfs_digit(three, five+other[:1], other[1:]):
        return True
while True:
    try:
        n, nums = int(input()), list(map(int, input().split()))
        three, five, other = [], [], []
        for ele in nums:
            if ele%3 == 0:
                three.append(ele)
            elif ele%5 == 0:
                five.append(ele)
            else:
                other.append(ele)
        if dfs_digit(three, five, other):
            print("true")
        else:
            print("false")
    except:
        break

# HJ94
while True:
    try:
        n, candidates = int(input()), input().split()
        m, votes = int(input()), input().split()
        valid_count = 0
        for ele in candidates:
            valid_count += votes.count(ele)
            print(ele + " : " + str(votes.count(ele)))
        print("Invalid : " + str(m - valid_count))
    except:
        break

# HJ95
while True:
    try:
        n, f = input().split(".")
        unit1 = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾']
        unit2 = ['元', '拾', '佰', '仟', '万', '拾', '佰', '仟', '亿', '拾', '佰', '仟']
        unit3 = ['角', '分']
        
        res = "人民币"
        length = len(n)
        for i in range(length):
            if i+1 == length: #整数部分读到最后一位
                res += unit1[int(n[i])]
                res += '元'
            elif n[i] == '0' and n[i+1] == '0':
                if unit2[length-i-1] == '万':
                    res += unit2[length-i-1]
                continue
            else:
                res += unit1[int(n[i])]
                res += unit2[length-i-1]
        res = res.replace("零元", "")
        res = res.replace("零佰", "零")
        res = res.replace("零仟", "零")
        res = res.replace("拾零", "拾")
        res = res.replace("壹拾", "拾")
        for i in range(2):
            if f[0] == '0' and f[1] == '0':
                res += "元整"
                break
            else:
                res += unit1[int(f[i])]
                res += unit3[i]
        res = res.replace("零角", "")
        res = res.replace("零分", "")
        print(res)
    except:
        break
                

# HJ96
while True:
    try:
        string = input()
        res = ""
        previous = ""
        for ele in string:
            if ele.isdigit():
                if not previous.isdigit():
                    res += "*"
            else:
                if previous.isdigit():
                    res += "*"
            res += ele
            previous = ele
        if ele.isdigit():
            res += "*"
        print(res)
    except:
        break
    
# HJ97
while True:
    try:
        n, nums = int(input()), list(map(int, input().split()))
        pos_num, neg_num = [], []
        for ele in nums:
            if ele>0:
                pos_num.append(ele)
            if ele<0:
                neg_num.append(ele)
        if nums.count(0) == len(nums):
            print("0 0.0")
        else:
            print(len(neg_num), round(sum(pos_num) / len(pos_num), 1))
    except:
        break

# HJ98
while True:
    try:
        order = input().split(";")
        commodity = {'A1':[2], 'A2':[3], 'A3':[4], 'A4':[5], 'A5':[8], 'A6':[6]}
        money = {'1':0, '2':0, '5':0, '10':0}
        coin = 0
        init = order[0].split()
        init_commodity = list(map(int, init[1].split('-')))
        commodity['A1'].append(init_commodity[0])
        commodity['A2'].append(init_commodity[1])
        commodity['A3'].append(init_commodity[2])
        commodity['A4'].append(init_commodity[3])
        commodity['A5'].append(init_commodity[4])
        commodity['A6'].append(init_commodity[5])
        init_money = list(map(int, init[2].split('-')))
        money['1'] += init_money[0]
        money['2'] += init_money[1]
        money['5'] += init_money[2]
        money['10'] += init_money[3]
        print("S001:Initialization is successful")
        for i in range(1, len(order)-1):
            if order[i][0] == 'p':
                input_money = int(order[i][2:])
                money_1_2 = money['1'] * 1 + money['2'] * 2
                commodity_num = 0
                for key, value in commodity.items():
                    commodity_num += value[1]
                if input_money!=1 and input_money!=2 and input_money!=5 and input_money!=10:
                    print("E002:Denomination error")
                elif input_money > money_1_2:
                    if input_money==1 or input_money==2:
                        coin += input_money
                        money[str(input_money)] += 1
                        print("S002:Pay success,balance=" + str(coin))
                    else:
                        print("E003:Change is not enough, pay fail")
                elif commodity_num == 0:
                    print("E005:All the goods sold out")
                else:
                    coin += input_money
                    money[str(input_money)] += 1
                    print("S002:Pay success,balance=" + str(coin))
            if order[i][0] == 'b':
                need_goods = order[i][2:]
                if need_goods not in commodity:
                    print("E006:Goods does not exist")
                elif commodity[need_goods][1] == 0:
                    print("E007:The goods sold out")
                elif coin < commodity[need_goods][0]:
                    print("E008:Lack of balance")
                else:
                    commodity[need_goods][1] -= 1
                    coin -= commodity[need_goods][0]
                    print("S003:Buy success,balance=" + str(coin))
            if order[i][0] == 'c':
                if coin == 0:
                    print("E009:Work failure")
                else:
                    need_back = [0,0,0,0]
                    while coin >= 0:
                        if(coin >= 10) and (money['10'] != 0):
                            coin -= 10
                            money['10'] -= 1
                            need_back[3] += 1
                        elif(coin >= 5) and (money['5'] != 0):
                            coin -= 5
                            money['5'] -= 1
                            need_back[2] += 1
                        elif(coin >= 2) and (money['2'] != 0):
                            coin -= 2
                            money['2'] -= 1
                            need_back[1] += 1
                        elif(coin >= 1) and (money['1'] != 0):
                            coin -= 1
                            money['1'] -= 1
                            need_back[0] += 1
                        else:
                            coin = 0 #余额清零
                            print('1 yuan coin number=' + str(need_back[0]))
                            print('2 yuan coin number=' + str(need_back[1]))
                            print('5 yuan coin number=' + str(need_back[2]))
                            print('10 yuan coin number=' + str(need_back[3]))
                            break
            if order[i][0] == 'q':
                if order[i] == 'q 0':
                    for key, value in commodity.items():
                        print(key + str(value[0]) + str(value[1]))
                elif order[i] == 'q 1':
                    for key, value in commodity.items():
                        print(key + ' yuan coin number=' + str(value))
                else:
                    print("E010:Parameter error")
    except:
        break
            
# HJ99
while True:
    try:
        n = int(input())
        count = 0
        for i in range(n+1):
            temp = i ** 2
            if str(temp).endswith(str(i)):
                count += 1
        print(count)
    except:
        break

# HJ100
while True:
    try:
        n = int(input())
        res = 0
        next_num = 2
        for i in range(1, n+1):
            res += next_num
            next_num += 3
        print(res)
    except:
        break

# HJ101
while True:
    try:
        n, num, flag = int(input()), list(map(int, input().split())), int(input())
        if flag == 0:
            for ele in sorted(num):
                print(ele, end=' ')
        if flag == 1:
            for ele in sorted(num, reverse=True):
                print(ele, end=' ')
    except:
        break
        
# HJ102
while True:
    try:
        string = input()
        temp = sorted(set(string))
        res = sorted(temp, key=lambda x:string.count(x), reverse=True)
        print(''.join(res))
    except:
        break

# HJ103
while True:
    try:
        n = int(input())
        num = list(map(int, input().split()))
        length = len(num)
        if length == 1:
            print(length)
        else:
            dp = [1 for i in range(length)]
            for i in range(1, length):
                for j in range(i):
                    if num[i] > num[j]:
                        dp[i] = max(dp[i], dp[j]+1)
        print(max(dp))
    except:
        break

# HJ105
pos_num = []
neg_num = []
while True:
    try:
        ele = int(input())
        if ele > 0:
            pos_num.append(ele)
        else:
            neg_num.append(ele)
    except(EOFError):
        print(len(neg_num))
        if len(pos_num):
            print(round(sum(pos_num)/len(pos_num), 1))
        else:
            print("0.0")
        break

# HJ106
while True:
    try:
        string = input()
        print(string[::-1])
    except:
        break
    
# HJ107
while True:
    try:
        n = float(input())
        e = 0.0001
        t = n
        while abs(t**3 - n) > e:
            t = t - (t**3 - n) * 1.0 / (3 * t**2)
        print("%.1f" % t)
    except:
        break
while True:
    try:
        n = float(input())
        e = 0.0001
        low = min(-1.0, n)
        high = max(1.0, n)
        res = (high+low) / 2.0
        while abs(res**3 - n) > e:
            if res**3 < n:
                low = res
            else:
                high = res
            res = (low + high) / 2.0
        print("%.1f" % res)
    except(EOFError, KeyboardInterrupt):
        break

# HJ108
while True:
    try:
        num = tuple(map(int, input().split()))
        a, b = max(num[0], num[1]), min(num[0], num[1])
        for i in range(a, a*b+1, a):
            if i%b == 0:
                print(i)
                break
    except:
        break

res = 0
def getScore(num):
    if num<=10:
        return 2
    elif 10<num<=20:
        return 4
    else:
        return 8
def dfs(total, error, problem):
    global res
    if problem>25 or error>=3 or total>=score:
        if total==score:
            res += 1
        return 
    dfs(total+getScore(problem), error, problem+1)
    dfs(total, error+1, problem+1)
if __name__ == "__main__":
    score = int(input())
    dfs(0,0,1)
    print(res)