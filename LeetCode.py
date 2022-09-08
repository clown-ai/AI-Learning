# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:22:24 2022

@author: 86153
"""

import heapq
from collections import deque
from collections import defaultdict
from collections import OrderedDict
#import random

#题目1383：最小堆找出元素再做计算
class Solution_1383(object):
    def MaxPerformance(n, speed, efficient, k):
        combine = [(speed[i], efficient[i]) for i in range(n)]
        combine = sorted(combine, key=lambda x: -x[1])
        res = 0
        MOD = 10**9+7
        min_heap = []
        speed_sum = 0
        for i in range(n):
            s, e = combine[i]
            if len(min_heap) < k:
                heapq.heappush(min_heap, s)
                speed_sum += s
            else:
                if min_heap and min_heap[0] < s:   #新来的元素s比堆中最大的元素还要大
                    speed_sum = speed_sum - min_heap[0] + s
                    heapq.heappop(min_heap)              #弹出堆顶
                    heapq.heappush(min_heap, s)          #元素入堆
            res = max(res, speed_sum * e)
        return res % MOD, min_heap
if __name__ == '__main__':
    A_1383, min_heap = Solution_1383.MaxPerformance(6, [2,10,3,1,5,8], [5,4,3,9,7,2], 4)
    print(A_1383)
    #print(min_heap)

#面试题17-14选择Top K个最小的元素
class Solution_17_14(object):
    def TopK(array, k):
        if k == 0:
            return []
        min_heap = []
        for i in array:
            if len(min_heap) < k:
                heapq.heappush(min_heap, -i)
            else:
                if min_heap and min_heap[0] < -i:
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap, -i)
        res =  [-i for i in min_heap]
        return sorted(res)
if __name__ == '__main__':
    A_17_14 = Solution_17_14.TopK([1,3,5,7,2,4,6,8], 4)
    print(A_17_14)

#选择Top K个最大的元素
class Solution_(object):
    def TopK(array, k):
        if k == 0:
            return []
        max_heap = []
        for ele in array:
            if len(max_heap) < k:
                heapq.heappush(max_heap, ele)
            else:
                if max_heap and max_heap[0] < ele:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, ele)
        res = [ele for ele in max_heap]
        return sorted(res)
if __name__ == "__main__":
    A_ = Solution_.TopK([1,3,5,7,2,4,6,8], 4)
    print(A_)

#题目1823：出队
class Solution_1823(object):
    def FindTheWinner(n, k):
        q = deque(range(1, n+1))
        while len(q) > 1:
            for _ in range(k-1):
                q.append(q.popleft())
            q.popleft()
        return q[0]
if __name__ == '__main__':
    A_1823 = Solution_1823.FindTheWinner(5,2)
    print(A_1823)

#快速排序
#class Solution(object):
def quickSort(array, start, end):
    if start >= end:
        return 
    mid_data, left, right = array[start], start, end
    while left < right:
        while array[right] >= mid_data and left < right:
            right -= 1
        array[left] = array[right]
        while array[left] < mid_data and left < right:
            left += 1
        array[right] = array[left]
    array[left] = mid_data
    quickSort(array, start, left-1)
    quickSort(array, right+1, end)
Array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
quickSort(Array, 0, len(Array)-1)
print(Array)

#接雨水
class Solution_42(object):
    def trap(height):
        N = len(height)
        if N<2: return 0
        left, right = 0, N-1
        lHeight, rHeight = height[left], height[right]
        res = 0
        while left < right:
            if lHeight < rHeight:
                left += 1
                lHeight = max(lHeight, height[left])
                res += lHeight - height[left]
            else:
                right -= 1
                rHeight = max(rHeight, height[right])
                res += rHeight - height[right]
        return res

#桶排序
class Solution_41(object):
    def firstMissingPostive(nums):
        if not nums:
            return 1
        n = len(nums)
        for i in range(n):
            while (0<nums[i]<=n and nums[i]!=nums[nums[i]-1]):
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n+1

# 76最短字串包含 / 剑指offer17
import collections
class Solution_76(object):
    def minWindow(s, t):
        if len(s) < len(t):
            return ""
        need, window = collections.defaultdict(int), collections.defaultdict(int)
        for ele in t:
            need[ele] += 1
        valid = 0
        left, right = 0, 0
        start, length = 0, len(s)+1
        while right<len(s):
            ele = s[right]
            right += 1
            if need.get(ele):
                window[ele] += 1
                if window[ele] == need[ele]:
                    valid += 1
            while valid == len(need):
                if right-left < length:
                    start = left
                    length = right - left
                temp = s[left]
                left += 1
                if need.get(temp):
                    if window[temp] == need[temp]:
                        valid -= 1
                    window[temp] -= 1
        return "" if length==len(s)+1 else s[start:start+length]

# 面试题 17-18 最短超串
class Solution_17_18:
    def shortestSeq(big, small):
        if len(big) < len(small):
            return []
        need, window = collections.Counter(small), collections.defaultdict(int)
        for ele in small:
            need[ele] += 1
        left, right = 0, 0
        start, length = 0, len(big)+1
        valid = 0
        while right < len(big):
            ele = big[right]
            right += 1
            if need.get(ele):
                window[ele] += 1
                if window[ele] == need[ele]:
                    valid += 1
            while valid == len(need):
                if right - left < length:
                    start = left
                    length = right - left
                temp = big[left]
                left += 1
                if need.get(temp):
                    if window[temp] == need[temp]:
                        valid -= 1
                    window[temp] -= 1
        #return [] if length==len(big)+1 else big[start:start+length]
        return [] if length==len(big)+1 else [start, start+length-1]

#最长回文子串
class Solution_5(object):
    def LongestReverseSubString(string):
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
                        dp[i][j] = dp[i+1][j-1] # i往右搜索，j往左搜索
                if dp[i][j] and j+1-i>max_len:
                    max_len = j+1-i
                    begin = i
        return string[begin:begin+max_len], max_len

#柱形图中的最大面积
class Solution_84(object):
    def largestRectangleArea(heights):
        length = len(heights) + 2
        heights = [0] + heights + [0] #增加两个哨兵
        stack = [0]
        res = 0
        for i in range(1, length):
            while (heights[i] < heights[stack[-1]]): #元素小于栈顶
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                res = max(res, height*width)
            stack.append(i)
        return res

# 0-1最大矩形面积
class Solution_85(object):
    def maximalRectangle(matrix):
        if not matrix:return 0
        m,n=len(matrix),len(matrix[0])
        pre=[0]*(n+1)
        res=0
        for i in range(m):
            for j in range(n):
                pre[j]=pre[j]+1 if matrix[i][j]=="1" else 0
            stack=[-1]
            for k,num in enumerate(pre):
                while num < pre[stack[-1]]:
                    height = pre[stack.pop()]
                    width = k - stack[-1] - 1
                    res = max(res, height*width)
                stack.append(k)
        return res
        
# 树最大路径和
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution_124:
    def maxPathSum(self, root:TreeNode):
        self.res = float("-inf")
        def maxGain(node):
            if not node:
                return 0
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)
            gain = node.val + leftGain + rightGain
            self.res = max(self.res, gain)
            return max(0, node.val + max(leftGain, rightGain))
        print(maxGain(root))
        return self.res

#股票-1
class Solution_121(object):
    def maxProfit(prices):
        n = len(prices)
        if n == 0: return 0 # 边界条件
        dp = [0] * n
        min_price = prices[0] 

        for i in range(1, n):
            min_price = min(min_price, prices[i])
            dp[i] = max(dp[i-1], prices[i]-min_price)
        return dp[-1]

#股票-2
class Solution_122(object):
    def maxProfit(prices):
        n = len(prices)
        res = 0
        for i in range(1, n):
            temp = prices[i] - prices[i-1]
            if temp > 0: res += temp
        return res

#股票-3
class Solution_123(object):
    def maxProfit(prices):
        n = len(prices)
        if n<2:
            return 0
        k = 2 #最多完成两笔交易
        dp = [[0]*2 for _ in range(k+1)]
        for j in range(k+1): #第一天的不合理状态
            dp[j][0] = float("-inf") 
            dp[j][1] = float("-inf")
        dp[0][0] = 0
        dp[1][1] = -prices[0]
        for i in range(1, n):
            for j in range(1, k+1):
                dp[j][0] = max(dp[j][0], dp[j][1] + prices[i]) #卖出
                dp[j][1] = max(dp[j][1], dp[j-1][0] - prices[i]) #买入
        return max(dp[j][0] for j in range(k+1))
    
    def MaxProfit(prices):
        buy1 = buy2 = prices[0]
        sell1 = sell2 = 0
        for ele in prices:
            buy1 = min(buy1, ele)
            sell1 = max(sell1, ele-buy1)
            buy2 = min(buy2, ele-sell1)
            sell2 = max(sell2, ele-buy2)
        return sell2

#糖果
class Solution_135(object):
    def candy(ratings):
        N = len(ratings)
        res = [1] * N
        for i in range(N-1):
            if ratings[i] < ratings[i+1] and res[i]>=res[i+1]: #向右扫一遍
                res[i+1] = res[i]+1
        for i in range(N-1, 0, -1):
            if ratings[i] < ratings[i-1] and res[i]>=res[i-1]: #向左扫一遍
                res[i-1] = res[i]+1
        return sum(res)

#打家劫舍
class Solution_198(object):
    def rob(nums):
        if not nums:
            return 0
        N = len(nums)
        if N == 1:
            return nums[0]
        dp = [0 for _ in range(N)]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, N):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[N-1]

#打家劫舍2
class Solution_213(object):
    def rob(self, nums):
        if not nums:
            return 0
        N = len(nums)
        if N == 1:
            return nums[0]
        return max(self.my_rob(nums[1:]), self.my_rob(nums[:N-1]))
    def my_rob(self, nums):
        if not nums:
            return 0
        N = len(nums)
        if N == 1:
            return nums[0]
        dp = [0 for _ in range(N)]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, N):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[N-1]
        
#打家劫舍3
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution_337:
    def rob(self, root:TreeNode):
        def dfs(root):
            if not root: return 0, 0

            left = dfs(root.left)
            right = dfs(root.right)
            v1 = root.val + left[1] + right[1] #偷节点，则不能偷左右节点
            v2 = max(left) + max(right) #不偷节点则去左右节点的最大值
            return v1, v2
        return max(dfs(root))

#岛屿数量
class Solution_200(object):
    def numIslands(grid):
        m, n = len(grid), len(grid[0])
        res = 0
        def dfs(i, j):
            grid[i][j] = '0'
            if 0<=i-1<m and grid[i-1][j] == '1':
                dfs(i-1, j)
            if 0<=i+1<m and grid[i+1][j] == '1':
                dfs(i+1, j)
            if 0<=j-1<n and grid[i][j-1] == '1':
                dfs(i, j-1)
            if 0<=j+1<n and grid[i][j+1] == '1':
                dfs(i, j+1)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1
        return res

#盛水
class Solution_11(object):
    def maxArea(height):
        l, r = 0, len(height)-1
        res = 0
        while l<r:
            area = min(height[l], height[r]) * (r-l)
            res = max(area, res)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return res

#滑动窗口最大值
class Solution_239:
    def maxSlidingWindow(nums, k):
        if not nums or not k:
            return []
        if len(nums) == 1:
            return nums[0]
        res = []
        queue = []
        for i in range(len(nums)):
            while queue and nums[queue[-1]] < nums[i]:
                queue.pop()
            queue.append(i)
            if queue and queue[0] == i-k:
                queue.pop(0)
            if i >= k-1:
                res.append(nums[queue[0]])
        return res

#搜索矩阵
class Solution_74(object):
    def searchMatrix(matrix, target):
        for i in range(len(matrix)):
            l, r = 0, len(matrix[0])-1
            while l<=r:
                m = (l+r)//2
                if matrix[i][m] == target:
                    return True
                elif matrix[i][m] > target:
                    r = m-1
                else:
                    l = m+1
        return False

#24点
class Solution_679(object):
    def judgePoint24(self, cards) -> bool:
        def dfs(nums):
            if len(nums) == 1:
                return abs(nums[0] - 24) < 0.00001

            for i in range(len(nums)-1):
                for j in range(i+1, len(nums)):
                    x, y = nums[i], nums[j]
                    rest = nums[:i] + nums[i+1:j] + nums[j+1:]
                    
                    a = dfs(rest + [x + y])
                    b = dfs(rest + [x - y])
                    c = dfs(rest + [y - x])
                    d = dfs(rest + [x * y])
                    e = y != 0 and dfs(rest + [x / y])
                    f = x != 0 and dfs(rest + [y / x])
                    
                    if a or b or c or d or e or f:
                        return True
            return False        
        return dfs(cards)

# 227基本计算器2
class Solution_227(object):
    def calculate(s):
        stack = []
        pre_op = "+"
        num = 0
        for i, ele in enumerate(s):
            if ele.isdigit():
                num = 10 * num + int(ele)
            if i==len(s)-1 or s[i] in "+-*/":
                if pre_op == '+':
                    stack.append(num)
                elif pre_op == '-':
                    stack.append(-num)
                elif pre_op == '*':
                    stack.append(stack.pop() * num)
                elif pre_op == '/':
                    top = stack.pop()
                    if top < 0:
                        stack.append(int(top/num))
                    else:
                        stack.append(top // num)
                pre_op = ele
                num = 0
        return sum(stack)

#字符串转整型
class Solution_8(object):
    def myAtoi(s):
        flag = 1
        n = len(s)
        i = 0
        num = 0
        if not s:
            return 0
        for i in range(n):
            if s[i] != ' ':
                break
        if s[i] == '-' and i<n-1:
            flag = -1
            i += 1
        elif s[i] == '+' and i<n-1:
            flag = 1
            i += 1
        else:
            if not s[i].isdigit():
                return 0
        while s[i].isdigit() and i<n:
            num = num*10 + int(s[i])
            if i==n-1 or not s[i].isdigit():
                break
            i += 1
        num = num * flag
        if num < -2**31:
            return -2**31
        elif num > 2**31-1:
            return 2**31-1
        else:
            return num

#合并K个升序链表
class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None
class Solution_23:
    def mergeKLists(self, lists:list[100000][ListNode]):
        minHeap = []
        for ele in lists:
            while ele:
                heapq.heappush(minHeap, ele.val)
                ele = ele.next
        res = ListNode()
        head = res 
        while minHeap:
            head.next = ListNode(heapq.heappop(minHeap))
            head = head.next
        return res.next

#整数翻译成英文
class Solution_273(object):
    def numberToWords(num):
        unit1 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
        unit2 = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
        res = []
        def lessHundred(n):
            if n>0:
                if n<20:
                    res.append(unit1[n])
                else:
                    res.append(unit2[n//10])
                    if num%10 != 0:
                        res.append(unit1[n%10])
        def lessThousand(n):
            if n>=100:
                res.append(unit1[n//100])
                res.append("Hundred")
            lessHundred(n%100)
        a = num%1000
        b = (num%1000) / 1000
        c = (num%1000000) / 1000
        d = (num%1000000000) / 1000
        if d>0:
            lessThousand(d)
            res.append("Billion")
        if c>0:
            lessThousand(c)
            res.append("Million")
        if b>0:
            lessThousand(b)
            res.append("Thousand")
        if a>0:
            lessThousand(a)
        if num!=0:
            return " ".join(res)
        else:
            return "Zero"

#字典序第k小数字
class Solution_440(object):
    def findKthNumber(n, k):
        res = 1
        k -= 1
        while k>0:
            count = 0
            interval = [res, res+1]
            while interval[0] <= n:
                count += min(n+1, interval[1]) - interval[0]
                interval = [10*interval[0], 10*interval[1]]
            if k >= count:
                res += 1
                k -= count
            else:
                res *= 10
                k -= 1
        return res
    
#最大为 n 的数字组合
class Solution_902:
    def atMostNGivenDigitSet(digits, n):
        s = str(n)
        length = len(s)
        dp = [0] * length + [1]
        for i in range(length-1, -1, -1):
            for ele in digits:
                if ele < s[i]:
                    dp[i] += len(digits) ** (length-i-1)
                elif ele == s[i]:
                    dp[i] += dp[i+1]
        return dp[0] + sum(len(digits)**i for i in range(1, length))

#字节大数据一题
while True:
    try:
        N = int(input())
        x = []
        y = []
        res = []
        temp = []
        for i in range(N):
            m, n = map(int, input().split())
            temp.append((m,n))
        temp = sorted(temp, key=lambda x: x[0])
        res.append(temp[N-1])
        max_y = temp[N-1][1]
        for i in range(N-2, -1, -1):
            if temp[i][1] > max_y:
                res.append(temp[i])
                max_y = temp[i][1]
        for i in range(len(res)-1, -1, -1):
            print(res[i][0], res[i][1])
    except:
        break

# Wendy Teacher assigned following missions

# 45跳跃游戏2
class Solution_45(object):
    def jump(nums):
        n = len(nums)
        #当前范围内可以跳到的最远距离，记录跳跃的次数，边界
        max_pos, count, end = 0, 0, 0
        for i in range(n-1):
            max_pos = max(max_pos, i+nums[i])
            if i == end:
                end = max_pos
                count += 1
        return count

# 1190反转没对括号间的子串（栈）
class Solution_1190(object):
    def reverseParentheses(s):
        stack = []
        for ele in s:
            if ele != ')':
                stack.append(ele)
            elif ele == ')':
                temp = []
                while stack and stack[-1] != '(':
                    temp.append(stack.pop())
                if stack:
                    stack.pop()   #抛出左括号
                stack += temp
        return "".join(stack)

# 781森林中的兔子（数学）
class Solution_781(object):
    def numRabbits(answers):
        temp = {}
        for ele in answers:
            if ele not in temp:
                temp[ele] = 1
            else:
                temp[ele] += 1
        res = 0
        for key, value in temp.items():
            ans = value // (key+1)
            if value % (key+1) !=0:
                ans += 1
            res += ans * (key+1)
        return res
    
# 739每日的温度 (单调栈)
class Solution_739(object):
    def dailyTemperatures(temperatures):
        stack = []
        length = len(temperatures)
        res = [0] * length
        for i, ele in enumerate(temperatures):
            while stack and ele > temperatures[stack[-1]]:
                res[stack.pop()] = i - stack[-1]
            stack.append(i)
        return res

# 3无重复字符的最长子串（滑动窗口）
class Solution_3(object):
    def lengthOfLongestSubstring(s):
        if not s:
            return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len, cur_len = 0, 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len:
                max_len = cur_len
            lookup.add(s[i])
        return max_len

# 46全排列（回溯）
class Solution_46(object):
    def permute(nums):
        res = []
        def backTrack(nums, temp):
            if not nums:
                res.append(temp)
                return
            for i in range(len(nums)):
                backTrack(nums[:i] + nums[i+1:], temp+[nums[i]])
        backTrack(nums, [])
        return res

# 475供暖器（双指针）
import math
class Solution_475(object):
    def findRadius(houses, heaters):
        heaters = heaters + [-math.inf, math.inf]
        heaters.sort()
        houses.sort()
        left, right, res = 0, 0, 0
        while left < len(houses):
            cur = math.inf
            while heaters[right] <= houses[left]:
                cur = houses[left] - heaters[right]
                right += 1
            cur = min(cur, heaters[right]-houses[left])
            res = max(cur, res)
            left += 1
            right -= 1
        return res

# 20有效的括号（栈）
class Solution_20(object):
    def isValid(s):
        mydict = {'(': ')', '[': ']', '{': '}', '?':'?'}
        stack = ['?']
        for ele in s:
            if ele in mydict:
                stack.append(ele)
            elif mydict[stack.pop()] != ele:
                return False
        return len(stack)==1

# 394字符串解码（栈）
class Solution_394(object):
    def decodeString(s):
        stack = []
        for ele in s:
            if ele != ']':
                stack.append(ele)
            else:
                string = ""
                while stack[-1] != '[':
                    string = stack.pop() + string
                stack.pop() #抛出 "["
                num = ""
                while stack and stack[-1].isnumeric():
                    num = stack.pop() + num
                stack.append(int(num) * string)
        return "".join(stack)

# 179最大数（字符串）
class Solution_179(object):
    def largestNumber(nums):
        num_str = map(str, nums)
        compare = lambda x, y: 1 if x+y < y+x else -1
        num_str.sort(cmp = compare)
        res = "".join(num_str)
        if res[0] == '0':
            return "0"
        return res
    def LargestNumber(nums):
        n = len(nums)
        temp = list(map(str, nums))
        for i in range(n):
            for j in range(i+1, n):
                if temp[i]+temp[j] < temp[j]+temp[i]:
                    temp[i], temp[j] = temp[j], temp[i]
        return str(int("".join(temp)))

# LCP.09最小跳次数
class Solution_09(object):
    def minJump(jump):
        n = len(jump)
        dp = [0] * n
        for i in range(n-1, -1, -1):
            if i+jump[i] >= n:
                dp[i] = 1
            else:
                dp[i] = 1 + dp[i+jump[i]]
            for j in range(i+1, n):
                if dp[j] <= dp[i]:
                    break
                else:
                    dp[j] = dp[i] + 1
        return dp[0]

# 135分发糖果
class solution_135(object):
    def candy(ratings):
        N = len(ratings)
        res = [1] * N
        for i in range(N-1):
            if ratings[i] < ratings[i+1] and res[i]>=res[i+1]: #向右扫一遍
                res[i+1] = res[i]+1
        for i in range(N-1, 0, -1):
            if ratings[i] < ratings[i-1] and res[i]>=res[i-1]: #向左扫一遍
                res[i-1] = res[i]+1
        return sum(res)

# 17.24最大子矩阵
class Solution_17_24(object):
    def getMaxMatrix(matrix):
        row, col = len(matrix), len(matrix[0])
        res = [0] * 4
        r1, c1, r2, c2 = 0, 0, 0, 0
        max_sum = matrix[0][0]
        for i in range(row):
            temp = [0] * col
            for j in range(i, row):
                cur_max = 0
                for k in range(col):
                    temp[k] += matrix[j][k]
                    if cur_max > 0:
                        cur_max += temp[k]
                    else:
                        cur_max = temp[k]
                        r1 = i
                        c1 = k
                    if cur_max > max_sum:
                        r2 = j
                        c2 = k
                        max_sum = cur_max
                        res = r1, c1, r2, c2
        return res

# 剑指offer 95最长公共子序列 (包含即可的形式，不是一定要连续)
class Solution_SO_95(object):
    def longestCommonSubsequnce(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

# 1两数之和
class Solution_1(object):
    def twoSum(nums, target):
        hashtable = dict()
        for i, ele in enumerate(nums):
            if target-ele in hashtable:
                return [hashtable[target-ele], i]
            hashtable[nums[i]] = i
        return []

# 2两数相加
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution_2:
    def addTowNumber(self, l1, l2, carry=0):
        num1, num2 = l1.val if l1 else 0, l2.val if l2 else 0
        s = num1 + num2 + carry
        val, carry = s%10, 1 if s>=10 else 0
        l1, l2 = l1.next if l1 else None, l2.next if l2 else None
        if l1 or l2 or carry:
            return ListNode(val, self.addTowNumber(l1, l2, carry))
        return ListNode(val)

# 两数相加（右对齐） 剑指offer25 / 445 两数相加2
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution_SO_25:
    def addTwoNumber(self, l1, l2):
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        carry = 0
        res = None
        while s1 or s2 or carry:
            a = 0 if not s1 else s1.pop()
            b = 0 if not s2 else s2.pop()
            s = a+b+carry
            val, carry = s%10, s//10
            node = ListNode(val)
            node.next = res
            res = node
        return res

# 面试题02.05 链表求和
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.neext = None
    def addTwoNumber(self, l1, l2):
        head = ListNode(-1)
        p = head
        carry = 0
        while l1 or l2 or carry:
            num1, num2 = l1.val if l1 else 0, l2.val if l2 else 0
            s = num1 + num2 + carry
            val, carry = s%10, s//10
            p.next = ListNode(val)
            p = p.next
            l1, l2 = l1.next if l1 else None, l2.next if l2 else None
        return head.next

# 22括号生成（回溯）
class Solution_22(object):
    def generateParenthesis(n):
        res = []
        def backtrack(s, left, right):
            if len(s) == 2*n:
                res.append("".join(s))
                return 
            if left < n:
                s.append("(")
                backtrack(s, left+1, right)
                s.pop()
            if right < left:
                s.append(")")
                backtrack(s, left, right+1)
                s.pop()
        backtrack([], 0, 0)
        return res

# 554砖墙
class Solution_554(object):
    def leastBricks(wall):
        mydict = defaultdict(int)
        for ele in wall:
            distance = 0
            for i in range(0, len(ele)-1):
                distance += ele[i]
                mydict[distance] += 1
        if len(mydict.values()) == 0:
            return len(wall)
        return len(wall) - max(mydict.values())

# 547省份数量
class Solution_547(object):
    def finCircleNum(isConnected):
        def dfs(i):
            for j in range(cities):
                if isConnected[i][j] == 1 and j not in visited:
                    visited.add(j)
                    dfs(j)
        cities = len(isConnected)
        visited = set()
        count = 0
        for i in range(cities):
            if i not in visited:
                dfs(i)
                count += 1
        return count

# 55跳跃游戏1
class Solution_55(object):
    def canJump(nums):
        k = 0
        n = len(nums)
        for i in range(n):
            if i<=k:
                k = max(k, i+nums[i])
                if k >= n-1: #最后一个位置的索引
                    return True
        return False

# 146LRU缓存        
class Solution_146:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache: 
            return -1
        val = self.cache[key]
        self.cache.move_to_end(key)
        return val

    def put(self, key, val):
        if key in self.cache: 
            del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# 剑指offer 16.25 LRU缓存
class Solution_16_25:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache: 
            return -1
        val = self.cache[key]
        self.cache.move_to_end(key)
        return val

    def put(self, key, val):
        if key in self.cache: 
            del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# 621任务调度器
class Solution_621(object):
    def leastInterval(tasks, n):
        dict_count = list(collections.Counter(tasks).values())
        maxCount = max(dict_count)
        eleCount = dict_count.count(maxCount)
        return max((maxCount-1) * (n+1) + eleCount, len(tasks))
    
# 1047删除字符串中所有相邻重复项
class Solution_1047(object):
    def removeDuplicates(s):
        stack = []
        for ele in s:
            if stack and stack[-1] == ele:
                stack.pop()
            else:
                stack.append(ele)
        return "".join(stack)

# 14最长公共前缀
class Solution_14(object):
    def longestCommonPrefix(strs):
        res = ""
        for ele in zip(*strs):
            temp = set(ele)
            if len(temp) == 1:
                res += ele[0]
            else:
                break
        return res
    def LongestCommonPrefix(strs):
        if not strs:
            return ""
        min_s = min(strs)
        max_s = max(strs)
        for i in range(len(min_s)):
            if min_s[i] != max_s[i]:
                return min_s[:i]
        return min_s
    
# 300最长递增子序列
class Solution_300(object):
    def lengthOfLIS(nums):
        if not nums:
            return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)

# 56合并区间
class Solution_56(object):
    def merge(intervals):
        intervals = sorted(intervals, key=lambda x: x[0])
        res = []
        for ele in intervals:
            if not res or res[-1][1] < ele[0]:
                res.append(ele)
            else:
                res[-1][1] = max(res[-1][1], ele[1])
        return res

# 799香槟塔
class Solution_799(object):
    def champagneTower(poured, query_row, query_glass):
        n = query_row
        dp = [[0 for _ in range(n+1)] for _ in range(n+1)]
        dp[0][0] = poured
        for i in range(n):
            for j in range(n+1):
                if dp[i][j] > 1:
                    dp[i+1][j] += (dp[i][j]-1)/2
                    dp[i+1][j+1] += (dp[i][j]-1)/2
                    dp[i][j] = 1
        return min(1, dp[query_row][query_glass])

# 316去除重复字母
class Solution_316(object):
    def removeDuplicateLetters(s):
        stack = []
        remain = collections.Counter(s)
        for ele in s:
            if ele not in stack:
                while stack and stack[-1] > ele and remain[stack[-1]] > 0:
                    stack.pop()
                stack.append(ele)
            remain[ele] -= 1
        return "".join(stack)

# 392判断子序列
class Solution_392(object):
    def isSubsequence(s, t):
        stack = []
        for ele in t:
            if ele in s:
                stack.append(ele)
        if s in "".join(stack):
            return True
        return False
    
    def IsSubsequence(s, t):
        stack = list(s)
        for ele in t:
            try:
                if ele == stack[0]:
                    stack.pop()
            except IndexError:
                return True
        if stack:
            return False
        else:
            return True
    
# 206反转链表
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution_206:
    def reverseList(self, head:ListNode):
        cur = head
        pre = None
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

# 221最大正方形
class Solution_221(object):
    def maximalSquare(matrix):
        if not matrix: return 0
        m, n = len(matrix), len(matrix[0])
        width = 0
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i*j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                width = max(width, dp[i][j])
        return width*width

# 62不同路径
class Solution_62(object):
    def uniquePath(m, n):
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i*j == 0: # 设置边界条件
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

# 63不同路径
class Solution_63(object):
    def uniquePathsWithObstacles(obstacleGrid):
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if not obstacleGrid[i][j]:
                    if i==j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

# 130被围绕的区域
class Solution_130(object):
    def solve(board):
        if not board:
            return 
        m, n = len(board), len(board[0])
        def dfs(x, y):
            if not 0<=x<m or not 0<=y<n or board[x][y]!='O':
                return 
            board[x][y] = 's'
            dfs(x+1, y)
            dfs(x-1, y)
            dfs(x, y+1)
            dfs(x, y-1)
        for i in range(m): #列上进行遍历
            dfs(i,0)
            dfs(i, n-1)
        for i in range(n-1): #行上进行遍历
            dfs(0,i)
            dfs(m-1, i)
        for i in range(m):
            for j in range(n):
                if board[i][j] == 's':
                    board[i][j] = 'O'
                elif board[i][j] == 'O':
                    board[i][j] = 'X'

# 70爬楼梯
class Solution_70(object):
    def climbStairs(n):
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

# 17电话号码的字母组合
class Solution_17(object):
    def letterCombinations(digits):
        if not digits:
            return []
        number = {'2':['a', 'b', 'c'],
                  '3':['d', 'e', 'f'],
                  '4':['g', 'h', 'i'],
                  '5':['j', 'k', 'l'],
                  '6':['m', 'n', 'o'],
                  '7':['p', 'q', 'r', 's'],
                  '8':['t', 'u', 'v'],
                  '9':['w', 'x', 'y', 'z']}
        def backtrack(combination, digit):
            if len(digit) == 0:
                res.append(combination)
            else:
                for ele in number[digit[0]]:
                    backtrack(combination+ele, digit[1:])
        res = []
        backtrack('', digits)
        return res
        
# 19删除链表中倒数第 N 个节点
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution_19:
    def removeNthFromEnd(head:ListNode, n):
        dummy = ListNode(next=head)
        fast, slow = dummy, dummy
        for _ in range(n):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
    
# 15三数之和
class Solution(object):
    def threeSum(nums):
        if not nums or len(nums)<3:
            return []
        nums.sort()
        n = len(nums)
        res = []
        for k in range(n-2): # k 为特殊指针
            if nums[k]>0:
                break
            if k>0 and nums[k] == nums[k-1]:
                continue
            i, j = k+1, n-1
            while i<j:
                s = nums[k]+nums[i]+nums[j]
                if s<0: #值小了，右边搜索
                    i += 1
                    while i<j and nums[i] == nums[i-1]: i += 1 #遇到重复的直接跳过
                elif s>0:
                    j -= 1
                    while i<j and nums[j] == nums[j+1]: j -= 1
                else:
                    res.append(nums[k], nums[i], nums[j])
                    i += 1
                    j -= 1
                    while i<j and nums[i] == nums[i-1]: i+= 1
                    while i<j and nums[j] == nums[j+1]: j-= 1
        return res

# 反转字符串中的单词
class Solution_151(object):
    def reverseWords(s):
        temp = s.split()
        res = temp[::-1]
        return " ".join(res)

# N 天后的牢房
class Solution_957(object):
    def prisonAfterNDays(cells, n):
        my_dict = collections.defaultdict(tuple)
        flag = False
        while n:
            cell = [0 for _ in range(8)]
            for i in range(1, 7):
                cell[i] = 1-(cells[i-1] ^ cells[i+1])
            cells = cell
            n -= 1
            if flag == False:
                if tuple(cells) not in my_dict:
                    my_dict[tuple(cells)] = 1
                else:
                    flag = True
                    n = n%len(my_dict)
        return cells

# 518零钱兑换2
class Solution_518(object):
    def change(amount, coins):
        n = len(coins)
        dp = [[0 for _ in range(amount+1)] for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(amount+1):
                if j<coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
        return dp[n][amount]
    
# 139单词拆分
class Solution_139(object):
    def wordBreak(s, wordDict):
        n = len(s)
        dp = [False for _ in range(n+1)]
        dp[0] = True
        for i in range(n):
            if dp[i]:
                for j in range(i+1, n+1):
                    if s[i:j] in wordDict:
                        dp[j] = True
        return dp[n]

# 47全排列2
class Solution_47(object):
    def permuteUnique(nums):
        if not nums:
            return []
        nums.sort()
        res = []
        def backtrack(nums, temp):
            if not nums:
                res.append(temp)
                return 
            for i in range(len(nums)):
                if i>0 and nums[i] == nums[i-1]:
                    continue
                backtrack(nums[:i]+nums[i+1:], temp+[nums[i]])
        backtrack(nums, [])
        return res

# 90子集2
class Solution_90(object):
    def subsetsWithDup(nums):
        nums.sort()
        res = []
        def backtrack(temp, k):
            res.append(nums)
            for i in range(k, len(nums)):
                if i>k and nums[i] == nums[i-1]:
                    continue
                else:
                    backtrack(temp+[nums[i]], i+1)
        backtrack([], 0)
        return res
    
# 93复原IP地址
class Solution_93(object):
    def restoreIpAddresses(s):
        if not s or len(s)<4:
            return []
        res = []
        def backtrack(temp, index, path, res):
            if index>4:
                return 
            if index == 4 and not s:
                res.append(path[:-1])
                return 
            for i in range(len(s)):
                if s[:i+1]=='0' or (s[0]!='0' and 0<int(s[:i+1])<256):
                    backtrack(s[i+1:], index+1, path+s[:i+1]+'.', res)
        backtrack(s, 0, '', res)

# 224基本计算器
class Solution_224(object):
    def calculate(s):
        res, sign = 0, 1
        stack = [1]
        i, n = 0, len(s)
        while i<n:
            if s[i] == ' ':
                i += 1
            elif s[i] == '+':
                sign = stack[-1]
                i += 1
            elif s[i] == '-':
                sign = -stack[-1]
                i += 1
            elif s[i] == '(':
                stack.append(sign)
                i += 1
            elif s[i] == ')':
                stack.pop()
                i += 1
            else:
                num = 0
                while i<n and s[i].isdigit():
                    num  = num*10 + int(s[i])
                    i += 1
                res += sign * num
        return res
    
# 692前 K 个高频单词
class Solution_692(object):
    def topKFrequent(words, k):
        my_dict = {}
        for ele in words:
            if ele not in my_dict:
                my_dict[ele] = 1
            else:
                my_dict[ele] += 1
        res = sorted(my_dict, key=lambda x: (-my_dict[x], x))
        return res[:k]
    
    def TopKFrequent(words, k):
        my_dict = {}
        for ele in words:
            if ele not in my_dict:
                my_dict[ele] = 1
            else:
                my_dict[ele] += 1
        max_heap = []
        res = []
        for key, value in my_dict.items():
            heapq.heappush(max_heap, (-value, key))
        for i in range(k):
            res.append(heapq.heappop(max_heap)[1]) #一号位置是key
        return res

# 347前 K 个高频元素
class Solution_347(object):
    def topKFrequent(nums, k):
        my_dict = {}
        for ele in nums:
            if ele not in my_dict:
                my_dict[ele] = 1
            else:
                my_dict[ele] += 1
        max_heap = []
        res = []
        for key, value in my_dict.items():
            heapq.heappush(max_heap, (-value, key))
        for i in range(k):
            res.append(heapq.heappop(max_heap)[1])
        return res
    
# 38外观数列
class Solution_38(object):
    def countAndSay(self, n):
        temp = "1"
        for i in range(n-1):
            pos = 0
            start = 0
            cur = ""
            while pos<len(temp):
                while pos<len(temp) and temp[pos]==temp[start]:
                    pos += 1
                cur += str(pos-start) + temp[start]
                start = pos
            temp = cur
        return temp

# 64最小路径和
class Solution_64(object):
    def minPathSum(grid):
        m, n = len(grid), len(grid[0])
        #dp = [[0 for _ in range(n)] for _ in range(m)]
        #dp[0][0] = grid[0][0]
        for i in range(m):
            for j in range(n):
                if i==0 and j==0:
                    continue
                elif i==0 and j!=0:
                    grid[i][j] += grid[i][j-1]
                elif i!=0 and j==0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]
    
# 735行星碰撞
class Solution_735(object):
    def asteroidCollision(asteroids):
        stack = []
        for ele in asteroids:
            alive = True
            while alive and ele<0 and stack and stack[-1]>0:
                alive = abs(stack[-1]) < abs(ele) # ele能否存活
                if abs(stack[-1]) <= abs(ele):
                    stack.pop()
            if alive:
                stack.append(ele)
        return stack
    
# 165比较版本号
class Solution_165(object):
    def compareVersion(version1, version2):
        m, n = len(version1), len(version2)
        i, j = 0, 0
        while i<m or j<n:
            x = 0
            while i<m and version1[i]!='.':
                x = x*10 + int(version1[i])
                i += 1
            i += 1
            y = 0
            while j<n and version2[j]!='.':
                y = y*10 + int(version2[j])
                j += 1
            j += 1
            if x!=y:
                return 1 if x>y else -1
        return 0

# 678有效的括号字符串
class Solution_678(object):
    def CheckValidString(s):
        stack_left, stack_star=[],[]
        for i in range(len(s)):
            if s[i]=='(': 
                stack_left.append(i)
            elif s[i]=='*':
                stack_star.append(i)
            elif s[i]==')':
                if stack_left: 
                    stack_left.pop()
                elif stack_star:
                    stack_star.pop()
                else:
                    return False
        while stack_left:
            if not stack_star: 
                return False
            elif stack_left[-1]>stack_star[-1]: 
                return False
            else:
                stack_left.pop()
                stack_star.pop()
        return True
    
# 102二叉树的层序遍历
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution_102:
    def levelOrder(self, root):
        if not root:
            return []
        res = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            m = len(queue)
            ans = []
            for i in range(m):
               temp = queue.popleft()
               ans.append(temp.val)
               if temp.left:
                   queue.append(temp.left)
               if temp.right:
                   queue.append(temp.right)
            res.append(ans)
        return res
        
# 287寻找重复数
class Solution_287(object):
    def findDuplicate(nums):
        n = len(nums)
        left, right = 0, n
        while left < right:
            mid_pos = left + (right - left)//2
            count = sum(ele<=mid_pos for ele in nums)
            if count <= mid_pos:
                left = mid_pos + 1
            else:
                right = mid_pos
        return left
    
# 442数组中重复的数据
class Solution_442(object):
    def findDuplicate(nums):
        for i in range(len(nums)):
            while nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        return [num for i, num in enumerate(nums) if num!=i+1]
    
# 16最接近的三数之和
class Solution_16(object):
    def threeSumClosest(nums, target):
        nums.sort()
        n = len(nums)
        res = 10**7
        for k in range(n):
            if k>0 and nums[k]==nums[k-1]:
                continue
            i, j = k+1, n-1
            while i<j:
                s = nums[k]+nums[i]+nums[j]
                if s==target:
                    return target
                if abs(s-target) < abs(res-target):
                    res = s
                if s > target:
                    j -= 1
                    while i<j and nums[j]==nums[j+1]: j-=1
                if s < target:
                    i += 1
                    while i<j and nums[i]==nums[i-1]: i+=1
        return res
    
# 264丑数
class Solution_264(object):
    def nthUglyNumber(n):
        if n<0:
            return 0
        dp = [1] * n
        index2, index3, index5 = 0, 0, 0
        for i in range(1, n):
            dp[i] = min(dp[index2]*2, dp[index3]*3, dp[index5]*5)
            if dp[i] == 2*dp[index2]: index2 += 1
            if dp[i] == 3*dp[index3]: index3 += 1
            if dp[i] == 5*dp[index5]: index5 += 1
        return dp[n-1]
    
# 剑指offer38 字符串的排列
class Solution_SO_38(object):
    def permutation(s):
        if not s:
            return []
        s = list(sorted(s))
        res = []
        def backtrack(s, temp):
            if not s:
                res.append("".join(temp))
                return 
            for i in range(len(s)):
                if i>0 and s[i] == s[i-1]:
                    continue
                backtrack(s[:i]+s[i+1:], temp+[s[i]])
        backtrack(s, temp)
        return res

# 40组合总和2 （出现了重复数字）
class Solution_40(object):
    def combinationSum2(candidates, target):
        candidates.sort()
        res = []
        def backtrack(index, target, temp):
            if target == 0:
                res.append(temp)
            for i in range(index, len(candidates)):
                if candidates[i] > target:
                    break
                if i>index and candidates[i]==candidates[i-1]: #加一步判定
                    continue
                backtrack(i+1, target-candidates[i], temp+[candidates[i]])
        backtrack(0, target, [])
        return res
    
# 746使用最小花费爬楼梯 / 剑指offer 88
class Solution_746(object):
    def minCostClimbingStairs(cost):
        n = len(cost)
        dp = [0 for _ in range(n+1)]
        for i in range(2, n+1):
            dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
        return dp[n]

# 974和可被 K 整除的子数组
class Solution_974(object):
    def subarraysDivByK(nums, k):
        records = {0:1}
        prefix_sum = 0
        res = 0
        for ele in nums:
            prefix_sum += ele
            mod = prefix_sum % k
            count = records.get(mod,0)
            res += count
            records[mod] = count + 1
        return res
    
# 260只出现一次的数字3
class Solution_260(object):
    def singleNumber(nums):
        my_set = set()
        for ele in nums:
            if ele in my_set:
                my_set.remove(ele)
            else:
                my_set.add(ele)
        return list(my_set)
    
# 695岛屿的最大面积
class Solution_695(object):
    def maxAreaOfIsland(grid):
        m, n = len(grid), len(grid[0])
        def dfs(i, j):
            if 0<=i<m and 0<=j<n and grid[i][j]:
                grid[i][j] = 0
                return 1+dfs(i-1, j) + dfs(i+1,j) + dfs(i,j-1) + dfs(i,j+1)
            return 0
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    res = max(res, dfs(i,j))
        return res

# 36有效的数独
class Solution_36(object):
    def isValidSudoku(board):
        row = [[] * 9 for _ in range(9)]
        col = [[] * 9 for _ in range(9)]
        block = [[] * 9 for _ in range(9)]
        
        for i in range(9):
            for j in range(9):
                temp = board[i][j]
                if not temp.isdigit():
                    continue
                if temp in row[i]:
                    return False
                if temp in col[j]:
                    return False
                if temp in block[(j//3)*3 + i//3]:
                    return False
                row[i].append(temp)
                col[j].append(temp)
                block[(j//3)*3 + i//3].append(temp)
        return True
    
# 148排序链表
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution_148:
    def sortedList(self, head:ListNode):
        if not head or not head.next: 
            return head
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        left = self.sortList(head)
        right = self.sortList(mid)
        return self.merge(left, right)
    def merge(self, left, right):
        dummy = ListNode(0)
        p = dummy
        l = left
        r = right
        while l and r:
            if l.val < r.val:
                p.next = l
                l = l.next
                p = p.next
            else:
                p.next = r
                r = r.next
                p = p.next
        if l:
            p.next = l
        if r:
            p.next = r
        return dummy.next
    
# 199二叉树的右视图
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution_199:
    def rightSlideView(self, root:TreeNode):
        res = []
        def dfs(root, depth):
            if not root:
                return 
            if depth == len(res):
                res.append(root.val)
            depth += 1
            dfs(root.right, depth)
            dfs(root.left, depth)
        dfs(root, 0)
        return res
    
# 322零钱兑换
class Solution_322(object):
    def coinChange(coins, amount):
        n = len(coins)
        dp = [[amount+1 for _ in range(amount+1)] for _ in range(n+1)]
        dp[0][0] = 0
        for i in range(1, n+1):
            for j in range(amount+1):
                if j < coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]]+1)
        res = dp[n][amount]
        return res if res!=amount+1 else -1
    
# 406根据身高重建队列
class Solution_406(object):
    def reconstructQueue(people):
        res = []
        people = sorted(people, key=lambda x: (-x[0], x[1]))
        for ele in people:
            if len(res) <= ele[1]:
                res.append(ele)
            else:
                res.insert(ele[1], ele)
        return res
    
# 1162地图分析
class Solotion_1162(object):
    def maxDistance(grid):
        m, n = len(grid), len(grid[0])
        stack = []
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    stack.append([i,j])
        if not stack or len(stack)==m*n:
            return -1
        count = 1
        while stack:
            temp = []
            for i, j in stack:
                for x, y in [[i-1,j], [i+1,j], [i,j-1], [i,j+1]]:
                    if 0<=x<m and 0<=y<n and not grid[x][y]:
                        temp.append([x,y])
                        grid[x][y] = count
            stack = temp
            count += 1
        return count-2
    
# 343整数拆分
class Solution_343(object):
    def integerBreak(n):
        dp = [0 for _ in range(n+1)]
        for i in range(2, n+1):
            for j in range(i):
                dp[i] = max(dp[i], j*(i-j), j*dp[i-j])
        return dp[-1]
    
# 34在排序数组中查找元素的第一个和最后一个位置
class Solution_34(object):
    def searchRange(nums, target):
        def left_bound(nums, target):
            index = -1
            left, right = 0, len(nums)-1
            while left<=right:
                mid = left + (right-left)//2
                if nums[mid] == target:
                    index = mid
                    right = mid-1 #往回缩一次
                elif nums[mid]>target:
                    right = mid-1
                else:
                    left = mid+1
            return index
        def right_bound(nums, target):
            index = -1
            left, right = 0, len(nums)-1
            while left<=right:
                mid = left + (right-left)//2
                if nums[mid] == target:
                    index = mid
                    left = mid+1 #往前进一次
                elif nums[mid]>target:
                    right = mid-1
                else:
                    left = mid+1
            return index
        left = left_bound(nums, target)
        right = right_bound(nums, target)
        return [left, right]
                    
# 29两数相除
class Solution_29(object):
    def divide(dividend, divisor):
        sign = 1
        if dividend * divisor < 0:
            sign = -1
            dividend = abs(dividend)
            divisor = abs(divisor)
        elif dividend<0 and divisor<0:
            dividend = abs(dividend)
            divisor = abs(divisor)
        remain = dividend
        res = 0
        while remain>=divisor:
            temp = 1
            div = divisor
            while div+div < remain:
                temp += temp
                div += div
            remain -= div
            res += temp
        if sign == -1:
            res = -res
        if res >= 2**31-1 or res<-2**31:
            return 2**31-1
        return res
    
# 105构造二叉树
class Solution_105:
    def buildTree(self, preorder, inorder):
        if not preorder and not inorder:
            return
        root = TreeNode(preorder[0])
        i = inorder.index(inorder[0]) # 由中序得到根节点的情况进而进行划分
        root.left = self.buildTree(preorder[1:i+1], inorder[:i])
        root.right = self.buildTree(preorder[i+1:], inorder[i+1:])
        return root

# 106构造二叉树
class Solution_106(object):
    def buildTree(self, inorder, postorder):
        if not inorder and not postorder:
            return 
        root = TreeNode(postorder[-1])
        i = inorder.index[postorder[-1]] # 由中序得到根节点的情况进行划分
        root.left = self.buildTree(inorder[:i], postorder[:i])
        root.right = self.buildTree(inorder[i+1:], postorder[i:-1])
        return root
    
# 451根据字符出现次数排序
class Solution_451(object):
    def frequencySort(s):
        res = ""
        my_dict = collections.Counter(s)
        temp = sorted(my_dict, key=lambda x: -my_dict[x])
        for ele in temp:
            res += ele*my_dict[ele]
        return res

# 1109航班预定统计
class Solution_1109(object):
    def corpFlightBookings(bookings, n):
        df = [0 for _ in range(n)]
        for first, last, seats in bookings:
            df[first-1] += seats
            if last < n:
                df[last] -= seats
        res = [0]*n
        res[0] = df[0]
        for i in range(1, n):
            res[i] = res[i-1]+df[i]
        return res
    
# 567字符串的排列 （异位词的出现导致了我们的窗口的长度必须是固定的）
class Solution_567(object):
    def checkInclusion(s1, s2):
        if len(s2)<len(s1):
            return False
        need = collections.Counter(s1)
        left, right = 0, 0
        valid = len(s1)
        for right in range(len(s2)):
            ele = s2[right]
            if ele in need:
                if need[ele] > 0:
                    valid -= 1
                need[ele] -= 1
            left = right - len(s1)
            if left >= 0:
                temp = s2[left]
                if temp in need:
                    if need[temp] >= 0:
                        valid += 1
                    need[temp] += 1
            if valid == 0:
                return True
        return False
    
# 438找到字符串中所有字母异位词
class Solution_438(object):
    def findAnagrams(s, p):
        res = []
        if len(s) < len(p):
            return []
        need = collections.Counter(p)
        left, right = 0, 0
        valid = len(p)
        for right in range(len(s)):
            ele = s[right]
            if ele in need:
                if need[ele]>0:
                    valid -= 1
                need[ele] -= 1
            left = right - len(p)
            if left>=0:
                temp = s[left]
                if temp in need:
                    if need[temp]>=0:
                        valid += 1
                    need[temp] += 1
            if valid == 0:
                res.append(right-len(p)+1)
        return res
    
# 166分数到小数
class Solution_166(object):
    def fractionToDecimal(numerator, denominator):
        res = ""
        if numerator % denominator == 0:
            return str(numerator // denominator)
        if (numerator<0) ^ (denominator<0):
            res += '-'
        numerator = abs(numerator)
        denominator = abs(denominator)
        integerPart = numerator // denominator
        res += str(integerPart)
        res += '.'
        
        indexMap = {}
        remainder = numerator % denominator
        while remainder and remainder not in indexMap:
            indexMap[remainder] = len(res)
            remainder *= 10
            res += str(remainder // denominator)
            remainder %= denominator
        if remainder:
            insertIndex = indexMap[remainder]
            res = list(res)
            res.insert(insertIndex, '(')
            res.append(')')
        return ''.join(res)

# 329矩阵中的最长递增路径
from functools import lru_cache
class Solution_329(object):
    def longestIncreasingPath(matrix):
        direction = [(-1,0), (1,0), (0,-1), (0,1)]
        m, n = len(matrix), len(matrix[0])
        if not matrix:
            return 0
        @lru_cache(None)
        def dfs(i, j):
            temp = 1
            for dx, dy in direction:
                row, col = i+dx, j+dy
                if 0<=row<m and 0<=col<n and matrix[row][col] > matrix[i][j]:
                    temp = max(temp, dfs(row, col)+1)
            return temp
        res = 0
        for i in range(m):
            for j in range(n):
                res = max(res, dfs(i,j))
        return res

# 207课程表
class Solution_207(object): #三种状态 ： 未搜索，搜索中，搜索完成
    def canFinish(numCourses, prerequisites):
        edges = collections.defaultdict(list)
        visited = [0] * numCourses
        flag = True
        res = []
        for ele in prerequisites:
            edges[ele[1]].append(ele[0])
        def dfs(u):
            nonlocal flag
            visited[u] = 1
            for v in edges[u]:
                if visited[v] == 0:
                    dfs(v)
                    if not flag:
                        return 
                elif visited[v] == 1:
                    flag = False
                    return
            visited[u] = 2
            res.append(u)
        for i in range(numCourses):
            if flag and not visited[i]:
                dfs(i)
        return flag

# 210课程表2
class Solution_210(object):
    def findOrder(numCourses, prerequisites):
        edges = collections.defaultdict(list)
        res = []
        visited = [0]* numCourses
        flag = True

        for ele in prerequisites:
            edges[ele[1]].append(ele[0])
        
        def dfs(u):
            nonlocal flag
            visited[u] = 1
            for v in edges[u]:
                if visited[v] == 0:
                    dfs(v)
                    if not flag:
                        return 
                elif visited[v] == 1:
                    flag = False
                    return 
            visited[u] = 2
            res.append(u)
        
        for i in range(numCourses):
            if flag and not visited[i]:
                dfs(i)
        if not flag:
            return []
        return res[::-1]

# 875 以最小速度吃香蕉
class Solution_875(object):
    def minEatingspeed(piles, h):
        def check(speed):
            return sum([math.ceil(ele/speed) for ele in piles]) <= h
        left = 1
        right = max(piles)
        while left < right:
            mid = left + (right-left)//2
            if check(mid) == True:
                right = mid
            else:
                left = mid+1
        return left
    
# 134加油站
class Solution_134(object):
    def canCompleteCircuit(gas, cost):
        if sum(gas) < sum(cost):
            return -1
        n = len(gas)
        start, cur = 0, 0 #出发点与目前的油量
        for i in range(n):
            cur += gas[i] - cost[i]
            if cur < 0:
                start = i+1
                cur = 0
        return start

# 面试01.06 字符串压缩
class Solution_(object):
    def compressString(S):
        if not S:
            return ""
        ch = S[0]
        res = ""
        count = 0
        for ele in S:
            if ele == ch:
                count += 1
            else:
                res += ch + str(count)
                ch = ele
                count = 1
        res += ch + str(count)
        return res if len(res)<len(S) else S

# 443字符串压缩
class Solution_443(object):
    def compress(chars):
        res = ""
        count = 0
        ch = chars[0]
        for ele in chars:
            if ele == ch:
                count += 1
            else:
                if count == 1:
                    res += ch
                if count > 1:
                    res += ch + str(count)
                ch = ele
                count = 1
        if count == 1:
            res += ch
        else:
            res += ch + str(count)
        return list(res)
    
    def Compress(chars):
        res = []
        i, j = 0, 1
        while i<j and j<len(chars):
            if chars[i] == chars[j]:
                j += 1
            else:
                if j-i>1:
                    res.append(chars[i] + str(j-i))
                else:
                    res.append(chars[i])
                i = j
                j += 1
        if i<len(chars):
            if j-i>1:
                res.append(chars[i]+str(j-i))
            else:
                res.append(chars[i])
        temp = "".join(res)
        L = len(temp)
        for i in range(L): #做替换
            chars[i] = temp[i]
        return L
    
# 1004最长连续1的个数
class Solution_1004(object):
    def longestOnes(nums, k):
        n = len(nums)
        res = 0
        left, right = 0, 0
        zeros = 0
        while right<n:
            if nums[right] == 0:
                zeros += 1
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            res = max(res, right-left+1)
            right += 1
        return res

# 1386电影院排座位
class Solution_1386(object):
    def maxNumberOfFamilies(n, reservedSeats):
        left = 0b11110000
        mid = 0b11000011
        right = 0b00001111
        temp = collections.defaultdict(int)
        for ele in reservedSeats:
            if 2<=ele[1]<=9:
                temp[ele[0]] |= 1<<(9-ele[1])
        res = (n-len(temp)) * 2 #没有占座的情况
        for value in temp.values():
            if value|left == left or value|mid == mid or value|right == right:
                res += 1
        return res

# 397整数替换
class Solution_397(object):
    def integerReplacement(n):
        count = 0
        while n != 1:
            if n%2 == 0:
                n /= 2
                count += 1
            else:
                if n!=3 and (n//2)%2 == 1:
                    n += 1
                else:
                    n -= 1
                count += 1
        return count
    
# 371两个整数之和
class Solution_371(object):
    def getSum(self, a, b):
        MASK1 = 1<<32     # 保证低32位有效，第32位为符号位
        MASK2 = 1<<31     # 补码的最大负数
        #MASK3 = (1<<31)-1 # 最大的正整数
        while b!=0:
            carry = (a&b)<<1 % MASK1
            a = (a^b) % MASK1
            b = carry
        return a if not a&MASK2 else a^(~((1<<32)-1))
    
# 373查找和最小的K对数字
class Solution_373(object):
    def kSmallestPairs(nums1, nums2, k):
        m, n = len(nums1), len(nums2)
        res = []
        heap = [(nums1[i]+nums2[0], i, 0) for i in range(min(k,m))]
        while heap and len(res)<k:
            _, i, j = heapq.heappop(heap)
            res.append([nums1[i], nums2[j]])
            if j+1<n:
                heapq.heappush(heap, (nums1[i]+nums2[j+1], i, j+1))
        return res

# 378有序矩阵中第K小的元素
class Solution_378(object):
    def kthSmallest(matrix, k):
        n = len(matrix)
        heap = [(matrix[i][0], i, 0) for i in range(n)]
        heapq.heapify(heap)
        while heap and len(heap)<k:
            _, i, j = heapq.heappop(heap)
            if j+1<n:
                heapq.heappush(heap, (matrix[i][j+1], i, j+1))
        return heapq.heappop(heap)[0]
    
# 面试题17-21 直方图接水
class Solution_17_21(object):
    def trap(height):
        N = len(height)
        if N<2:
            return 0
        left, right = 0, N-1
        lHeight = rHeight = 0
        res = 0
        while left < right:
            if height[left]<height[right]:
                if height[left]<lHeight:
                    res += lHeight - height[left]
                else:
                    lHeight = height[left]
                left += 1
            else:
                if height[right]<rHeight:
                    res += rHeight - height[right]
                else:
                    rHeight = height[right]
                right -= 1
        return res
    
# 面试题 04.01 节点间通路
class Solution_04_01(object):
    def findWhetherExistPath(n, graph, start, target):
        my_dict = collections.defaultdict(list)
        visited = [False]*n
        for i, j in graph:
            my_dict[i].append(j)
        def dfs(start, target, visited):
            if start == target:
                return True
            if visited[start]:
                return False
            visited[start] = True
            flag = False
            for ele in my_dict[start]:
                if not visited[ele]:
                    flag = flag or dfs(ele, target, visited)
            return flag
        return dfs(start, target, visited)
    
# 面试题 16.19 水域大小
class Solution_16_19(object):
    def pondSizes(land):
        directions = [(0,1), (0,-1), (-1,0), (1,0), (-1,1), (-1,-1), (1,1), (1,-1)]
        m, n = len(land), len(land[0])
        if not land:
            return 0
        res = []
        def dfs(land, i, j):
            if i<0 or i>=m or j<0 or j>=n or visited[i][j] or land[i][j]>0:
                return 0
            visited[i][j] = True
            ans = 1
            for dx, dy in directions:
                row, col = i+dx, j+dy
                ans += dfs(land, row, col)
            return ans
        visited = [[False for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and land[i][j]==0:
                    res.append(dfs(land, i, j))
        return res

# 874模拟机器人行走
class Solution_874(object):
    def robotSim(commands, obstacles):
        directions = [(0,1), (1,0), (0,-1), (-1,0)] #做好上右下左的次序，方便之后的索引
        x = y = index = 0
        obstacles = set(map(tuple, obstacles))
        res = 0
        for cmd in commands:
            if cmd == -1:
                index = (index+1)%4
            elif cmd == -2:
                index = (index-1)%4
            else:
                for _ in range(cmd):
                    if (x+directions[index][0], y+directions[index][1]) not in obstacles:
                        x = x + directions[index][0]
                        y = y + directions[index][1]
                        res = max(res, x*x+y*y)
        return res
    
# 934最短的桥
class Solution_934(object):
    def shortestBridge(grid):
        m, n = len(grid), len(grid[0])
        directions = [(0,1), (0,-1), (-1,0), (1,0)]
        queue = collections.deque()
        def dfs(i,j):
            if i<0 or i>=m or j<0 or j>=n or grid[i][j]==0 or grid[i][j]==2:
                return 
            if grid[i][j] == 1:
                grid[i][j] = 2
                queue.append((i,j))
                for dx, dy in directions:
                    row, col = i+dx, j+dy
                    dfs(row, col)
        def bfs(i,j):
            level = 0
            while queue:
                L = len(queue)
                for _ in range(L):
                    i, j = queue.popleft()
                    for dx, dy in directions:
                        row, col = i+dx, j+dy
                        if row<0 or row>=m or col<0 or col>=n or grid[row][col]==2:
                            continue
                        if grid[row][col] == 1:
                            return level
                        grid[row][col] = 2
                        queue.append((row, col))
                level += 1
        for i, row in enumerate(grid):
            for j, ele in enumerate(row):
                if ele == 1:
                    dfs(i,j)
                    return bfs(i,j)
                
# 1497数组对是否可以被 k 整除
class Solution_1497(object):
    def canArrange(arr, k):
        mod = [0]*k
        for ele in arr:
            mod[ele%k] += 1
        if any(mod[i]!=mod[k-i] for i in range(1, k//2+1)):
            return False
        return mod[0]%2 == 0

# 1755最接近目标值的子序列之和
class Solution_1755(object):
    def minAbsDifference(nums, goal):
        m = len(nums)
        arr1, arr2 = [0], [0]
        length1 = 1
        for i in range(m//2):
            for j in range(length1):
                arr1.append(nums[i]+arr1[j])
                length1 += 1
        length2 = 1
        for i in range(m//2, m):
            for j in range(length2):
                arr2.append(nums[i]+arr2[j])
                length2 += 1
        arr1.sort()
        arr2.sort()
        Min = abs(goal)
        i, j = 0, length2-1
        while i<length1 and j>=0:
            res = arr1[i] + arr2[j]
            Min = min(Min, abs(res-goal))
            if res < goal:
                i += 1
            elif res > goal:
                j -= 1
            else:
                return 0
        return Min

# 1100长度为 K 的无重复字符子串
class Solution_1100(object):
    def numKLenSubstrNoRepeats(S, K):
        res = 0
        for i in range(len(S)-K+1):
            res += len(set(S[i:i+K]))==K
        return res
                    
# 494目标和
class Solution_494(object):
    def findTargetSumWays(nums, target):
        total = sum(nums)
        if abs(target) > total:
            return 0
        if (total + target) % 2 == 1:
            return 0
        pos = (total + target) // 2
        neg = (total - target) // 2
        capacity = min(pos, neg)
        n = len(nums)
        
        dp = [[0 for _ in range(capacity+1)] for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(capacity+1):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
        return dp[n][capacity]

# 18四数之和
class Solution_18(object):
    def fourSum(nums, target):
        res = []
        if len(nums)<4 or not nums:
            return []
        nums.sort()
        n = len(nums)
        for k in range(n-3): #第一个特殊指针
            if k>0 and nums[k] == nums[k-1]:
                continue
            if nums[k] + nums[k+1] + nums[k+2] + nums[k+3] > target: #最小的情况
                break
            if nums[k] + nums[n-3] + nums[n-2] + nums[n-1] < target: #最大的情况
                continue
            for l in range(k+1, n-2): #第二个特殊指针
                if l>k+1 and nums[l] == nums[l-1]:
                    continue
                if nums[k] + nums[l] + nums[l+1] + nums[l+2] > target:
                    break
                if nums[k] + nums[l] + nums[n-2] + nums[n-1] < target:
                    continue
                i, j = l+1, n-1
                while i<j:
                    s = nums[k] + nums[l] + nums[i] + nums[j]
                    if s == target:
                        res.append([nums[k], nums[l], nums[i], nums[j]])
                        i += 1
                        while i<j and nums[i] == nums[i-1]: i += 1
                        j -= 1
                        while i<j and nums[j] == nums[j+1]: j -= 1
                    elif s < target:
                        i += 1
                    else:
                        j -= 1
        return res
                

# 402移掉 K 位数字（剩下最小数字，不改变相对位置）
class Solution_402(object):
    def removeKdigits(num, k):
        stack = []
        remain = len(num)-k
        for ele in num:
            while k and stack and stack[-1]>ele:
                stack.pop()
                k -= 1
            stack.append(ele)
        return "".join(stack[:remain]).lstrip('0') or '0'

# 7整数反转
class Solution_7(object):
    def reverse(x):
        temp = str(x)
        res = ""
        if len(temp) == 0 and temp[0] == '0':
            return 0
        if temp[len(temp)-1] == '0':
            temp = temp[:-1]
        if temp[0] == '-':
            res += '-'
            res += temp[1:][::-1]
        else:
            res += temp[::-1]
        res = int(res)
        if res < (-2**31) or res > (2**31 + 1):
            return 0
        return res

# 43字符串相乘
class Solution_43(object):
    def mutiply(num1, num2):
        if num1 == '0' or num2 == '0':
            return '0'
        if len(num1)<len(num2):
            num1, num2 = num2, num1
        m, n = len(num1), len(num2)  #大数放在上面
        res = 0
        for i in range(n-1, -1, -1): #先固定住下排的数
            x = int(num2[i])
            Sum, carry = "", 0       #当前层的总和与进位情况
            for j in range(m-1, -1, -1):
                y = int(num1[j])
                temp = x*y+carry
                cur = temp % 10
                carry = temp // 10
                Sum = str(cur) + Sum
            if carry>=1:
                Sum = str(carry) + Sum
            res = int(Sum)*pow(10,n-1-i) + res
        return str(res)
    
# 279完全平方数
class Solution_279(object):
    def numSquares(n):
        dp = [0 for _ in range(n+1)]
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = i
            ceil = math.floor(math.sqrt(i))
            for j in range(1, ceil+1):
                k = i-j**2
                if dp[k]+1<dp[i]:
                    dp[i] = dp[k]+1
        return dp[n]
    
    def Knapsack(n):
        nums = []
        i = 1
        while i*i<=n:
            nums.append(i*i)
            i += 1
        dp = [[n+1 for _ in range(n+1)] for _ in range(len(nums)+1)]
        dp[0][0] = 0
        for i in range(1, len(nums)+1):
            for j in range(n+1):
                if j<nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-nums[i-1]]+1)
        return dp[len(nums)][n]

# 593有效的正方形
class Solution_593(object):
    def validSquare(p1, p2, p3, p4):
        distance = {}
        temp = [(p1,p2), (p1,p3), (p1,p4), (p2,p3), (p2,p4), (p3,p4)]
        for ele in temp:
            distance.add((ele[0][0]-ele[1][0])**2 + (ele[0][1]-ele[1][1])**2)
        if len(distance) == 2 and 0 not in distance:
            return True
        return False

# 647回文子串的数量
class Solution_647(object):
    def countSubstrings(s):
        n = len(s)
        res = 0
        for i in range(2*n-1):
            left, right = i//2, i//2+i%2
            while left>=0 and right<n and s[left] == s[right]:
                left -= 1
                right += 1
                res += 1
        return res
    
# 204计数质数
class Solution_204(object):
    def countPrimes(n): #暴力超时
        def isPrime(num):
            if num == 1:
                return False
            for i in range(2, int(math.sqrt(num)+1)):
                if num%i == 0:
                    return False
            return True
        res = 0
        for i in range(1, n):
            if isPrime(i):
                res += 1
        return res
    
    def CountPrimes(n):
        isPrime = [1] * n
        res = 0
        for i in range(2, n):
            if isPrime[i]:
                res += 1
                for j in range(i*i, n, i):
                    isPrime[j] = 0
        return res

# 48旋转图像
class Solution_48(object):
    def rotate(matrix):
        n = len(matrix)
        matrix_new = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix_new[j][n-i-1] = matrix[i][j]
        matrix = matrix_new.copy()
    
# 实现字典树
class Trie:
    def __init__(self):
        self.dict = {}
        self.end = -1
    def insert(self, word):
        node = self.dict
        for ele in word:
            if ele not in node:
                node[ele] = {}
            node = node[ele]
        node[self.end] = True
    def search(self, word):
        node = self.dict
        for ele in word:
            if ele not in node:
                return False
            node = node[ele]
        return self.end in node
    def startsWith(self, prefix):
        node = self.dict
        for ele in prefix:
            if ele not in node:
                return False
            node = node[ele]
        return True
    
# 870优势洗牌

#将nums1升序排列，并利用合适的数据结构每次取出nums2中的最大元素maxq及其索引idx,
#与nums1中的最大元素与maxq比较，如果nums1中对应元素打得过maxq那就自己上res[idx] = nums1
#中当前的最大元素，打不过就派出自己的劣马当炮灰res[idx] = nums1中当前的最小元素。

class Solution_870(object):
    def advantageCount(nums1, nums2):
        heap, n = [], len(nums1)
        for i, ele in enumerate(nums2):
            heapq.heappush(heap, (-ele, i))
        nums1.sort()
        left, right = 0, n-1
        res = [0 for _ in range(n)]
        while heap:
            max_2, index = heapq.heappop(heap) 
            max_2 = -max_2
            if max_2 < nums1[right]:
                res[index] = nums1[right]
                right -= 1
            else:
                res[index] = nums1[left]
                left += 1
        return res
      
# 77组合
class Solution_77(object):
    def combine(n, k):
        res, temp = [], []
        def backtrack(n, k, start):
            if k == len(temp):
                res.append(temp[:])
                return 
            for i in range(start, n-(k-len(temp))+2):
                temp.append(i)
                backtrack(n, k, i+1)
                temp.pop()
        backtrack(n, k, 1)
        return res
    
# 539最小时间差
class Solution_539(object):
    def findMinDifference(timePoints):
        temp = []
        for ele in timePoints:
            time = ele.split(":")
            time = int(time[0])*60+int(time[1])
            temp.append(time)
        N = len(temp)
        temp.sort()
        a = min(temp[i+1]-temp[i] for i in range(N-1))
        b = 1440 - temp[-1] + temp[0]
        return min(a, b)
    
# 881救生艇
class Solution_881(object):
    def numRescueBoats(people, limit):
        people.sort()
        res = 0
        i, j = 0, len(people)-1
        while i<j:
            if people[i] + people[j] > limit:
                res += 1
                j -= 1
            else:
                res += 1
                j -= 1
                i += 1
        if i==j:
            res += 1
        return res

# 75颜色分类
class Solution_75(object):
    def sortColors(nums):
        def quickSort(nums, start, end):
            if start > end:
                return 
            mid_data, left, right = nums[start], start, end
            while left<right:
                while nums[right]>=mid_data and left<right:
                    right -= 1
                nums[left] = nums[right]
                while nums[left] < mid_data and left<right:
                    left += 1
                nums[right] = nums[left]
            nums[left] = mid_data
            quickSort(nums, start, left-1)
            quickSort(nums, right+1, end)
        quickSort(nums, 0, len(nums)-1)
        
# 560和为K的子数组
class Solution_560(object):
    def subarraySum(nums, k):
        preSum = collections.defaultdict(int)
        preSum[0] = 1
        count, temp = 0, 0
        for i in range(len(nums)):
            temp += nums[i]             #保证连续性
            count += preSum[temp-k]     #字典中是否出现了
            preSum[temp] += 1
        return count

# 6 Z字型变换
class Solution_6(object):
    def convert(s, numRows):
        if numRows == 1:
            return s
        res = ["" for _ in range(numRows)]
        sign = -1
        i=0
        for ele in s:
            res[i] += ele
            if i==0 or i==numRows-1:   #属于拐点情况
                sign = -sign           # +表示向下，-表示向上
            i += sign
        return "".join(res)

# 44通配符匹配
class Solution_44(object):
    def isMatch(s, p):
        m, n = len(p), len(s)
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, m+1):
            if p[i-1] == "*":
                dp[i][0] = True
            else:
                break
        for i in range(1, m+1):
            for j in range(1, n+1):
                if p[i-1] == "*":
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif p[i-1] == "?" and s[j-1].isalnum():
                    dp[i][j] = dp[i-1][j-1]
                elif p[i-1] == s[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        return dp[m][n]
    
# 233数字1的个数
class Solution_233(object):
    def countDigitOne(n):
        base, count, k = 1, 0, 1
        former, latter, cur = n//10, 0, n%10
        if n<=0:
            return 0
        while n//base != 0:
            cur = (n//base)%10
            former = n//(base*10)
            latter = n - n//base*base
            if cur>k:
                count += (former+1) * base
            elif cur == k:
                count += former * base + latter + 1
            else:
                count += former * base
            base *= 10
        return count

# 820单词的压缩编码
class Solution_820(object):
    def minimumLengthEncoding(words):
        temp = set(words)
        for ele in words:
            for k in range(1, len(ele)):
                temp.discard(ele[k:])
        return sum(len(ele)+1 for ele in temp)
    
    def MinimumLengthEncoding(words):
        N = len(words)
        temp = []
        for word in words:
            temp.append(word[::-1])
        temp.sort()
        res = 0
        for i in range(N):
            if i+1 < N and temp[i+1].startswith(temp[i]):
                pass
            else:
                res += len(temp[i]) + 1
        return res
    
# 172阶乘后尾随零的数量
class Solution_172(object):
    def trailingZeros(n):
        res = 0
        while n:
            n //= 5
            res += n
        return res
    
# 39组合总和
class Solution_39(object):
    def combinationSum(candidates, target):
        candidates.sort()
        res = []
        def backtrack(index, target, temp):
            if target == 0:
                res.append(temp)
            else:
                for i in range(index, len(candidates)):
                    if candidates[i]>target:   #进行剪枝
                        break
                    else:
                        backtrack(i, target-candidates[i], temp+[candidates[i]])
        backtrack(0, target, [])
        return res

# 79单词搜索
class Solution_79(object):
    def exist(board, word):
        m,n = len(board), len(board[0])
        directions = [(0,1), (0,-1), (-1,0), (1,0)]
        def dfs(i, j, index):
            if board[i][j] != word[index]:
                return False
            if index == len(word) - 1:
                return True
            board[i][j] = '0'
            for dx, dy in directions:
                row, col = i+dx, j+dy
                if 0<=row<m and 0<=col<n and dfs(row, col, index+1):
                    return True
            board[i][j] = word[index]
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
    
# 223计算矩形的面积
class Solution_223(object):
    def computeArea(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        s1 = abs(ax2-ax1) * abs(ay2-ay1)
        s2 = abs(bx2-bx1) * abs(by2-by1)
        x = max(0, min(ax2, bx2) - max(ax1, bx1))
        y = max(0, min(ay2, by2) - max(ay1, by1))
        overlap = x*y
        return s1+s2-overlap
    
# 468验证IP地址
import re
class olution_468(object):
    def validIPAddress(queryIP):
        def ip4(queryIP):
            temp = queryIP.split(".")
            sign = 0
            if len(temp) != 4:
                return False
            for ele in temp:
                if ele.isdigit() and 0<=int(ele)<=255:
                    if ele.startswith("0") and len(ele)>1:
                        sign = sign
                    else:
                        sign += 1
            return sign==4
            
        def ip6(queryIP):
            temp = queryIP.split(":")
            if len(temp) != 8:
                return False
            if all(re.match(r"[0-9a-fA-F]{1,4}$", ele) for ele in temp):
                return True
        
        if ip4(queryIP):
            return "IPv4"
        elif ip6(queryIP):
            return "IPv6"
        else:
            return "Neither"
    
# 142环形链表2
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution_142:
    def detectCycle(self, head:ListNode):
        fast, slow = head, head  #快慢节点用来判断是否存在环
        while True:
            if not (fast and fast.next):
                return 
            fast, slow = fast.next.next, slow.next
            if fast == slow: #相遇的则退出循环，重新从头开始
                break
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast
    
# 652寻找相同的子树
class Solution_652:
    def findDuplicateSubtrees(root:TreeNode):
        hashTable = collections.defaultdict(int)
        res = []
        def dfs(node:TreeNode):
            if node is None:
                return ""
            s = str(node.val) + "," + dfs(node.left) + "," + dfs(node.right)
            hashTable[s] += 1
            if hashTable[s] == 2:
                res.append(node)
            return s
        dfs(root)
        return res
    
# 97交错字符串
class Solution_97(object):
    def isInterleave(s1, s2, s3):
        m, n, l = len(s1), len(s2), len(s3)
        if m+n != l:
            return False
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, m+1):
            dp[i][0] = dp[i-1][0] and s1[i-1]==s3[i-1]
        for j in range(1, n+1):
            dp[0][j] = dp[0][j-1] and s2[j-1]==s3[j-1]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = (dp[i][j-1] and s2[j-1]==s3[i+j-1]) or (dp[i-1][j] and s1[i-1]==s3[i+j-1])
        return dp[m][n]

# 1277统计全为 1 的正方形子矩阵
class Solution_1277(object):
    def countSquares(matrix):
        if not matrix: return 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        count = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j]:
                    dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
                    count += dp[i+1][j+1]
        return count

# 870矩形面积2
class Solution_850(object):
    def rectangleArea(rectangles):
        MOD = 10**9+7
        X, Y = set(),set()
        for ele in rectangles:
            X.add(ele[0]); X.add(ele[2])
            Y.add(ele[1]); Y.add(ele[3])
        tempX = list(X)
        tempX.sort()
        tempY = list(Y)
        tempY.sort()
        mapX = {x:i for i, x in enumerate(tempX)}
        mapY = {y:j for j, y in enumerate(tempY)}
        area = [[0 for _ in range(len(mapY))] for _ in range(len(mapX))]
        
        for x1, y1, x2, y2 in rectangles:
            for x in range(mapX[x1], mapX[x2]):
                for y in range(mapY[y1], mapY[y2]):
                    area[x][y] = 1
        res = 0
        for i in range(len(mapX)):
            for j in range(len(mapY)):
                if area[i][j]:
                    res += (tempX[i+1]-tempX[i]) * (tempY[j+1]-tempY[j])
        return res % MOD

# 174地下城游戏
class Solution_174(object):
    def calculateMininumHP(dungeon):
        m, n = len(dungeon), len(dungeon[0])
        BIG = 10**9
        dp = [[BIG for _ in range(n+1)] for _ in range(m+1)]
        dp[m][n-1] = dp[m-1][n] = 1
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                Min = min(dp[i+1][j], dp[i][j+1])
                dp[i][j] = max(Min-dungeon[i][j], 1)
        return dp[0][0]

# 组合总和4
class Solution_(object):
    def combinationSum4(nums, target):
        if not nums:
            return 0
        dp = [0] * (target+1)
        dp[0] = 1
        for i in range(1, target+1):
            for num in nums:
                if i>=num:
                    dp[i] += dp[i-num]
        return dp[target]