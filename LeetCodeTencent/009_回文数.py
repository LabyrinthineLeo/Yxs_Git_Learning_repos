# coding  : utf-8
# fun     : Leetcode 009 回文数
# @Author : Labyrinthine Leo
# @Time   : 2021.01.13

class Solution:
    def isPalindrome(self, x: int) -> bool:
    	if x < 0:
    		return False
 
    	reverse_list = [] # 将数值各位转为列表存储
    	while x > 0:
    		reverse_list.append(x%10)
    		x //=10
    	# print(reverse_list)
    	l, r = 0, len(reverse_list)-1
    	while l < r: # 首尾双指针比较
    		if reverse_list[l] != reverse_list[r]:
    			return False
    		l += 1
    		r -= 1
    	return True

# 测试样例
x = 121
s = Solution()
print(s.isPalindrome(x))

# 字符串求解
class Solution:
    def isPalindrome(self, x: int) -> bool:
    	s = str(x)
    	l = len(s)
    	h = l // 2
    	return s[:h] == s[-1:-h-1:-1] # 顺序i+逆序j = -1, 0+-1=-1

# 测试样例
x = -121
s = Solution()
print(s.isPalindrome(x))