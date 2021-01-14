# coding  : utf-8
# fun     : Leetcode 007 整数反转
# @Author : Labyrinthine Leo
# @Time   : 2021.01.14

class Solution:
	def reverse(self, x: int) -> int:
		k = -1 if x < 0 else 1 # 系数
		x = -x if x < 0 else x # 取正
		y = 0 # 初始值
		while x >= 10:
			y = (y + x%10) * 10 # 取尾部余数求值
			x = x // 10
		y = k * (y + x)
		if y < -(2**31) or y > (2**31 - 1): # 特判
			return 0
		else:
			return y

# 测试用例
x = 1534236469
s = Solution()
print(s.reverse(x))



