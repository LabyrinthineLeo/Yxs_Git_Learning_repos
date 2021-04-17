# coding  : utf-8
# fun     : Leetcode 002 两数相加
# @Author : Labyrinthine Leo
# @Time   : 2021.01.08

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class SolutionTest:
    # def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    def addTwoNumbers(self, l1, l2):
    	carry_bit = 0
    	new_list = []
    	# 比较长度
    	if len(l1) > len(l2):
    		for i in range(len(l1) - len(l2)):
    			l2.append(0)
    	else:
    		for i in range(len(l2) - len(l1)):
    			l1.append(0)
    	# 打印测试
    	print(l1)
    	print(l2)
    	# 进位计算
    	for i in range(len(l1)):
    		new_list.append((l1[i] + l2[i] + carry_bit) % 10) # 记录该位新值
    		carry_bit = (l1[i] + l2[i] + carry_bit) // 10 # 记录进位
    	# 最后一位
    	if carry_bit != 0:
    		new_list.append(carry_bit)

    	return new_list

# 测试样例
l1 = [2, 4, 3]
l2 = [5, 6, 4]
l3 = [0]
l4 = [0]
l5 = [9, 9, 9, 9, 9, 9, 9]
l6 = [9, 9, 9, 9]
s = SolutionTest()
print(s.addTwoNumbers(l5, l6))

# -------------------- Solution --------------------
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    	carry_bit = 0
    	new_list = []
    	while l1 and l2:
    		new_list.append((l1.val + l2.val + carry_bit) % 10) # 记录该位新值
    		carry_bit = (l1.val + l2.val + carry_bit) // 10 # 记录进位
    		l1 = l1.next
    		l2 = l2.next
    	# 获取较小数
    	if l1 != None:
    		l = l1
    	else:
    		l = l2
    	while l:
    		new_list.append((l.val + carry_bit) % 10) # 记录该位新值
    		carry_bit = (l.val + carry_bit) // 10 # 记录进位
    		l = l.next
    	# 最后一位
    	if carry_bit != 0:
    		new_list.append(carry_bit)

    	link_list = None
    	for i in new_list[::-1]:
    		link_list = ListNode(i, link_list)
    	return link_list

# 测试样例
l = ListNode(9)
k = ListNode(9)
for i in range(6):
	l = ListNode(9, l)
for i in range(3):
	k = ListNode(9, k)
s = Solution()
x = s.addTwoNumbers(l, k)
while x:
	print(x.val)
	x = x.next
# print(s.addTwoNumbers(l, k))