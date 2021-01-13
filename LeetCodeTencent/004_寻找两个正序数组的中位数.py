# coding  : utf-8
# fun     : Leetcode 004 寻找两个正序数组的中位数
# @Author : Labyrinthine Leo
# @Time   : 2021.01.11

# 自写归并排序
class Solution:
    # def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    def findMedianSortedArrays(self, nums1, nums2):
        m = len(nums1)
        n = len(nums2)
        nums = [0] * (m+n) # 开辟新数组
        # 判断边界
        if m == 0:
        	if n % 2 == 0:
        		return (nums2[n // 2 - 1] + nums2[n // 2]) / 2
        	else:
        		return nums2[n // 2]

        if n == 0:
        	if m % 2 == 0:
        		return (nums1[m // 2 - 1] + nums1[m // 2]) / 2
        	else:
        		return nums1[m // 2]

        # 归并排序
        count = 0
        i = 0
        j = 0
        while count != (m+n):
        	# 边界条件
        	if i == m: 
        		while j != n:
        			nums[count] = nums2[j]
        			count += 1
        			j += 1
        		break
        	if j == n: 
        		while i != m:
        			nums[count] = nums1[i]
        			count += 1
        			i += 1
        		break

        	if nums1[i] < nums2[j]:
        		nums[count] = nums1[i]
        		count += 1
        		i += 1
        	else:
        		nums[count] = nums2[j]
        		count += 1
        		j += 1

        # 合并后数组判断
        if count % 2 == 0:
        		return (nums[count // 2 - 1] + nums[count // 2]) / 2
        else:
        	return nums[count // 2]



nums1 = [1, 3]
nums2 = [2]
s = Solution()
print(s.findMedianSortedArrays(nums1, nums2))

# 调用sort函数排序
class Solution:
    # def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    def findMedianSortedArrays(self, nums1, nums2):
    	m = len(nums1)
    	n = len(nums2)
    	nums1.extend(nums2) # 两个数组合并
    	nums1.sort() # 排序
    	if (m + n) & 1: # 奇数
    		return nums1[(m + n - 1) >> 1]
    	else:
    		return (nums1[(m + n) >> 1 - 1] + nums1[(m + n) >> 1]) / 2

# 二分策略，划分数组
class Solution:
    # def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    def findMedianSortedArrays(self, nums1, nums2):
    	m, n = len(nums1), len(nums2)
    	if m > n: # 保证m<n
    		return self.findMedianSortedArrays(nums2, nums1)
    	k = (m + n + 1) // 2 # 中间变量，i+j=m+n-i-j或i+j=m+n-i-j+1
    	l, r = 0, m # 二分上下界
    	while l <= r:
    		i = (l + r) // 2 # n1数组划分位置
    		j = k - i # n2数组划分位置
    		if j != 0 and i != m and nums2[j-1] > nums1[i]: # i需要增大
    			l = i + 1
    		elif i != 0 and j != n and nums1[i-1] > nums2[j]: # i需要减小
    			r = i - 1
    		else: # 边界终止条件
    			if i == 0:
    				maxLeft = nums2[j-1]
    			elif j == 0:
    				maxLeft = nums1[i-1]
    			else: # 普通情况
    				maxLeft = max(nums1[i-1], nums2[j-1])
    			if (m + n) & 1: # 总长度为奇数
    				return maxLeft

    			if i == m:
    				minRight = nums2[j]
    			elif j == n:
    				minRight = nums1[i]
    			else: # 普通情况
    				minRight = min(nums1[i], nums2[j])

    			return (maxLeft + minRight) / 2



