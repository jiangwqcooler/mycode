#
# @lc app=leetcode.cn id=410 lang=python3
#
# [410] 分割数组的最大值
#

# @lc code=start
# class Solution:

#     def helper(self, nums, m):
#         if nums == []:
#             return 0
#         elif m == 1:
#             return sum(nums)
#         else:
#             min_result = float('inf')
#             for i in range(1, len(nums)+1):
#                 left, right = sum(nums[:i]), self.helper(nums[i:], m-1)
#                 min_result = min(min_result, max(left, right))
#             return min_result
#     def splitArray(self, nums, m):
#         """
#         :type nums: List[int]
#         :type m: int
#         :rtype: int
#         """
#         return self.helper(nums, m)
# @lc code=end

