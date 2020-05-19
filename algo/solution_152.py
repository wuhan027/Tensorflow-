class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tmax = nums[0]
        tmin = nums[0]
        result = nums[0]
        for n in nums[1:]:
            ttmax, ttmin = tmax, tmin
            tmax = max(n, n*ttmax, n*ttmin)
            tmin = min(n, n*ttmax, n*ttmin)
            result = max(tmax, result)
            print(tmax, tmin, result)
        return result


sol = Solution()
print(sol.maxProduct([-4, -3, -2]))
