class Solution(object):
    def verifyPostorder(self, postorder):
        """
        :type postorder: List[int]
        :rtype: bool
        """
        if not postorder or len(postorder) <= 2:
            return True
        pre = True
        mid = 0
        for i, n in enumerate(postorder[:-1]):
            if n < postorder[-1]:
                if pre:
                    continue
                else:
                    return False
            else:
                if pre:
                    mid = i
                    pre = False
        left = self.verifyPostorder(postorder[:mid])
        right = self.verifyPostorder(postorder[mid:])
        if left and right:
            return True
        return False
