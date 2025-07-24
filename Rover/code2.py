def max_subarray_sum(nums, k):
    for i, num in enumerate(nums):
        if num % 2 == 0:
            nums[i] -= 1
    nums.sort()
    max_sum = 0
    for i in range(len(nums) - k + 1):
        if len(set(nums[i:i+k])) == k:
            max_sum = max(max_sum, sum(nums[i:i+k]))
    return max_sum

if __name__ == "__main__":
    n,k = input().split()
    k = int(k)
    
    nums = list(map(int, input().split()))


    print(max_subarray_sum(nums, k))