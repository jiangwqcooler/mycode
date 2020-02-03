class Solution:
    def isMatch(self, str_s, str_p):

        if str_s is None:
            return True
        if str_p is None and str_s is not False:
            return False


        arr_p = list(str_p)
        arr_s = list(str_s)

        if '*' not in arr_p and len(arr_p) < len(arr_s):
            return False

        is_match = False
        mat_str = ''
        len_arr_p = len(arr_p)
        for index_p, p in enumerate(arr_p):
            tmp_index_p = index_p
            last_match_state = False
            is_next = True
            for index_s, s in enumerate(arr_s):
                if arr_p[tmp_index_p] == s or arr_p[tmp_index_p] == '.':
                    mat_str += s
                    last_match_state = True
                elif arr_p[tmp_index_p] != s and last_match_state and arr_p[tmp_index_p] == '*':
                    mat_str += s
                    last_match_state = True
                elif last_match_state and arr_p[tmp_index_p] == '*' and arr_p[index_p - 1] == '.':
                    mat_str += s
                    last_match_state = True
                    is_next = False
                else:
                    break
                if is_next and tmp_index_p < len_arr_p - 1:
                    tmp_index_p += 1
                else:
                    break
            
            if mat_str == str_s:
                break
        
        print(mat_str, str_s)
        if mat_str == str_s:
            return True
        else:
            return False
                    

if __name__ == "__main__":

    sol = Solution()
    res = sol.isMatch('mississippi', 'mis*is*p*.')

    print(res)