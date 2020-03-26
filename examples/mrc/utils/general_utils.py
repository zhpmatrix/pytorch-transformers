class GeneralUtils():
    def is_chinese(self, uchar):
        if uchar >= '\u4e00' and uchar<='\u9fa5':
            return True
        return False
    def is_alphabet(self, uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= '\u0041' and uchar<='\u005a') or (uchar >= '\u0061' and uchar<='\u007a'):
            return True
        else:
            return False
