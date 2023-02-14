#%%
import os
from enum import Enum
import time
import threading


class Counter:
    Line_numbers = 0
    Code = 0
    total_comment_numbers = 0
    Blanks = 0

    def get_filelist(path, Filelist):
        extendList = [".mqh", ".mq5", ".py"]
        newPath = path

        if os.path.isfile(path) and os.path.splitext(path)[1] in extendList:
            # 文件扩展名属于列表中其中一种时，文件路径添加到filelist中
            Filelist.append(path)

        elif os.path.isdir(path):
            # 路径为目录时，遍历目录下的所有文件和目录
            for s in os.listdir(path):
                newPath = os.path.join(path, s)
                Counter.get_filelist(newPath, Filelist)

        return Filelist

    def CodeCounter(filename, path):
        codes_numbers = 0
        empty = 0
        comment_numbers = 0

        # 打开文件并获取所有行
        fp = open(filename, encoding='gbk', errors='ignore')
        lines = fp.readlines()

        row_cur_status = Status.Common  # 设置初始状态为Common
        temp = ""

        for line in lines:
            line = temp + line
            line = line.strip("\r\t ")  # 去除两端空白
            if line[-1] == "\\":  # 检查末尾是否有续行符，若有续行符，则保存当前line值，准备与下一行进行拼接
                temp += line[:-1]
                continue
            else:
                temp = ""

            lineLen = len(line)

            if lineLen == 1 and line == '\n':
                # 空行，空行数量+1
                empty += 1
                continue

            skipStep = 0  # 需要跳过的字符数，用于跳过一些符号，例如遇到//时进入行注释状态，跳过到//后面第一个字符
            is_effective_code = False  # 有效代码行标识

            for i in range(lineLen):

                if skipStep != 0:
                    skipStep -= 1
                    continue

                if row_cur_status == Status.Common:
                    # 普通状态下

                    if line[i] == '"' or line[i] == "'":
                        row_cur_status = Status.CharString  # 切换到字符串状态
                        CharStringStart = line[i]  # 记录字符串开始时的标识符，用于判断后续退出位置
                        continue

                    # 检查是否进入行注释状态
                    if i + 1 < lineLen and line[i:i + 2] == '//':
                        row_cur_status = Status.LineComment  # 切换到行注释状态
                        skipStep = 1
                        continue

                    # 检查是否进入块注释状态
                    if i + 1 < lineLen and line[i:i + 2] == '/*':
                        row_cur_status = Status.BlockComments  # 切换到块注释状态
                        skipStep = 1
                        continue

                    if line[i] == '\n':
                        continue
                    if line[i] == ' ':
                        continue
                    else:
                        is_effective_code = True  # 代码行有效
                        continue

                elif row_cur_status == Status.CharString:
                    # 字符串状态下
                    if line[i] == CharStringStart:
                        row_cur_status = Status.Common  # 字符串结束，切换回普通状态
                        is_effective_code = True  # 字符串也属于有效代码
                        continue
                    else:
                        continue

                elif row_cur_status == Status.BlockComments:
                    # 块注释状态下
                    # 检查是否退出块注释状态
                    if i + 1 < lineLen and line[i:i + 2] == '*/':
                        # 退出块注释，注释行加上块注释的最后一行，切换回普通状态
                        comment_numbers += 1
                        row_cur_status = Status.Common
                        skipStep = 1
                        continue
                    else:
                        continue

            # 单行遍历结束后，以当前状态记录行数
            # 代码行有效，有效代码行数+1
            if is_effective_code == True:
                codes_numbers += 1

            # 当前状态为块注释或行注释状态下，注释代码行数+1
            if row_cur_status in (Status.BlockComments, Status.LineComment):
                comment_numbers += 1

            # 当前状态不为块注释时，进入下一行前，初始化当前状态
            if row_cur_status != Status.BlockComments:
                row_cur_status = Status.Common

        total = len(lines)

        if (lines[-1][-1] == '\n'):
            total += 1
            empty += 1

        fp.close()

        print("file:{0} total:{1} empty:{2} effective:{3} comment:{4}".format(filename.replace(path + "\\", ""), total, empty, codes_numbers, comment_numbers))

        Counter.Line_numbers += total
        Counter.Blanks += empty
        Counter.Code += codes_numbers
        Counter.total_comment_numbers += comment_numbers


#%%
if __name__ == "__main__":
    path = r"C:\Users\i2011\AppData\Roaming\MetaQuotes\Terminal\6E8A5B613BD795EE57C550F7EF90598D\MQL5\Include\My_Include"
    # path = r"C:\Users\i2011\AppData\Roaming\MetaQuotes\Terminal\6E8A5B613BD795EE57C550F7EF90598D\MQL5\Indicators\My_Indicators"
    # path = r"C:\Users\i2011\AppData\Roaming\MetaQuotes\Terminal\6E8A5B613BD795EE57C550F7EF90598D\MQL5\Experts\My_Experts"
    # path = r"C:\Users\i2011\AppData\Roaming\MetaQuotes\Terminal\6E8A5B613BD795EE57C550F7EF90598D\MQL5\Scripts\My_Scripts"


    # path = r"C:\Users\i2011\AppData\Local\Programs\Python\Python37\Lib\site-packages\MyPackage"
    # path = r"C:\Users\i2011\PycharmProjects\PythonLearning"

    list = Counter.get_filelist(path, [])
    threads = []

    # 将可能遇到的情况枚举
    # Common:表示普通状态
    # CharString:表示字符串状态
    # LineComment:表示行注释状态
    # BlockComments:表示块注释状态
    Status = Enum('Status', 'Init Common CharString LineComment BlockComments')

    for file in list:
        t = threading.Thread(target=Counter.CodeCounter, args=(file, path))
        threads.append(t)

    for thr in threads:
        thr.start()

    for the in threads:
        thr.join()

    time.sleep(0.1)

    print("-" * 56)
    print("- {0:<10} {1:<10} {2:<10} {3:<10} {4:<10}".format("Files", "Lines", "Code", "Comments", "Blanks"))
    print("-" * 56)
    print("  {0:<10} {1:<10} {2:<10} {3:<10} {4:<10}".format(len(list), Counter.Line_numbers, Counter.Code, Counter.total_comment_numbers, Counter.Blanks))
    print("-" * 56)

