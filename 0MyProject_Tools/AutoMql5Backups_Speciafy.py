# Author:Zhang Yuan

from MyPackage.MyFile.File import MyClass_File
myfile = MyClass_File()
from MyPackage.MyPath import MyClass_Path
mypath = MyClass_Path()

Mql5Path = "C:\\Users\\i2011\\AppData\\Roaming\\MetaQuotes\\Terminal\\6E8A5B613BD795EE57C550F7EF90598D\\MQL5"
tozip = mypath.get_desktop_path()
print("Mql5Path=",Mql5Path)
print("tozip=",tozip)

myfile.zip_dir(dirpath=Mql5Path, zipPath=tozip, zipName="MQL5_无源码.zip", autoName=True, onlyAffix=None, ignoreAffix=[".mq5", ".mqh"])


