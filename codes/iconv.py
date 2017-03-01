import os

files = os.listdir('./')
print files

Markers = ['.py','.cpp','.c','.java']
filePaths = []
extensions = []

for fileObject in files:
	filePath ,extension = os.path.splitext(fileObject)
	if extension in Markers:
		filePaths.append(filePath)
		extensions.append(extension)

print filePaths
print extensions


for i in range(len(filePaths)):
	oldPath = filePaths[i] + extensions[i]
	newPath = filePaths[i] + '_utf' + extensions[i]
	command = 'iconv -f gbk -t utf-8 '
	os.system(command + oldPath + ' > ' + newPath)







