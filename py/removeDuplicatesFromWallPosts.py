

basePath = '/homes/pgurumur/dan/labelPrediction/data/'

f = open(basePath + 'all-wall-entries.txt')
lines = f.readlines()
f.close()

removeDuplicates = set()

#this is to preserve the ordering of entries, sets destroy the ordering of lines.
removeDuplicatesToFile = []
c = 0
c1 = 0
c2 = 0
for line in lines:
	t = line.strip().split(' ')
	c1+=1
	if line not in removeDuplicates:
		removeDuplicates.add( line )
		removeDuplicatesToFile.append(line)
		#print "added"
	else:
		if line in removeDuplicates:
			c+=1
			#print "true",c
		else:
			c2+=1
			#print "false"

print c1,c,c2


f = open(basePath + 'unique-wall-entries.txt','w')

for line in removeDuplicatesToFile:
	f.write(line)

f.close()
