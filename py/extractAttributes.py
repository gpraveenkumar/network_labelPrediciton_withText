from collections import Counter

f = open('../data/all-profile-details.txt')
lines = f.readlines()
f.close()

map_id_details = {}

lastId = '0'
flag = 0 

for line in lines:
	line = line.strip().split("->")[1]
	line = line.split("::")

	if line[1] == "Interested In" or line[1] == "Hometown" or line[1] == "Looking For":
		continue
	
	# To remove rows like
	#DETAILS->10001089::Sex::
	#DETAILS->10013686::Networks::

	if len(line[len(line)-1]) == 0:
		continue

	if lastId != line[0]:
		if flag == 0:
			flag = 1
			details = {}
			lastId = line[0]
		else:
			map_id_details[lastId] = details
			details = {}
			lastId = line[0]

	if line[1] == "Networks":
		t = [ '::'.join(line[2:]) ]
	else:
		t = line[2]

	
	if line[1] == "Relationship Status":
		if t == 'It&#039;s Complicated' or t == "It's Complicated":
			t = 'Its Complicated'

	if line[1] == "Relationship Status" and line[1] in details:
		relationshipDomain = ['Single', 'In a Relationship', 'Married', 'Engaged','Its Complicated', 'In an Open Relationship']
		if t in relationshipDomain and details[ line[1] ] not in relationshipDomain:
			details[ line[1] ] = t
		
		continue
		

	details[ line [1] ] = t


print "No. of records:",len(map_id_details)


sex = 0
religiousView = 0
politicalView = 0
relationshipStatus = 0
tempCtr = 0
c_conserv = 0
c_christ = 0
c_female = 0
c = Counter()
c1 = Counter()
c_relationship = Counter()

for id in map_id_details:
	if "Sex" in map_id_details[ id ] and "Political Views" in map_id_details[ id ] and "Religious Views" in map_id_details[ id ] and "Relationship Status" in map_id_details[ id ] and  "Networks" in map_id_details[ id ]:
	#if "Sex" in map_id_details[ id ] and "Political Views" in map_id_details[ id ] and "Religious Views" in map_id_details[ id ] and "Relationship Status" in map_id_details[ id ]:
	#if "Sex" in map_id_details[ id ] and "Political Views" in map_id_details[ id ] and "Religious Views" in map_id_details[ id ]:
	#if map_id_details[ id ]["Relationship Status"] == "In a Relationship":
		print id,map_id_details[ id ]["Relationship Status"]
		if "Sex" in map_id_details[ id ]:
			sex += 1
			if "female" == map_id_details[ id ]["Sex"].lower():
				#print id
				c_female += 1
		if "Political Views" in map_id_details[ id ]:
			politicalView += 1
			c1[map_id_details[ id ]["Political Views"] ] += 1 
			if "conserv" in map_id_details[ id ]["Political Views"].lower():
				#print id
				c_conserv += 1
		if "Religious Views" in map_id_details[ id ]:
			religiousView += 1
			c[map_id_details[ id ]["Religious Views"] ] += 1
			if "christ" in map_id_details[ id ]["Religious Views"].lower():
				#print id
				c_christ += 1
		if "Relationship Status" in map_id_details[ id ]:
			relationshipStatus += 1
			c_relationship[map_id_details[ id ]["Relationship Status"] ] += 1

		tempCtr += 1
#c[ map_id_details[ id ]["Sex"]] += 1
	#c[ map_id_details[ id ]["Relationship Status"] ] +=1

print "Listed Sex:",sex
print "Listed Political Views:",religiousView
print "Listed Religious Views:",politicalView
print "Listed Relationship Status:",relationshipStatus
print "c_conserv:",c_conserv
print "c_christ:",c_christ
print "c_female:",c_female
print c
print len(c)
#print c
#print c1
print c_relationship
print "tempCtr:",tempCtr

sum = 0
for i in c_relationship:
	sum += c_relationship[i]
print sum