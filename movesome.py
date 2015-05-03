import shutil, glob, random, os 

images = []

for file in glob.glob("*.JPEG"):
    images.append(file)

totalnum = len(images)
trainnum = 10000
left = totalnum - trainnum
testnum = int(left / 2)
valnum = (left - testnum)

random.shuffle(images)

trainimages = images[0:trainnum]
valimages = images[trainnum:trainnum+valnum]
testimages = images[-testnum:]


# move images to their repsective folders
for img in trainimages:
        shutil.move(img, './train/')

for img in valimages:
        shutil.move(img, './val/')

for img in testimages:
        shutil.move(img, './test/')






#print(len(trainimages))
#print(len(valimages))
#print(len(testimages))
#print("\n".join(images))
