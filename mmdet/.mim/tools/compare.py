import os

j1 = 'results/rtm_mask/exp1_ep50_1280_0.5'
j2 = 'results/rtm_mask/exp1_ep50_1280_0.9'


imgs1 = os.listdir(j1)
imgs2 = os.listdir(j2)

num = 0
print(j1,'有而',j2,'没有')
for img1 in imgs1:
    if img1 not in imgs2:
        print(img1)
        num +=1
print(num)
num = 0
print(j2,'有而',j1,'没有')
for img2 in imgs2:
    if img2 not in imgs1:
        print(img2)
        num +=1
print(num)