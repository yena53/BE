'''
# 데이터 분포 확인
# 'dogType': 여기서 경계(1), 화남(2), 슬픔(3), 행복(4)
cnt = 0
for i in range(len(data)):
    if data[i]['dogType'] == 4:
        cnt += 1

print(cnt)

# 396~399의 길이를 가진 오디오!
cnt = 0
for i in range(len(data)):
    if data[i]['melSpec'].shape[1] >= 396:
        if data[i]['melSpec'].shape[1] < 400:
            cnt += 1

print(cnt)
# 224!
# 정작 400(10sec)인것은 4개 밖에 안됨! 198로 잘라야되는 이유!
########################################################################
# 경계는 280, 화남은 952, 슬픔은 430, 행복은 338
# 198랑 같거나 큰 오디오의 수
# 경계는 280, 화남은 946, 슬픔은 430, 행복은 337
# 396보다 같거나 큰 오디오의 수
# 경계는 211, 화남은 622, 슬픔은 317, 행복은 251
# 594랑 같거나 큰 오디오의 수
# 경계는 6, 화남은 11, 슬픔은 5, 행복은 1 --> 따라서 그냥 [198,396,594]까지만 자르기
'''

import os, pickle
import numpy as np

# 데이터 읽기
dir = os.getcwd()
f = open(dir + '/dataset' + '/16k_50ms_25ms_128.pkl','rb')
data = pickle.load(f)

### melSpec 뽑기 & testData 뽑기 & 경계(1)만 2배로 data balancing

np.random.shuffle(data)

def make_dataset(cnt,specX,chop_len,balancing,label,trainLabel, trainSpec, testLabel, testSpec):
    if cnt > 25:
        make_trainset(cnt,specX,chop_len,balancing,label,trainLabel,trainSpec)
    else:
        make_testset(cnt,specX,chop_len,label,testLabel,testSpec)

def make_trainset(cnt,specX,chop_len,balancing,label,trainLabel,trainSpec):
    for X in specX:
        if X.shape[1] == chop_len:
            for i in range(balancing):
                tempX = np.reshape(X,[1,X.shape[0]*chop_len])
                trainLabel.append(np.ones([1,X.shape[0]*chop_len])*label)
                trainSpec.append(tempX)

# test dataset을 만들때, 이 데이터가 이 오디오인지 바로 소리 들을 수 있게 다시 짜야할듯!
def make_testset(cnt,specX,chop_len,label,testLabel,testSpec):
    X = specX[0]
    if X.shape[1] == chop_len:
        tempX = np.reshape(X,[1,X.shape[0]*chop_len])
        testLabel.append(np.ones([1,X.shape[0]*chop_len])*label)
        testSpec.append(tempX)

def dataset():
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    trainLabel = []
    trainSpec = []
    testLabel = []
    testSpec = []

    chop_len = 198
    for specFile in data:
        label = specFile['dogType']
        specX = np.split(specFile['melSpec'],[chop_len,chop_len*2,chop_len*3],axis=1)
        
        if label == 1:
            balancing = 2
            cnt1 += 1
            make_dataset(cnt1,specX,chop_len,balancing,label,trainLabel, trainSpec, testLabel, testSpec)
        elif label == 2:
            balancing = 1
            cnt2 += 1
            make_dataset(cnt2,specX,chop_len,balancing,label,trainLabel, trainSpec, testLabel, testSpec)
        elif label == 3:
            balancing = 1
            cnt3 += 1
            make_dataset(cnt3,specX,chop_len,balancing,label,trainLabel, trainSpec, testLabel, testSpec)
        else:
            balancing = 1
            cnt4 += 1
            make_dataset(cnt4,specX,chop_len,balancing,label,trainLabel, trainSpec, testLabel, testSpec)

    trainSpec = np.asarray(trainSpec)
    trainLabel = np.asarray(trainLabel)
    testSpec = np.asarray(testSpec)
    testLabel = np.asarray(testLabel)
    trainData = np.concatenate((trainSpec,trainLabel), axis=1)
    testData = np.concatenate((testSpec,testLabel), axis=1)

    return trainData, testData
