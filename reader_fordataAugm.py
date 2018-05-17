import os, pickle
import numpy as np

# 데이터 몇배 augmentation한지
n_augment = 4
chop_len = 198

# 데이터 읽기
dir = os.getcwd()
# f = open(dir + '/barking' + '/data_noiseAugmentation.pkl','rb')
f = open(dir + '/16k_50ms_25ms_128.pkl','rb')
data = pickle.load(f) #[7974,]=[1996*4,]
data = np.expand_dims(data,axis=0)
data = np.reshape(data,(-1,4)) # [N,4]=[1996,4]

### melSpec 뽑기 & testData 뽑기 & 경계(1)만 2배로 data balancing

# np.random.shuffle(data) 여기서는 절대 하면 안됨! 4의 배수의 index마다 같은 오디오로 부터 저장되어 있기 때문에!
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
trainLabel = []
trainSpec = []
testLabel = []
testSpec = []
balancing = 2

for specX in data:
    label = specX[2]['emotionType']

    if label == 1:
        cnt1 += 1
        if cnt1 > 25:
            for i in range(n_augment):
                spec = np.split(specX[i]['melSpec'],[chop_len,chop_len*2,chop_len*3],axis=1)
                for s in spec:
                    if s.shape[1] == chop_len:
                        tempx = np.reshape(s,[s.shape[0]*chop_len])
                        trainLabel.append(np.ones([s.shape[0]*chop_len])*label)
                        trainSpec.append(tempx)
        else:
            s = np.split(specX[0]['melSpec'],[chop_len],axis=1)
            s = s[0]
            if s.shape[1] == chop_len:
                tempx = np.reshape(s,[s.shape[0]*chop_len])
                testLabel.append(np.ones([s.shape[0]*chop_len])*label)
                testSpec.append(tempx)

    elif label == 2:
        cnt2 += 1
        if cnt2 > 25:
            for i in range(n_augment):
                spec = np.split(specX[i]['melSpec'],[chop_len,chop_len*2,chop_len*3],axis=1)
                for s in spec:
                    if s.shape[1] == chop_len:
                        tempx = np.reshape(s,[s.shape[0]*chop_len])
                        trainLabel.append(np.ones([s.shape[0]*chop_len])*label)
                        trainSpec.append(tempx)
        else:
            s = np.split(specX[i]['melSpec'],[chop_len],axis=1)
            s = s[0]
            if s.shape[1] == chop_len:
                tempx = np.reshape(s,[s.shape[0]*chop_len])
                testLabel.append(np.ones([s.shape[0]*chop_len])*label)
                testSpec.append(tempx)

    elif label == 3:
        cnt3 += 1
        if cnt3 > 25:
            for i in range(n_augment):
                spec = np.split(specX[i]['melSpec'],[chop_len,chop_len*2,chop_len*3],axis=1)
                for _ in range(balancing):
                    for s in spec:
                        if s.shape[1] == chop_len:
                            tempx = np.reshape(s,[s.shape[0]*chop_len])
                            trainLabel.append(np.ones([s.shape[0]*chop_len])*label)
                            trainSpec.append(tempx)
        else:
            s = np.split(specX[i]['melSpec'],[chop_len],axis=1)
            s = s[0]
            if s.shape[1] == chop_len:
                tempx = np.reshape(s,[s.shape[0]*chop_len])
                testLabel.append(np.ones([s.shape[0]*chop_len])*label)
                testSpec.append(tempx)
    
    else:
        cnt4 += 1
        if cnt4 > 25:
            for i in range(n_augment):
                spec = np.split(specX[i]['melSpec'],[chop_len,chop_len*2,chop_len*3],axis=1)
                for _ in range(balancing):
                    for s in spec:
                        if s.shape[1] == chop_len:
                            tempx = np.reshape(s,[s.shape[0]*chop_len])
                            trainLabel.append(np.ones([s.shape[0]*chop_len])*label)
                            trainSpec.append(tempx)
        else:
            s = np.split(specX[i]['melSpec'],[chop_len],axis=1)
            s = s[0]
            if s.shape[1] == chop_len:
                tempx = np.reshape(s,[s.shape[0]*chop_len])
                testLabel.append(np.ones([s.shape[0]*chop_len])*label)
                testSpec.append(tempx)

trainSpec = np.expand_dims(np.asarray(trainSpec),axis=1)
trainLabel = np.expand_dims(np.asarray(trainLabel),axis=1)
testSpec = np.expand_dims(np.asarray(testSpec),axis=1)
testLabel = np.expand_dims(np.asarray(testLabel),axis=1)

trainData = np.concatenate((trainSpec,trainLabel), axis=1)
testData = np.concatenate((testSpec,testLabel), axis=1)
