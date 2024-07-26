# 1.訓練CNN模型進行貓狗二分類任務
> 測試集準確率需達98%以上。

模型訓練方面使用了兩種模型
**ResNet / ConvNeXt**

## ResNet101
ResNet101是一種深度殘差網絡（Deep Residual Network），它通過引入殘差結構來解決深層網絡中的梯度消失問題。ResNet101包含101個卷積層，能夠學習到豐富的圖像特徵。

```
model = models.resnet101(pretrained=True)
```

**train**

```
num_epochs = 5
best_accuracy = 0
best_model_path = ''
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs,labels in train_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_accuracy = 100 * correct / total
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_path = f'model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), best_model_path)   
        
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, acc: {test_accuracy}')

if best_model_path:
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f'Loaded best model from: {best_model_path}')
else:
    print('No best model found.')
```
Epoch 1, Loss: 0.06807169313670601, acc: 98.22  
Epoch 2, Loss: 0.04478731499036075, acc: 98.16  
Epoch 3, Loss: 0.04207785350538325, acc: 98.68  
Epoch 4, Loss: 0.029907871747447645, acc: 98.32  
Epoch 5, Loss: 0.027521255947111058, acc: 98.38  
Loaded best model from: model_epoch_2.pth  

**val**
```
val_dataset = datasets.ImageFolder('datasets_Cats_vs_Dogs/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model.eval()
corrects = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

accuracy = corrects.double() / total
print(f'Validation Accuracy: {accuracy:.4f}')
```
Validation Accuracy: 0.9825


## ConvNeXt-tiny
ConvNeXt是一種近期提出的改進版CNN模型。它通過對標準CNN進行多種改進，如使用標準化層、改進的激活函數等，來提升模型的性能和穩定性。

```
import timm
import torch.nn as nn

# 加載預訓練的ConvNeXt模型
model = timm.create_model('convnext_tiny', pretrained=True)
```

**train**
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 5
best_accuracy = 0
best_model_path = ''

train_losses = []
test_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # 訓練損失
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # test acc
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
    
 
if best_model_path:
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f'Loaded best model from: {best_model_path}')
else:
    print('No best model found.')
```
Epoch 1, Loss: 0.0604, Test Acc: 98.70  
Epoch 2, Loss: 0.0309, Test Acc: 98.74  
Epoch 3, Loss: 0.0307, Test Acc: 98.46  
Epoch 4, Loss: 0.0223, Test Acc: 98.62  
Epoch 5, Loss: 0.0219, Test Acc: 98.54  
Loaded best model from: 2_ConvNeXt_model_epoch_1.pth  

**val**
```
val_dataset = datasets.ImageFolder('datasets_Cats_vs_Dogs/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model.eval()
corrects = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

accuracy = corrects.double() / total
print(f'Validation Accuracy: {accuracy:.4f}')
```
Validation Accuracy: 0.9880

## 實驗過程
我使用相同的超參數設置對兩個模型進行訓練，包括：

* 學習率: 0.0001
* 批量大小: 32
* 訓練週期：5
* 損失函數: 交叉熵損失`nn.CrossEntropyLoss()`
* 數據增強（Data Augmentation）：
```
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 將圖像大小調整為224x224像素
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉圖像
    transforms.RandomRotation(10),  # 隨機旋轉圖像，最大旋轉角度為10度
    transforms.ToTensor(),  # 將圖像轉換為張量（Tensor）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用均值和標準差進行標準化
])

```
訓練過程中，我們記錄了模型在訓練集和測試集上的準確率和損失值，以評估模型的性能。

## 實驗結果
**準確率**
在訓練後，兩種模型在驗證集上的準確率結果如下：

ResNet101：準確率為 98.25%  
ConvNeXT-tiny：準確率為 98.80%

> 從結果可以看出，兩種模型在貓狗分類任務中的準確率相近，差異微乎其微。

**訓練時間**
訓練時間的比較結果如下：
ResNet101：訓練總時長約30min  
ConvNeXT-tiny：訓練總時長約10min

> ConvNeXT 模型的訓練時間略低於 ResNet101，顯示出其在訓練效率上的潛在優勢。

## 討論
儘管 ResNet101 和 ConvNeXT-tiny 模型在貓狗分類的準確率上差異不大，但在訓練時間上，ConvNeXT 表現出更高的效率。這表明 ConvNeXT 在相同的計算資源下能夠更快速地完成訓練，這對於處理大規模數據集和實時應用尤為重要。未來的研究可以進一步探索 ConvNeXT 在其他計算機視覺任務中的表現以及其對計算資源的需求。

![image](https://hackmd.io/_uploads/By036rDuR.png)
[ConvNext:Github](https://github.com/facebookresearch/ConvNeXt?tab=readme-ov-file)



# 2.對訓練所得到的CNN模型進行特徵分析
> 觀察模型是不是具有可解釋性。
> 在驗證集中，隨機挑選貓/狗各三張進行GradCam分析

## 使用訓練ResNet101所得到的`best model :model_epoch_2.pth`


**Cat1**
![image](https://hackmd.io/_uploads/rkUVILv_0.png)
![image](https://hackmd.io/_uploads/SkCE8Uwd0.png)


**Cat2**
![image](https://hackmd.io/_uploads/B1hCr8wuA.png)
![image](https://hackmd.io/_uploads/HkKJLIvuC.png)


**Cat3**
![image](https://hackmd.io/_uploads/S1TrfIv_C.png)
![image](https://hackmd.io/_uploads/SJl8fIvuA.png)


**Dog1**
![image](https://hackmd.io/_uploads/r1DTLLDO0.png)
![image](https://hackmd.io/_uploads/B1vyv8wOC.png)

**Dog2**
![image](https://hackmd.io/_uploads/S1nHvIv_0.png)
![image](https://hackmd.io/_uploads/SkswwIwO0.png)

**Dog3**
![image](https://hackmd.io/_uploads/SJ26PIv_C.png)
![image](https://hackmd.io/_uploads/rJFAwUPO0.png)


## 使用訓練ConvNeXT-T所得到的`best model :2_ConvNeXt_model_epoch_1.pth`
**Cat1**
![image](https://hackmd.io/_uploads/rJhf88vuR.png)
![image](https://hackmd.io/_uploads/r1bQLIPu0.png)

**Cat2**
![image](https://hackmd.io/_uploads/ry8CHUD_A.png)
![image](https://hackmd.io/_uploads/HJIgI8PdA.png)

**Cat3**
![image](https://hackmd.io/_uploads/S1TrfIv_C.png)
![image](https://hackmd.io/_uploads/Syh7S8Dd0.png)


**Dog1**
![image](https://hackmd.io/_uploads/SJp6UIwu0.png)
![image](https://hackmd.io/_uploads/SyMA88DdA.png)

**Dog2**
![image](https://hackmd.io/_uploads/BJmIDLD_0.png)
![image](https://hackmd.io/_uploads/r1qIvIDdC.png)


**Dog3**
![image](https://hackmd.io/_uploads/BJM0wIPOR.png)
![image](https://hackmd.io/_uploads/H1lyOUv_A.png)

## ResNet vs. ConvNeXt 的熱力圖差異
ResNet 的熱力圖特徵:
> 頭部焦點：ResNet 的熱力圖通常集中在頭部區域，如鼻子或耳朵。這可能是因為 ResNet 在較淺層的特徵圖中對這些明顯的區域有較強的響應，並且這些區域對分類任務非常重要。

ConvNeXt 的熱力圖特徵:
> 細節聚焦：ConvNeXt 的熱力圖則更傾向於細節特徵。例如，對於貓，它們更強調鬍鬚和耳朵，而對於狗，則關注於較長的四肢或尾巴。這可能是因為 ConvNeXt 的架構設計更能捕捉細微的特徵和更深層次的語義信息。



---


--以下任務皆使用ResNet101繼續--



---


# 3.對訓練得到的CNN模型進行拆解(decoupling)
> 將除去FC層後的捲基層結果輸出作為特徵，結合random forest 分類器進行分類預測。
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


model_fc = model.load_state_dict(torch.load('model_epoch_2.pth'))

model_fc = nn.Sequential(*list(model.children())[:-1])


model_fc.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to('cuda')
            outputs = model_fc(inputs)
            outputs = outputs.view(outputs.size(0), -1) # dim = 1
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

# train-rf
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features, train_labels)

# pd
train_predictions = rf_classifier.predict(train_features)
test_predictions = rf_classifier.predict(test_features)

# acc
train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
```
Train Accuracy: 100.00%
Test Accuracy: 98.56%
Validation Accuracy: 98.50%


## model_fc
除去fc層的ResNet101
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]           4,096
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]          16,384
      BatchNorm2d-12          [-1, 256, 56, 56]             512
           Conv2d-13          [-1, 256, 56, 56]          16,384
      BatchNorm2d-14          [-1, 256, 56, 56]             512
             ReLU-15          [-1, 256, 56, 56]               0
       Bottleneck-16          [-1, 256, 56, 56]               0
           Conv2d-17           [-1, 64, 56, 56]          16,384
      BatchNorm2d-18           [-1, 64, 56, 56]             128
             ReLU-19           [-1, 64, 56, 56]               0
           Conv2d-20           [-1, 64, 56, 56]          36,864
      BatchNorm2d-21           [-1, 64, 56, 56]             128
             ReLU-22           [-1, 64, 56, 56]               0
           Conv2d-23          [-1, 256, 56, 56]          16,384
      BatchNorm2d-24          [-1, 256, 56, 56]             512
             ReLU-25          [-1, 256, 56, 56]               0
       Bottleneck-26          [-1, 256, 56, 56]               0
           Conv2d-27           [-1, 64, 56, 56]          16,384
      BatchNorm2d-28           [-1, 64, 56, 56]             128
             ReLU-29           [-1, 64, 56, 56]               0
           Conv2d-30           [-1, 64, 56, 56]          36,864
      BatchNorm2d-31           [-1, 64, 56, 56]             128
             ReLU-32           [-1, 64, 56, 56]               0
           Conv2d-33          [-1, 256, 56, 56]          16,384
      BatchNorm2d-34          [-1, 256, 56, 56]             512
             ReLU-35          [-1, 256, 56, 56]               0
       Bottleneck-36          [-1, 256, 56, 56]               0
           Conv2d-37          [-1, 128, 56, 56]          32,768
      BatchNorm2d-38          [-1, 128, 56, 56]             256
             ReLU-39          [-1, 128, 56, 56]               0
           Conv2d-40          [-1, 128, 28, 28]         147,456
      BatchNorm2d-41          [-1, 128, 28, 28]             256
             ReLU-42          [-1, 128, 28, 28]               0
           Conv2d-43          [-1, 512, 28, 28]          65,536
      BatchNorm2d-44          [-1, 512, 28, 28]           1,024
           Conv2d-45          [-1, 512, 28, 28]         131,072
      BatchNorm2d-46          [-1, 512, 28, 28]           1,024
             ReLU-47          [-1, 512, 28, 28]               0
       Bottleneck-48          [-1, 512, 28, 28]               0
           Conv2d-49          [-1, 128, 28, 28]          65,536
      BatchNorm2d-50          [-1, 128, 28, 28]             256
             ReLU-51          [-1, 128, 28, 28]               0
           Conv2d-52          [-1, 128, 28, 28]         147,456
      BatchNorm2d-53          [-1, 128, 28, 28]             256
             ReLU-54          [-1, 128, 28, 28]               0
           Conv2d-55          [-1, 512, 28, 28]          65,536
      BatchNorm2d-56          [-1, 512, 28, 28]           1,024
             ReLU-57          [-1, 512, 28, 28]               0
       Bottleneck-58          [-1, 512, 28, 28]               0
           Conv2d-59          [-1, 128, 28, 28]          65,536
      BatchNorm2d-60          [-1, 128, 28, 28]             256
             ReLU-61          [-1, 128, 28, 28]               0
           Conv2d-62          [-1, 128, 28, 28]         147,456
      BatchNorm2d-63          [-1, 128, 28, 28]             256
             ReLU-64          [-1, 128, 28, 28]               0
           Conv2d-65          [-1, 512, 28, 28]          65,536
      BatchNorm2d-66          [-1, 512, 28, 28]           1,024
             ReLU-67          [-1, 512, 28, 28]               0
       Bottleneck-68          [-1, 512, 28, 28]               0
           Conv2d-69          [-1, 128, 28, 28]          65,536
      BatchNorm2d-70          [-1, 128, 28, 28]             256
             ReLU-71          [-1, 128, 28, 28]               0
           Conv2d-72          [-1, 128, 28, 28]         147,456
      BatchNorm2d-73          [-1, 128, 28, 28]             256
             ReLU-74          [-1, 128, 28, 28]               0
           Conv2d-75          [-1, 512, 28, 28]          65,536
      BatchNorm2d-76          [-1, 512, 28, 28]           1,024
             ReLU-77          [-1, 512, 28, 28]               0
       Bottleneck-78          [-1, 512, 28, 28]               0
           Conv2d-79          [-1, 256, 28, 28]         131,072
      BatchNorm2d-80          [-1, 256, 28, 28]             512
             ReLU-81          [-1, 256, 28, 28]               0
           Conv2d-82          [-1, 256, 14, 14]         589,824
      BatchNorm2d-83          [-1, 256, 14, 14]             512
             ReLU-84          [-1, 256, 14, 14]               0
           Conv2d-85         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-86         [-1, 1024, 14, 14]           2,048
           Conv2d-87         [-1, 1024, 14, 14]         524,288
      BatchNorm2d-88         [-1, 1024, 14, 14]           2,048
             ReLU-89         [-1, 1024, 14, 14]               0
       Bottleneck-90         [-1, 1024, 14, 14]               0
           Conv2d-91          [-1, 256, 14, 14]         262,144
      BatchNorm2d-92          [-1, 256, 14, 14]             512
             ReLU-93          [-1, 256, 14, 14]               0
           Conv2d-94          [-1, 256, 14, 14]         589,824
      BatchNorm2d-95          [-1, 256, 14, 14]             512
             ReLU-96          [-1, 256, 14, 14]               0
           Conv2d-97         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-98         [-1, 1024, 14, 14]           2,048
             ReLU-99         [-1, 1024, 14, 14]               0
      Bottleneck-100         [-1, 1024, 14, 14]               0
          Conv2d-101          [-1, 256, 14, 14]         262,144
     BatchNorm2d-102          [-1, 256, 14, 14]             512
            ReLU-103          [-1, 256, 14, 14]               0
          Conv2d-104          [-1, 256, 14, 14]         589,824
     BatchNorm2d-105          [-1, 256, 14, 14]             512
            ReLU-106          [-1, 256, 14, 14]               0
          Conv2d-107         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-108         [-1, 1024, 14, 14]           2,048
            ReLU-109         [-1, 1024, 14, 14]               0
      Bottleneck-110         [-1, 1024, 14, 14]               0
          Conv2d-111          [-1, 256, 14, 14]         262,144
     BatchNorm2d-112          [-1, 256, 14, 14]             512
            ReLU-113          [-1, 256, 14, 14]               0
          Conv2d-114          [-1, 256, 14, 14]         589,824
     BatchNorm2d-115          [-1, 256, 14, 14]             512
            ReLU-116          [-1, 256, 14, 14]               0
          Conv2d-117         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-118         [-1, 1024, 14, 14]           2,048
            ReLU-119         [-1, 1024, 14, 14]               0
      Bottleneck-120         [-1, 1024, 14, 14]               0
          Conv2d-121          [-1, 256, 14, 14]         262,144
     BatchNorm2d-122          [-1, 256, 14, 14]             512
            ReLU-123          [-1, 256, 14, 14]               0
          Conv2d-124          [-1, 256, 14, 14]         589,824
     BatchNorm2d-125          [-1, 256, 14, 14]             512
            ReLU-126          [-1, 256, 14, 14]               0
          Conv2d-127         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-128         [-1, 1024, 14, 14]           2,048
            ReLU-129         [-1, 1024, 14, 14]               0
      Bottleneck-130         [-1, 1024, 14, 14]               0
          Conv2d-131          [-1, 256, 14, 14]         262,144
     BatchNorm2d-132          [-1, 256, 14, 14]             512
            ReLU-133          [-1, 256, 14, 14]               0
          Conv2d-134          [-1, 256, 14, 14]         589,824
     BatchNorm2d-135          [-1, 256, 14, 14]             512
            ReLU-136          [-1, 256, 14, 14]               0
          Conv2d-137         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-138         [-1, 1024, 14, 14]           2,048
            ReLU-139         [-1, 1024, 14, 14]               0
      Bottleneck-140         [-1, 1024, 14, 14]               0
          Conv2d-141          [-1, 256, 14, 14]         262,144
     BatchNorm2d-142          [-1, 256, 14, 14]             512
            ReLU-143          [-1, 256, 14, 14]               0
          Conv2d-144          [-1, 256, 14, 14]         589,824
     BatchNorm2d-145          [-1, 256, 14, 14]             512
            ReLU-146          [-1, 256, 14, 14]               0
          Conv2d-147         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-148         [-1, 1024, 14, 14]           2,048
            ReLU-149         [-1, 1024, 14, 14]               0
      Bottleneck-150         [-1, 1024, 14, 14]               0
          Conv2d-151          [-1, 256, 14, 14]         262,144
     BatchNorm2d-152          [-1, 256, 14, 14]             512
            ReLU-153          [-1, 256, 14, 14]               0
          Conv2d-154          [-1, 256, 14, 14]         589,824
     BatchNorm2d-155          [-1, 256, 14, 14]             512
            ReLU-156          [-1, 256, 14, 14]               0
          Conv2d-157         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-158         [-1, 1024, 14, 14]           2,048
            ReLU-159         [-1, 1024, 14, 14]               0
      Bottleneck-160         [-1, 1024, 14, 14]               0
          Conv2d-161          [-1, 256, 14, 14]         262,144
     BatchNorm2d-162          [-1, 256, 14, 14]             512
            ReLU-163          [-1, 256, 14, 14]               0
          Conv2d-164          [-1, 256, 14, 14]         589,824
     BatchNorm2d-165          [-1, 256, 14, 14]             512
            ReLU-166          [-1, 256, 14, 14]               0
          Conv2d-167         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-168         [-1, 1024, 14, 14]           2,048
            ReLU-169         [-1, 1024, 14, 14]               0
      Bottleneck-170         [-1, 1024, 14, 14]               0
          Conv2d-171          [-1, 256, 14, 14]         262,144
     BatchNorm2d-172          [-1, 256, 14, 14]             512
            ReLU-173          [-1, 256, 14, 14]               0
          Conv2d-174          [-1, 256, 14, 14]         589,824
     BatchNorm2d-175          [-1, 256, 14, 14]             512
            ReLU-176          [-1, 256, 14, 14]               0
          Conv2d-177         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-178         [-1, 1024, 14, 14]           2,048
            ReLU-179         [-1, 1024, 14, 14]               0
      Bottleneck-180         [-1, 1024, 14, 14]               0
          Conv2d-181          [-1, 256, 14, 14]         262,144
     BatchNorm2d-182          [-1, 256, 14, 14]             512
            ReLU-183          [-1, 256, 14, 14]               0
          Conv2d-184          [-1, 256, 14, 14]         589,824
     BatchNorm2d-185          [-1, 256, 14, 14]             512
            ReLU-186          [-1, 256, 14, 14]               0
          Conv2d-187         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-188         [-1, 1024, 14, 14]           2,048
            ReLU-189         [-1, 1024, 14, 14]               0
      Bottleneck-190         [-1, 1024, 14, 14]               0
          Conv2d-191          [-1, 256, 14, 14]         262,144
     BatchNorm2d-192          [-1, 256, 14, 14]             512
            ReLU-193          [-1, 256, 14, 14]               0
          Conv2d-194          [-1, 256, 14, 14]         589,824
     BatchNorm2d-195          [-1, 256, 14, 14]             512
            ReLU-196          [-1, 256, 14, 14]               0
          Conv2d-197         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-198         [-1, 1024, 14, 14]           2,048
            ReLU-199         [-1, 1024, 14, 14]               0
      Bottleneck-200         [-1, 1024, 14, 14]               0
          Conv2d-201          [-1, 256, 14, 14]         262,144
     BatchNorm2d-202          [-1, 256, 14, 14]             512
            ReLU-203          [-1, 256, 14, 14]               0
          Conv2d-204          [-1, 256, 14, 14]         589,824
     BatchNorm2d-205          [-1, 256, 14, 14]             512
            ReLU-206          [-1, 256, 14, 14]               0
          Conv2d-207         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-208         [-1, 1024, 14, 14]           2,048
            ReLU-209         [-1, 1024, 14, 14]               0
      Bottleneck-210         [-1, 1024, 14, 14]               0
          Conv2d-211          [-1, 256, 14, 14]         262,144
     BatchNorm2d-212          [-1, 256, 14, 14]             512
            ReLU-213          [-1, 256, 14, 14]               0
          Conv2d-214          [-1, 256, 14, 14]         589,824
     BatchNorm2d-215          [-1, 256, 14, 14]             512
            ReLU-216          [-1, 256, 14, 14]               0
          Conv2d-217         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-218         [-1, 1024, 14, 14]           2,048
            ReLU-219         [-1, 1024, 14, 14]               0
      Bottleneck-220         [-1, 1024, 14, 14]               0
          Conv2d-221          [-1, 256, 14, 14]         262,144
     BatchNorm2d-222          [-1, 256, 14, 14]             512
            ReLU-223          [-1, 256, 14, 14]               0
          Conv2d-224          [-1, 256, 14, 14]         589,824
     BatchNorm2d-225          [-1, 256, 14, 14]             512
            ReLU-226          [-1, 256, 14, 14]               0
          Conv2d-227         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-228         [-1, 1024, 14, 14]           2,048
            ReLU-229         [-1, 1024, 14, 14]               0
      Bottleneck-230         [-1, 1024, 14, 14]               0
          Conv2d-231          [-1, 256, 14, 14]         262,144
     BatchNorm2d-232          [-1, 256, 14, 14]             512
            ReLU-233          [-1, 256, 14, 14]               0
          Conv2d-234          [-1, 256, 14, 14]         589,824
     BatchNorm2d-235          [-1, 256, 14, 14]             512
            ReLU-236          [-1, 256, 14, 14]               0
          Conv2d-237         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-238         [-1, 1024, 14, 14]           2,048
            ReLU-239         [-1, 1024, 14, 14]               0
      Bottleneck-240         [-1, 1024, 14, 14]               0
          Conv2d-241          [-1, 256, 14, 14]         262,144
     BatchNorm2d-242          [-1, 256, 14, 14]             512
            ReLU-243          [-1, 256, 14, 14]               0
          Conv2d-244          [-1, 256, 14, 14]         589,824
     BatchNorm2d-245          [-1, 256, 14, 14]             512
            ReLU-246          [-1, 256, 14, 14]               0
          Conv2d-247         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-248         [-1, 1024, 14, 14]           2,048
            ReLU-249         [-1, 1024, 14, 14]               0
      Bottleneck-250         [-1, 1024, 14, 14]               0
          Conv2d-251          [-1, 256, 14, 14]         262,144
     BatchNorm2d-252          [-1, 256, 14, 14]             512
            ReLU-253          [-1, 256, 14, 14]               0
          Conv2d-254          [-1, 256, 14, 14]         589,824
     BatchNorm2d-255          [-1, 256, 14, 14]             512
            ReLU-256          [-1, 256, 14, 14]               0
          Conv2d-257         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-258         [-1, 1024, 14, 14]           2,048
            ReLU-259         [-1, 1024, 14, 14]               0
      Bottleneck-260         [-1, 1024, 14, 14]               0
          Conv2d-261          [-1, 256, 14, 14]         262,144
     BatchNorm2d-262          [-1, 256, 14, 14]             512
            ReLU-263          [-1, 256, 14, 14]               0
          Conv2d-264          [-1, 256, 14, 14]         589,824
     BatchNorm2d-265          [-1, 256, 14, 14]             512
            ReLU-266          [-1, 256, 14, 14]               0
          Conv2d-267         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-268         [-1, 1024, 14, 14]           2,048
            ReLU-269         [-1, 1024, 14, 14]               0
      Bottleneck-270         [-1, 1024, 14, 14]               0
          Conv2d-271          [-1, 256, 14, 14]         262,144
     BatchNorm2d-272          [-1, 256, 14, 14]             512
            ReLU-273          [-1, 256, 14, 14]               0
          Conv2d-274          [-1, 256, 14, 14]         589,824
     BatchNorm2d-275          [-1, 256, 14, 14]             512
            ReLU-276          [-1, 256, 14, 14]               0
          Conv2d-277         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-278         [-1, 1024, 14, 14]           2,048
            ReLU-279         [-1, 1024, 14, 14]               0
      Bottleneck-280         [-1, 1024, 14, 14]               0
          Conv2d-281          [-1, 256, 14, 14]         262,144
     BatchNorm2d-282          [-1, 256, 14, 14]             512
            ReLU-283          [-1, 256, 14, 14]               0
          Conv2d-284          [-1, 256, 14, 14]         589,824
     BatchNorm2d-285          [-1, 256, 14, 14]             512
            ReLU-286          [-1, 256, 14, 14]               0
          Conv2d-287         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-288         [-1, 1024, 14, 14]           2,048
            ReLU-289         [-1, 1024, 14, 14]               0
      Bottleneck-290         [-1, 1024, 14, 14]               0
          Conv2d-291          [-1, 256, 14, 14]         262,144
     BatchNorm2d-292          [-1, 256, 14, 14]             512
            ReLU-293          [-1, 256, 14, 14]               0
          Conv2d-294          [-1, 256, 14, 14]         589,824
     BatchNorm2d-295          [-1, 256, 14, 14]             512
            ReLU-296          [-1, 256, 14, 14]               0
          Conv2d-297         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-298         [-1, 1024, 14, 14]           2,048
            ReLU-299         [-1, 1024, 14, 14]               0
      Bottleneck-300         [-1, 1024, 14, 14]               0
          Conv2d-301          [-1, 256, 14, 14]         262,144
     BatchNorm2d-302          [-1, 256, 14, 14]             512
            ReLU-303          [-1, 256, 14, 14]               0
          Conv2d-304          [-1, 256, 14, 14]         589,824
     BatchNorm2d-305          [-1, 256, 14, 14]             512
            ReLU-306          [-1, 256, 14, 14]               0
          Conv2d-307         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-308         [-1, 1024, 14, 14]           2,048
            ReLU-309         [-1, 1024, 14, 14]               0
      Bottleneck-310         [-1, 1024, 14, 14]               0
          Conv2d-311          [-1, 512, 14, 14]         524,288
     BatchNorm2d-312          [-1, 512, 14, 14]           1,024
            ReLU-313          [-1, 512, 14, 14]               0
          Conv2d-314            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-315            [-1, 512, 7, 7]           1,024
            ReLU-316            [-1, 512, 7, 7]               0
          Conv2d-317           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-318           [-1, 2048, 7, 7]           4,096
          Conv2d-319           [-1, 2048, 7, 7]       2,097,152
     BatchNorm2d-320           [-1, 2048, 7, 7]           4,096
            ReLU-321           [-1, 2048, 7, 7]               0
      Bottleneck-322           [-1, 2048, 7, 7]               0
          Conv2d-323            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-324            [-1, 512, 7, 7]           1,024
            ReLU-325            [-1, 512, 7, 7]               0
          Conv2d-326            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-327            [-1, 512, 7, 7]           1,024
            ReLU-328            [-1, 512, 7, 7]               0
          Conv2d-329           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-330           [-1, 2048, 7, 7]           4,096
            ReLU-331           [-1, 2048, 7, 7]               0
      Bottleneck-332           [-1, 2048, 7, 7]               0
          Conv2d-333            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-334            [-1, 512, 7, 7]           1,024
            ReLU-335            [-1, 512, 7, 7]               0
          Conv2d-336            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-337            [-1, 512, 7, 7]           1,024
            ReLU-338            [-1, 512, 7, 7]               0
          Conv2d-339           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-340           [-1, 2048, 7, 7]           4,096
            ReLU-341           [-1, 2048, 7, 7]               0
      Bottleneck-342           [-1, 2048, 7, 7]               0
AdaptiveAvgPool2d-343           [-1, 2048, 1, 1]               0
================================================================
Total params: 42,500,160
Trainable params: 42,500,160
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 429.72
Params size (MB): 162.13
Estimated Total Size (MB): 592.42
----------------------------------------------------------------

```

# 4.利用 random forest 分類器中的 feature_importances_ 功能對所有特徵進行重要度排序。
> 接著，只挑選排序後重要度前幾名的特徵進行訓練分類器(logistic regression 或 random forest 都可)觀察不同的特徵數(10個/100個/500個/1000個/全部)對於分類模型在測試集分類上效能的影響。
```
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]


def train_and_evaluate_classifier(features, labels, test_features, test_labels, n_features, classifier_type='rf'):
    selected_indices = indices[:n_features]
    selected_train_features = features[:, selected_indices]
    selected_test_features = test_features[:, selected_indices]
    
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == 'lr':
        classifier = LogisticRegression(max_iter=1000)
    
    classifier.fit(selected_train_features, labels)
    predictions = classifier.predict(selected_test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


feature_counts = [10, 100, 500, 1000, train_features.shape[1]]
results_rf = []
results_lr = []

for count in feature_counts:
    accuracy_rf = train_and_evaluate_classifier(train_features, train_labels, test_features, test_labels, count, 'rf')
    accuracy_lr = train_and_evaluate_classifier(train_features, train_labels, test_features, test_labels, count, 'lr')
    results_rf.append(accuracy_rf)
    results_lr.append(accuracy_lr)
    print(f'Feature count: {count}, RF Accuracy: {accuracy_rf:.4f}, LR Accuracy: {accuracy_lr:.4f}')


plt.figure(figsize=(10, 5))
plt.plot(feature_counts, results_rf, label='Random Forest')
plt.plot(feature_counts, results_lr, label='Logistic Regression')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Features on Classification Performance')
plt.legend()
plt.show()
```


| Feature count | RF Accuracy | LR Accuracy |
| -------- | --------  |-------- |
| 10       | 0.9808    | 0.9794  |
| 100      | 0.9864    | 0.9874  |
| 500      | 0.9860    | 0.9868  |
| 1000     | 0.9862    | 0.9864  |
| 2048     | 0.9854    | 0.9872  |



![image](https://hackmd.io/_uploads/H1SgyPDuA.png)

## 結果分析
**特徵數對準確率的影響:**
> 隨著特徵數量的增加，隨機森林和邏輯迴歸的準確率均有小幅度的提高。
> 這表明更多的特徵可以帶來更好的分類性能。
> 在特徵數量為 100 時，兩種分類器的準確率達到了最高，之後的增加對準確率的提升效果較小。
> 在全部特徵（2048）時，邏輯迴歸的準確率略高於隨機森林，而隨機森林的準確率略低於前幾組特徵數量的結果。

**分類器性能比較:**
> 邏輯迴歸和隨機森林的分類表現相似，但在特徵數量較少時，隨機森林的表現稍好。
> 邏輯迴歸在特徵數量較多時略微優於隨機森林，這可能與邏輯迴歸的特徵處理能力有關。

**結論**
> 特徵選擇: 適當的特徵選擇可以提高分類性能，特徵數量的增加對分類準確率有正面影響，但增加到一定數量後效果趨於平穩。
> 分類器選擇: 對於本實驗的數據，邏輯迴歸在特徵較多時表現略優於隨機森林。

# 5.分別對於貓類與狗類其重要特徵度排名前5名的特徵在訓練集中找出響應最大(activation分數最高)的前10張案例影像觀察該特徵對於貓狗的那些部位有響應，以及解釋是否該特徵具解釋性。

```
def extract_feature_maps(loader):
    feature_maps = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to('cuda')
            outputs = model_fc(inputs)
            feature_maps.append(outputs.squeeze().cpu().numpy())
            labels.append(targets.numpy()) 
    return np.array(feature_maps), np.array(labels)

# 提取特徵映射(第一個捲積層)
feature_maps, labels = extract_feature_maps(train_loader)
importances = rf_classifier.feature_importances_


feature_maps, labels = extract_feature_maps(train_loader)

n_top_features = 5
top5_feature_indices = np.argsort(importances)[::-1][:n_top_features]
print("Top 5 important feature indices for cat class:", top5_feature_indices)
```
Top 5 important feature indices for cat class: [1031 1377  236  127  723]

## 1031
![image](https://hackmd.io/_uploads/r1MZpOwdA.png)
![image](https://hackmd.io/_uploads/H1EUTOvu0.png)

## 1377
![image](https://hackmd.io/_uploads/HJzf6dvuC.png)
![image](https://hackmd.io/_uploads/ByVDT_P_R.png)

## 236
![image](https://hackmd.io/_uploads/rJZQ6uw_0.png)
![image](https://hackmd.io/_uploads/ryM_T_wO0.png)

## 127
![image](https://hackmd.io/_uploads/HkxNaOwOR.png)
![image](https://hackmd.io/_uploads/rJodp_wdC.png)

## 723
![image](https://hackmd.io/_uploads/BJ1ra_vO0.png)
![image](https://hackmd.io/_uploads/BJPtauv_A.png)

## 討論
* 特徵重疊：在前五名特徵中，部分特徵輸出的圖片有重複，顯示這些特徵在模型中的獨特性不夠。這可能是因為模型在訓練過程中發現某些特徵（如頭部）在區分貓狗時特別重要，因此這些特徵在多個特徵中都有反應。
* 特徵分析：頭部的部分，主要分類依據可能是鼻子，耳朵和貓咪的鬍鬚。而在分類狗類時，也經常出現狗狗相較於貓咪較長的四肢(如feature236和feature127)，這也可能是一個主要分類特徵。

![image](https://hackmd.io/_uploads/B1s2kKPO0.png)
![image](https://hackmd.io/_uploads/rklpaZYDdR.png)
![image](https://hackmd.io/_uploads/BJHF5FvOR.png)
