# CNN网络发展介绍
https://zhuanlan.zhihu.com/p/76275427
![](https://pic2.zhimg.com/v2-86b55993300634d8c5e3256c1b784480_1440w.jpg?source=172ae18b)

# resnet网络特点
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
ResNet指出，在许多的数据库上都显示出一个普遍的现象：增加网络深度到一定程度时，更深的网络意味着更高的训练误差。
 
 
误差升高的原因是网络越深，梯度消失的现象就越明显，所以在后向传播的时候，无法有效的把梯度更新到前面的网络层，靠前的网络层参数无法更新，导致训练和测试效果变差。所以ResNet面临的问题是怎样在增加网络深度的情况下有可以有效解决梯度消失的问题。

ResNet中解决深层网络梯度消失的问题的核心结构是**残差网络**：
![残差单元示意图](https://pic2.zhimg.com/80/90e58f36fc1b0ae42443b69176cc2a75_720w.png)
残差网络增加了一个identity mapping（恒等映射），把当前输出直接传输给下一层网络（全部是1:1传输，不增加额外的参数），相当于走了一个捷径，跳过了本层运算，这个直接连接命名为“skip connection”，同时在后向传播过程中，也是将下一层网络的梯度直接传递给上一层网络，这样就解决了深层网络的梯度消失问题。

可以注意到残差网络有这样几个特点：1. 网络较瘦，控制了参数数量；2. 存在明显层级，特征图个数逐层递进，保证输出特征表达能力；3. 使用了较少的池化层，大量使用下采样，提高传播效率；4. 没有使用Dropout，利用BN和全局平均池化进行正则化，加快了训练速度；5. 层数较高时减少了3x3卷积个数，并用1x1卷积控制了3x3卷积的输入输出特征图数量，称这种结构为“瓶颈”(bottleneck)。
# imagenet图像预处理
预处理将得到一个BHWC结构的张量，张量的shape为(1,224,224,3)，其中第0个维度表示batch_size,第1个维度表示图像height,第2个维度表示图像的weight,第3个维度表示图像的channel,这里是一张3通道(RGB)的图像。
```python
def preprocess(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch
```
# 基于resnet50模型进行推理
输出为shape为(1,1000)的张量，代表每个类别对应的预测概率
```python
def running(input):
    net = models.resnet50().to(device)
    net.load_state_dict(torch.load(model_file, map_location=device))
    net = net.to(device)
    net.eval()
    output = net(input)
    output = torch.nn.functional.softmax(output[0], dim=0).tolist()
    return output
```
# 预测结果后处理
后处理是筛选出概率值最高的类别，并将该类别对应的label与概率值作为返回值
```python
def postprocess(output, label_file="./labels.txt"):
    labels = __load_labels(label_file)
    top_k = np.array(output).argsort()[-1:][::-1]
    result = {}
    for i in top_k:
        result[labels[i]] = output[i]
    return result
```

# 基于flask封装api
基于flask封装图像分类的api，将用户上传的图像作为输入，预测概率最高的类别和概率值作为输出。
```python
@app.route('/', methods=['POST'])
def serving():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    input = preprocess(filename).to(device)
    output = running(input)
    result = postprocess(output)
    return result
```
# 接口请求测试
```bash
curl -XPOST http://127.0.0.1:5000 -F "file=@./dog.jpg"
```
#### 输出结果
```
{
  "Great Pyrenees": 0.8732960820198059
}
```