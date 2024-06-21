## 基于 LSTM 三分类的文本情感分析

### 参考

项目

- `https://github.com/Edward1Chou/SentimentAnalysis`

参考博客：

- `https://blog.csdn.net/m0_46202073/article/details/109303450`
- `https://blog.csdn.net/weixin_40959890/article/details/127398128`
- `https://blog.csdn.net/qq_44951759/article/details/126001215`

数据集：

- `https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated`

可视化库参考：

- `https://zhuanlan.zhihu.com/p/644644243`
- `https://pyecharts.org/#/zh-cn/intro`

### 数据集划分

使用`data/data-proportion.ipynb`文件，可以对 Netflix 原始数据集进行划分，划分为`positive.csv`、`negative.csv`和`neutral.csv`三个文件。

### 安装&使用

1. 安装依赖包：
   `pip install -r requirements.txt`
2. 运行文件：
   - `python lstm/lstm_train.py`
   - `python lstm/lstm_test.py`
     或者运行脚本：
   - `sh scripts/run_train.py` （一次运行多个模型的训练，可以参考该文件进行不同模型的训练）
3. 可视化
   可以根据自己的训练数据修改`draw_img.ipynb`文件中的数据，运行即可实现可视化
4. 数据分析
   可以运行 `data-analysis.ipynb` 文件来获得数据分析的可视化结果

### `scripts/run_train.py`脚本说明：

支持的参数：

- `-d`，`--data_path`指定数据集所在的目录
- `-m`，`--model_name`指定使用什么模型进行训练
- `-s`，`--save_path`指定模型配置和权重文件的保存目录
- `-a`，`--attention`指定是否使用 Attention 层或 SelfAttention 层

目录说明：

- 文件的 log 会输出在项目根目录的 logs 目录下。
- 默认情况下，数据集目录为根目录下的 data 目录，可以通过`-d`或`--data_path`参数进行自定义。
- 模型保存目录为根目录下的 model 目录，可以通过`-m`或`--model_name`参数进行自定义。
- Word2Vec 模型的权重保存在根目录下的 word2vec 目录下，没有提供参数进行修改，如需修改，只能在代码中进行修改。
