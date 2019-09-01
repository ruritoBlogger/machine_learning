import chainer
from chainer import training, iterators, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
 
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import numpy as np

def data_read( ft_name, fa_name ):
    teachers = np.array([] )
    answers =  np.array([] )

    ft = open( ft_name, mode = "r" )
    fa = open( fa_name, mode = "r" )

    ft_data_string = ft.readlines()
    fa_data_string = fa.readlines()

    for i in range( 0, len( ft_data_string ) ):
        fa_data = fa_data_string[i].replace( "\n", "" )
        answers = np.append( answers, float( fa_data ) )
        
        ft_data = ft_data_string[i].replace( "\n", "" )
        ft_data = ft_data.split( " " )

        for r in range( 0, len( ft_data ) ):
            teachers = np.append( teachers ,float( ft_data[r] ) )

    ft.close()
    fa.close()

    teachers = teachers.astype( np.float32 )
    answers = answers.astype( np.float32 )

    teachers = np.reshape( teachers, ( int( len( teachers ) / 3 ), 3 ) )
    answers = np.reshape( answers, ( len( answers ) , 1 ) )

    return teachers, answers

#ニューラルネットワークの構築。
class MyChain(Chain):
    
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(MyChain, self).__init__(
        l1=L.Linear(n_input, n_hidden1),
        l2=L.Linear(n_hidden1, n_hidden2),
        l3=L.Linear(n_hidden2, n_hidden2),
        l4=L.Linear(n_hidden2, n_output),
    )
 
    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        o = self.l4(h3)
        return o

teachers, answers = data_read( "teacher_data1.txt", "answer_data1.txt" )

#学習、検証データの割合(単位：割)
trainSt = 0 #学習用データの開始位置 0割目から〜
trainPro = 8 #学習用データの終了位置　8割目まで
testPro = 10 #検証用データの終了位置 8割目から10割目まで

#総データの長さ
N = len(teachers)

#teacherの次元調整
#teachers = teachers.flatten()
#answers = answers.flatten()

#teachers = chainer.Variable(teachers)
#answers = chainer.Variable(answers)

#学習用データと検証用データに分ける
#x_train= teachers[:N*trainPro//10]
#y_train = teachers[N*trainPro//10:]

#x_ans = answers[:N*trainPro//10]
#y_ans = answers[N*trainPro//10:]

# ログの保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}

#学習、検証データの割合(単位：割)
trainSt = 0 #学習用データの開始位置 0割目から〜
trainPro = 8 #学習用データの終了位置　8割目まで
testPro = 10 #検証用データの終了位置 8割目から10割目まで

#学習用データと検証用データに分ける
train_teach= teachers[:N*trainPro//10]
test_teach = teachers[N*trainPro//10:]

train_ans= answers[:N*trainPro//10]
test_ans = answers[N*trainPro//10:]

#モデルを使う準備。オブジェクトを生成
n_input = 3
n_hidden1 = 3
n_hidden2 = 2
n_output = 1
model = MyChain(n_input, n_hidden1, n_hidden2, n_output)

#最適化を行う
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

n_epoch = 30
n_batchsize = 16

# 各バッチ毎の目的関数の出力と分類精度の保存用
loss_list = []
accuracy_list = []

print(teachers.shape)
print(answers.shape)

from chainer.dataset import concat_examples
from chainer import Variable
for epoch in range(n_epoch):
        
    ite  = chainer.iterators.SerialIterator(train_teach, batch_size=n_batchsize, 
                                            repeat=False, shuffle=True)
    for i in range(0, train_teach.shape[0],n_batchsize):
         
        # 予測値を出力
        #train_batch = ite.next()
        train_batch = train_teach[i:i+n_batchsize-1]
        train_ans_batch = train_ans[i:i+n_batchsize-1]
        #train_batch = concat_examples(train_batch)
        #train_ans_batch = concat_examples(train_ans_batch)
        y_train_batch = model(train_batch)
        print(y_train_batch)
        print(train_ans_batch)
        
        # 目的関数を適用し、分類精度を計算
        loss_train_batch = F.mean_squared_error(y_train_batch, train_ans_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, train_ans_batch.flatten().astype(np.int32))

        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)
    
        # 勾配のリセットと勾配の計算
        model.cleargrads()
        loss_train_batch.backward()

        # パラメータの更新
        optimizer.update() 
        
    # 訓練データに対する目的関数の出力と分類精度を集計
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # 1エポック終えたら、検証データで評価
    # 検証データで予測値を出力
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        test_ans_y = model(test_teach)

    # 目的関数を適用し、分類精度を計算
    loss_val = F.mean_squared_error(test_ans_y, test_ans)
    accuracy_val = F.accuracy(test_ans_y, test_ans.flatten().astype(np.int32))

    # 結果の表示
    print('epoch: {},loss (train): {:.4f}, loss (valid): {:.4f}'.format(
        epoch, loss_train, loss_val.array))

    # ログを保存
    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)
