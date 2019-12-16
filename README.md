# ISIPS2019_mizokami
ISIPS2019で溝上くんが発表した内容です。

## 必要パッケージ
* sporco
* opencv
* torchvision

## コードレビューの前にやっていただきたいこと
1. 仮想環境を作る
現在それぞれのパソコンに入っているpython の環境を極力維持しておきたいので、今回は仮想環境を作ってやります。
コマンドラインで以下を実行します。
```
conda create -n review python=3.7 opencv
```
すると
```
Proceed ([y]/n)?
```
と出てくるはずなので、そのままEnter

2. 仮想環境をアクティベート
作った仮想環境を有効化してやるために
```
conda activate review
```
を実行します

3. パッケージのインストール
必要なパッケージをインストールします。
```
pip install sporco
conda install pytorch torchvision cpuonly -c pytorch
```
4. コード実行
レポジトリに移動し、以下を実行
```
jupyter lab &
```

5. 仮想環境をディアクティベート
今いる仮想環境「review」から抜け出します
```
conda deactivate
```
6. 仮想環境を破壊
環境「review」を破壊します
```
conda env remove -name review
```