# Surakav: Generating Realistic Traces for a Strong Website Fingerprinting Defense
<div >
<img src="https://img.shields.io/badge/Surakav-1.0.0-brightgreen.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAlAAAAJQBeb8N7wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAD+SURBVDiNndO/SgNBEMfxz4ZELEyhNoJ2lqksI1Y2PoEKNoKN4AMIvoC9IJY+gg+gmMZeULDyGRQEFfwTx8K7cIE7ucvAwDI7v+/Ozu6kiFDXUkpdLETE4ygYEbUcy7jANPZwiUFd8QaecYJ7ROZbZcnz6GTrhCMMC6LcT9EqChdxgC/cYh2DEmFkVbQiQjul1MMSZvCND6zguqKXD+hHxE9pE7FTcWrgBr2x/IqmnZWIj/Oy6wA6JfdfK8395+lmcVUAnDcCFEC7eMcruo0BGeQwq2J/UkA/A9xNCpjCE96wWtxrV3yWMYuIz5TSJrb9DdPIUsNxnsMwIl7y2C/QuaOo70VRKQAAAABJRU5ErkJggg==">
</div>

## What?
This is the repository for training the trace generator. We mainly introduce the usage of the code as below. 
This is only for research purpose, so use carefully.

## How to use

### Feature extraction
The raw traces must be in the **cell sequence** format or **packet sequence** format where each file has two columns:
the first column lists the timestamps and the second column lists the direction (+-1) or the directional bytes (+-bytes). 
We extract features in the **burst sequence** format, that is, a sequence of +-N representing the size of a burst (N is the number of cells). 
The sign shows the packet direction. 
Usage:
```
python3 src/extract.py --dir [your_dir] --length [your_preferred_length] --format [file_suffix]
```
You will get two .npz files: one saves the burst sequence features together with the labels; another saves the time features used for modeling o2o time gaps.


### Training an DF observer
Our GAN involves an observer which is a pre-trained DF model. Using the features generated above, we can train a DF model based on the burst sequences by
```angular2html
python3 src/train_df.py [your_feature_dir]
```
The model will be saved to `./f_model/xxx.ckpt`.


### Training GAN
Here is an example of training the GAN:
```angular2html
python3 src/mlp_df_wgan_train.py -d ./dump/my_dataset/feature/raw_feature_0-100x0-1000_clip.npz --f_model ./f_model/df_raw_feature_0-100x0-1000.ckpt 
```
`--f_model` is the pre-trained observer and `-d` provide the dir of training set. 
Please also take a look at other arguments in the code. 


### Time Gap Modelling
We need to model the o2o time gap for the defense based on the dataset. 
Remember we have generated a `time_feature_xxx.npz` file during the extraction phase. 
We now use the time_feature to generate the o2o distribution with KDE method:
```angular2html
python3 src/ipt_sampler.py --tdir [your_time_feature_path] -n 1000000
```
You will get a `xxx.ipt` file. 
The first row shows the computed kde kernel_std and the rest 1,000,000 values are the time gaps sampled from the original dataset (seconds, in log scale).
This file is enough to model the hidden distribution of o2o time gap. 
To sample a time gap from the distribution is equivalent to compute
```
t + normal(0,1) * kernel_std, 
```
where *t* is randomly sampled from the 1,000,000 time gaps. 
Remember to convert back to seconds from the log values. 