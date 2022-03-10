# Surakav: Generating Realistic Traces for a Strong Website Fingerprinting Defense
<div >
<img src="https://img.shields.io/badge/Surakav-1.0.0-brightgreen.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAA3NCSVQICAjb4U/gAAAACXBIWXMAAASdAAAEnQF8NGuhAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAp1QTFRF////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+jJ5vAAAAN50Uk5TAAECAwQFBgcICQoLDA0ODxESExQVFhcYGRscHR4fICEiIyQlJicpKissLS4vMDEzNDY3ODk6Ozw9QEFCQ0RFRkdJSkxNTk9RUlNUVVZXWFlaW1xeX2BhYmNkZmdpamtsbW5xcnN0dXd4fH1+f4CBgoOFhoeIiYqLjI2OkJGSk5SVlpeYmZucnZ6foKGjpKWmp6irrK2ur7CxsrO0tri5u7y9v8DBwsPExcbHyMnKy8zNzs/R0tPV1tjZ2tvc3d7f4OHi5OXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+IPqEcwAABMhJREFUGBntwflfk3UAB/AP22DAJqIRamreKeaVQmZW5lVooqlFZhqmeabikWZ5onllhoJKUuaRR6GQiHhgGIcaiSg4HePY529pL+CFe8Y29mzP97tffL+B58SL0SGYjLtsvRBEplO8jCCKziGXIXi6XCVtPRE0vYtJboJHYZEQKu4uyf86wp1eM9cevVHPnRBoVBUdPkMbPT7aX8JmRRDnfQsdCvRQ6DprTzGf2QhRwrawyVg46bfmGhXKzRCkdx6bZKJVp3k5dDUFgkx9xCaVsWhmeO+IjW1kQQzjdrZIQrPEIrpR8xKE6JvPFhlokvAn3UqBEEk1bFERA4f+R+lekR4ChO9kq0QA5m319CAZAvS7wlbpAEbeoif/GKC9pBq2utcJ+pX19CgFmjOm0clk9MmhZ7ZO0FrffDo5gE8e04tj0Nq0ajop73uEXk2BtkJSqbD5Lr16aISmjOlUaLDTu73Q1AsXqE79QGhpQDFVWgEtvVlFdeyLoKXkOqrz8zBoKGQ91ckeDtUiukUb4J7pGNVZDZUMkzIfkbQXbp+AtrpfoTrZUGnYTbY6NwIuRt6jSolQ5wsbndhXQ2GGlSrVhEOVbXSxx4BWIalU7QeospRtHDegRWQG1RsLNWba2dZWNBtSQPXKQqDC1Dq6MwcOhq/q6Icl8JVhaMo1umd7C+bp+fTHevjAHDdpQdolK73Ie0p/lM6FNxGvjJ+/MSPvPgUpmB0K97okzFp54PwdO0W6/y48OGQh2WipLC++fuXi2ZO/HEnfv2vLxk079v509EQZtdI4CJ506B4bFQYPYqbcoSZy4a/YM9TCGvhN/w01kIAAbGbAKvQIgC6LgfoAAYnMZWAOIkCxJQxEZgcE6rUG+q06BRpIpZ+qVkVDC6GX6Y+HX5qhkYFWqpffG9pZRLXs3xmhId15qlM2Ftoa1kg1fuwIre2j76qmQXtdH9NXp7pBhOX0Te3CEAgRUUpfFMZBlBn0QdmLEOcC22UdAYEG17M9uyHUJrZnDYQyl7Idf0Cs+Dq2YyjEWsB27IVgh+idtTPEMp2jd8sgmOl3elWqh2CmM/QqEaJFnqY3ZyBcxEl6MxjChf9GL76HeOFZ9OxJNCRYaKNHiyHD8Fv05LYOMnQ4SE8mQI6PLXTvMCQZkEu3rFGQZeIlujMH8ow7z7bOQqYxp+nK/jKkGn+TLlZArtDF1VQogmyvN1JhFGTbQIU0yGYspLMHYZBtHBUSIZuhks6yIN0+OquLgWwTqbAUshktdHZbB9kuUGECZNtKhWzIlkyFxp6QbAiV1kKy0FoqVIRBsr+oNB2S7abSWUg2ny4GQa54utgOuUyNVKo2Qa4bdDEPcqXTxXXINZeu3oFUPegqG3Jdo4vGPpDqWz5jL6HDZkj1NpvU5dBhSSXJajNkMlrIJySTG0gefsNGMgVSZZPrSH6aRtISOZtkIaSazceR98m0zg9ITsM6kq9CJt2JVdhBXsTnJDOhO05+DdniSWuYvoB8akbUdZaEQLZiMh5jSH4I9HvI0ZAtlVwMZJC/Akj4dzlk608eA3rVsiEWQLcBkC6X5QA2kAsRHEm8CyCqgvkIkm1X4TCXjEOQ6OCgL+AGBNXo2jwE1+S/EWSd8ZxX/wP1mbRZY1/8zgAAAABJRU5ErkJggg==">
</div>

## What?
This is the repository for training the trace generator. We mainly introduce the usage of the code as below. 
This is only for research purpose, so use carefully.

## How to use

### Feature extraction
First modify `conf.ini` (only `MONITORED_SITE_NUM` and `MONITORED_INST_NUM` matters).
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
Remember to convert back to second from the log values. 