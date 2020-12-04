# ASL Alphabet Recognition

Classifier for images of letters in American Sign Language. Written to
help me learn CNNs and PyTorch.

## Running

Dependencies:
- python
- make
- rsync
- p7zip
- [kaggle-api](https://github.com/Kaggle/kaggle-api)

Make sure the credentials for `kaggle-api` are configured.

Install python dependencies:

```
$ pip -r requirements.txt
```

Download the data:

```
$ make
```

Then train the model:

```
$ ./model.py train
```

This will save a `*.pt` model file every five epochs in `build/<training
start time>/`. For example, the directory might be called
`2020-12-03T12:40:13`.

Run the tests on a chosen model file:

```
$ ./model.py test build/<start time>/<file name>.pt
```

The server will load the file `model.pt` in the project root. So just
copy (or symlink) the chosen model.

```
$ cp build/<start time>/<file name>.pt ./model.pt
```

Run the web server:

```
$ flask run
```

Finally, open [http://localhost:5000](http://localhost:5000) in your
browser.
