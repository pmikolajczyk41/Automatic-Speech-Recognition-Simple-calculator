# Automatic Speech Recognition: Simple calculator

### How to start

```
git clone https://github.com/pmikolajczyk41/Automatic-Speech-Recognition-Simple-calculator.git
cd Automatic-Speech-Recognition-Simple-calculator/

python3 -m venv ./venv
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH=`pwd`
```

### Data format

There are 15 base models: one for silence, one for each digit and one for each operator:

- "+" (plus)
- "-" (minus)
- "*" (times)
- "/" (by)

To train them you need to provide .wav files to `resources/train/` in a format `<name>-<id>.wav`
where `<name>` corresponds to a symbol and `<id>` is a unique number for distinguishing between different recordings.

### How to train

The `main` section within `calculator/atom_training.py` script is responsible for training atom models. After providing
training data to the `resources/train/` directory you are ready.

```
python data/convert.py
cd calculator/
python atom_training.py
```

Notice that by default Viterbi training preceded by a uniform segmentation is applied. To retrain you models just
comment/uncomment appropriate lines in `calculator/atom_training`.

### How to test

```
cd calculator/
python testing.py
```

### How to run app

```
cd calculator/
python app.py
```

You may get some errors connected with audio setup within your
system. [Here](https://stackoverflow.com/questions/49333582/portaudio-library-not-found-by-sounddevice?fbclid=IwAR1dy3bpxTe9R0i1c0RGTjZFUGeQD1qbXCh7xtLur4gSpgntbHKCPoaSTT0)
is a sample common problem raised with some solutions.