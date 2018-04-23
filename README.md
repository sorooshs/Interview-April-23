# Instruction

Write about your project in this area.

# How to setup

### Prerequisites
- [virtualenv](https://virtualenv.pypa.io/en/latest/)

### Initialize the project
Create and activate a virtualenv:

```bash
virtualenv venv
source venv/bin/activate
```

#### Install dependencies:

```bash
pip install -r requirements.txt
```

#### Part 1: Modeling

Modeling part is done in the jupyter notebook. To run the notebook, please run the following command.

To run the part one please run:
```bash
jupyter notebook Modeling.ipynb
```

#### Part 2: Application

The code is written in python. The output is already saved in output.tsv.
If you run the code, first all models will be created and saved in models folder. The next time you run the code the saved models will be loaded.

```bash
./run app.main Data_Science/data_to_predict.json
```
The output file will be saved in output.tsv
