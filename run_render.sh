source dmc-venv/bin/activate

quarto render 01-eda.ipynb
quarto render 02-class.ipynb
quarto render 03-cluster.ipynb

deactivate