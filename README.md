# Titanic: Machine Learning from Disaster

## What's this?

We are simply playing around with Titanic dataset. We plan to develop multiple models to predict whether a passanger would survive or not. For more information, [visit Kaggle](https://www.kaggle.com/c/titanic).

## Set up the enviroment

Always **remember to use a virtual enviroment**. You can install *venv* simply typing:

    >> sudo pip install virtualenv

After you navigated to the correct folder, type:

    >> virtualenv env
    >> source env/bin/activate

And you have done. The last line must be executed every time one wants to use the virtual enviroment. The *requirements.txt* file contains a list of all the libraries that have been used to write the code. You can install them using pip once the virtual enviroment has been activated:

    >> pip install -r requirements.txt

To create a new requirements list just use the *pip freeze* comand:

    >> pip freeze > requirements.txt

For more infos visit [the relevant python-guide](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
