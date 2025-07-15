# Driver-Profile-Classification
Driver Profile Classification thesis repository. This project aims to develop a system that takes advantage of the use of machine learning techniques to
establish a driving profile based on trips data obtained in a non-intrusive way.

Pre-Requisites
--------------

Working Python 3 installation (Python version = 3.9.2), with packages presented in the file requirements.txt


Running the Scripts
-------------------

1. Create virtual environment.
   * On Windows, use "python -m venv .venv".
2. Activate virtual environment.
   * On Windows, use "cd .venv/Scripts".
   * On Windows, use "activate" to activate the virtual environment (make sure you are in the ".venv/Scripts" directory).
   * On Windows, navigate back to tests directory "cd ../..".
3. Install requirements.
   * On Windows, use "pip install -r requirements.txt".
4. Try out the diferent sripts.
   * Use the following command "python name_of_the_script.py".
5. Press enter to run the script.


Notes
-----

1. It is necessary to have an API token in order to run the script preprocess_1/a_idreams_pull_trips.py.
2. Because the events are private, they have not been uploaded to the repository so the script preprocess/b_construct_dataset.py does not run.
3. Only one dataset is available to make sure the repository is not overloaded with datasets.
