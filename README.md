# Voice Identification

This project is to better understand how to identify a user by their speech and further to understand what they are saying. It will be tackled in two parts:

   1. Speaker Identification
   
   2. Speech to Text Recognition

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The following dependencies are required for this project to work. The commands given below are how to install each one.

#### Pip Installation:
    $ sudo apt-get install python-pip python-dev --yes
    $ pip install --upgrade pip --user
    $ pip install --upgrade setuptools --user

#### SciPy Stack Installation:
    $ sudo apt-get install python-numpy python-scipy python-skimage python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose --yes

#### TensorFlow Installation (CPU only):
    $ pip install tensorflow --user

#### Seaborn, Pandas Installation:
    $ pip install seaborn pandas --user

#### Jupyter Notebook Installation:
    $ pip install jupyter --user

#### TFLearn Installation:
    $ pip install tflearn --user

#### Librosa Installation:
    $ pip install librosa --user

#### Pydub Installation:
    $ pip install pydub --user

#### pyAudioAnalysis Installation:
    $ pip install numpy matplotlib scipy sklearn hmmlearn simplejson eyed3 pydub --user
    $ sudo apt-get install python-tk --yes
    $ git clone https://github.com/tyiannak/pyAudioAnalysis.git

#### Python Audio Tools Installation:
    $ wget https://downloads.sourceforge.net/project/audiotools/audiotools/2.16/audiotools.2.16.tar.gz?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Faudiotools%2Ffiles%2F&ts=1490043434&use_mirror=dronedata
    $ mv audiotools.2.16.tar.gz\?r\=https%3A%2F%2Fsourceforge.net%2Fprojects%2Faudiotools%2Ffiles%2F audiotools.2.16.tar.gz
    $ tar -xzvf audiotools.2.16.tar.gz
    $ sudo apt-get install libcdio-dev libcdio-paranoia-dev
    $ cd audiotools-2.16/
    $ sudo make install
    
    Test Installation with:
    
    $ audiotools-config

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* Item 1
* Item 2
* Item 3

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* Authors

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
