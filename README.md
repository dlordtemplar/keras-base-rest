### Requirements

- flask (1.0.2)
- hunspell (0.5.5, Requires python-dev and libhunspell-dev. Install with 'sudo apt-get install libhunspell-dev')
- numpy (1.16.2)
- editdistance (0.5.2)
- gensim (3.7.1)
- keras (2.2.4)
- nltk (3.4)
- pandas (0.24.1)
- scikit-learn (0.21.3)
- spacy (2.0.18, You will need to get the English model: "python -m spacy download en")
- tensorflow-gpu (NOT tensorflow. You need the GPU. [Install 1.12.0 for CUDA 9.0, and 1.14.0 for CUDA 10.0](https://www.tensorflow.org/install/source#tested_build_configurations))

NOTE: If you do not have sudo access, you cannot install the spellchecking library and will not be able to generate your own model. In that case, switch to the "prod" branch for a pre-generated model.

### Setup
1. Install CUDA. [Install 1.12.0 for CUDA 9.0, and 1.14.0 for CUDA 10.0](https://www.tensorflow.org/install/source#tested_build_configurations). I recommend [this guide](https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_10) if your graphics card is compatible with CUDA 10.0. The relevant commands are posted below:
   ~~~
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo apt-get update
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt-get update
    
    # Install NVIDIA driver
    # Get your latest driver, just has to be at least this version.
    sudo apt-get install --no-install-recommends nvidia-driver-418
    # Reboot. Check that GPUs are visible using the command: nvidia-smi
    
    # Install development and runtime libraries (~4GB)
    sudo apt-get install --no-install-recommends \
        cuda-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0  \
        libcudnn7-dev=7.6.2.24-1+cuda10.0
   ~~~
2. Git clone the repository.
    ~~~
    git clone https://github.com/dlordtemplar/keras-base-rest.git
    ~~~
3. If you have sudo access, you can continue to preprocessing and training. Otherwise, switch your branch to "prod" and continue to step 11.
4. In the [resources/embeddings](resources/embeddings) folder, you need to extract the word vectors from here:
   https://github.com/mmihaltz/word2vec-GoogleNews-vectors
   ~~~
   Final path to this file: resources/embeddings/GoogleNews-vectors-negative300.bin
   ~~~
5. Install the dependencies for hunspell (spellchecking library):
   ~~~
   sudo apt-get install python-dev libhunspell-dev
   ~~~
6. Install requirements and update pip to latest version (if necessary):
   ~~~
   pip install -r requirements.txt
   ~~~
7. Install spacy english model:
   ~~~
   python -m spacy download en
   ~~~
8. Install nltk stopwords:
   ~~~
   import nltk
   nltk.download('stopwords')
   ~~~
9. Run preprocess.py to clean and create text for training the model.
10. Run [train.py](train.py) to create the model used to generate data for visualizations.
11. Set the following environment variables below and execute the following command to run the server:
    ~~~
    FLASK_APP=flaskr
    FLASK_ENV=development
    ~~~
    ~~~
    flask run --port 5000
    ~~~
12. Go back to [TSNE-Visualization](https://github.com/dlordtemplar/TSNE-Visualization) to complete setup.